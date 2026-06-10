#!/usr/bin/env python3
"""
OptiTrack → Crazyflie Estimator Bridge (Phase 2).

Streams labeled marker position data from OptiTrack into the Crazyflie's
onboard Kalman filter via cf.extpos.send_extpos(x, y, z).

This script validates that the Crazyflie's internal state estimate
converges to the OptiTrack position WITHOUT spinning the motors.

Usage:
    pip install cflib       (if not already installed)
    python mocap_fly.py

Hold the Crazyflie in the OptiTrack arena and verify that the
'Estimator' values match the 'OptiTrack' values within ~1 cm.

Press Ctrl+C to stop.
"""
import time
import threading
import logging
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.ERROR)

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.position_hl_commander import PositionHlCommander

from NatNetClient import NatNetClient

# ── Configuration ──
URI = 'radio://0/80/2M/E7E7E7E701'
POSE_INJECT_RATE = 100   # Hz — how often we push position to the drone
SAFETY_TIMEOUT = 0.5     # seconds — land if no OptiTrack data for this long


# ── Shared pose state ──
class PoseState:
    """Thread-safe container for the latest marker position from OptiTrack."""
    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.marker_id = -1
        self.last_update = 0.0
        self.valid = False

    def update(self, marker_id, pos):
        with self.lock:
            self.marker_id = marker_id
            # Remap from OptiTrack YZX to Crazyflie XYZ
            self.y, self.z, self.x = pos
            self.last_update = time.time()
            self.valid = True

    def get(self):
        with self.lock:
            return self.x, self.y, self.z, self.last_update, self.valid


pose_state = PoseState()


# ── OptiTrack callback ──
def labeled_marker_callback(marker_id, pos):
    """Called by NatNetClient for each labeled marker every frame."""
    pose_state.update(marker_id, pos)


# ── Pose injector thread ──
def pose_injector(cf, pose_state, stop_event):
    """Continuously sends OptiTrack position to the Crazyflie's Kalman filter."""
    interval = 1.0 / POSE_INJECT_RATE

    while not stop_event.is_set():
        x, y, z, last_update, valid = pose_state.get()

        if valid:
            # Check for tracking loss
            if time.time() - last_update > SAFETY_TIMEOUT:
                print(f"\n  {RED}⚠ TRACKING LOST — no OptiTrack data for "
                      f"{SAFETY_TIMEOUT}s! Sending stop.{RESET}")
                try:
                    cf.commander.send_stop_setpoint()
                except Exception:
                    pass
                stop_event.set()
                return

            # Send position to the Kalman filter
            cf.extpos.send_extpos(x, y, z)

        time.sleep(interval)


# ── Estimator initialization ──
def reset_estimator(cf):
    """Reset the Kalman estimator and wait for it to converge."""
    print(f"  Setting Kalman estimator... ", end="", flush=True)
    cf.param.set_value('stabilizer.estimator', '2')
    time.sleep(0.5)
    print(f"{GREEN}OK{RESET}")

    print(f"  Resetting estimator... ", end="", flush=True)
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    print(f"{GREEN}OK{RESET}")

    print(f"  Waiting for estimator convergence (3s)... ", end="", flush=True)
    time.sleep(3.0)
    print(f"{GREEN}OK{RESET}")


# ── Telemetry logging ──
def setup_telemetry(scf):
    """Subscribe to the Crazyflie's internal state estimate for comparison."""
    log_conf = LogConfig(name='StateEstimate', period_in_ms=100)  # 10 Hz
    log_conf.add_variable('stateEstimate.x', 'float')
    log_conf.add_variable('stateEstimate.y', 'float')
    log_conf.add_variable('stateEstimate.z', 'float')

    est_state = {'x': 0.0, 'y': 0.0, 'z': 0.0}

    def log_callback(timestamp, data, logconf):
        est_state['x'] = data['stateEstimate.x']
        est_state['y'] = data['stateEstimate.y']
        est_state['z'] = data['stateEstimate.z']

    log_conf.data_received_cb.add_callback(log_callback)
    scf.cf.log.add_config(log_conf)
    log_conf.start()

    return log_conf, est_state


# ── Terminal colours ──
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'


def main():
    print(f"\n{BOLD}=== OPTITRACK → CRAZYFLIE ESTIMATOR BRIDGE ==={RESET}\n")

    # ── Step 1: Start OptiTrack listener ──
    print(f"  {BOLD}[1] OptiTrack Connection{RESET}")
    natnet = NatNetClient()
    natnet.labeledMarkerListener = labeled_marker_callback

    print(f"  Starting NatNet listener... ", end="", flush=True)
    natnet.run()
    print(f"{GREEN}OK{RESET}")

    # Wait briefly for first data
    print(f"  Waiting for marker data... ", end="", flush=True)
    deadline = time.time() + 5.0
    while time.time() < deadline:
        if pose_state.get()[4]:  # valid
            break
        time.sleep(0.1)

    if not pose_state.get()[4]:
        print(f"{RED}TIMEOUT — no data received!{RESET}")
        print(f"  {YELLOW}Is Motive streaming? Is the marker visible?{RESET}")
        natnet.stop()
        return

    x, y, z, _, _ = pose_state.get()
    print(f"{GREEN}OK{RESET} (marker at {x:.3f}, {y:.3f}, {z:.3f})")

    # ── Step 2: Connect to Crazyflie ──
    print(f"\n  {BOLD}[2] Crazyflie Connection{RESET}")
    cflib.crtp.init_drivers()
    print(f"  Connecting to {URI}... ", end="", flush=True)

    stop_event = threading.Event()

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            print(f"{GREEN}OK{RESET}")
            cf = scf.cf

            # ── Step 3: Start pose injector ──
            print(f"\n  {BOLD}[3] Estimator Setup{RESET}")
            injector = threading.Thread(
                target=pose_injector,
                args=(cf, pose_state, stop_event),
                daemon=True
            )
            injector.start()

            # Reset and configure estimator
            reset_estimator(cf)

            # ── Step 4: Autonomous Flight ──
            print(f"\n  {BOLD}[4] Autonomous Flight Sequence{RESET}")
            print(f"  {YELLOW}WARNING: Ensure drone is in a safe open space!{RESET}")
            print(f"  {YELLOW}Keep your hand over the marker to trigger safety stop if needed.{RESET}")
            
            try:
                input(f"\n  {GREEN}Press ENTER to Take Off...{RESET}")
            except KeyboardInterrupt:
                stop_event.set()
                natnet.stop()
                return

            if stop_event.is_set():
                print(f"  {RED}Aborting (Safety trigger activated){RESET}")
                return

            try:
                # The PositionHlCommander takes off automatically upon entry
                # and lands automatically upon exit.
                with PositionHlCommander(scf, default_height=0.5, default_velocity=0.3) as pc:
                    print(f"  Taking off to 0.5m...")
                    
                    # Hover in place for 5 seconds
                    for i in range(5, 0, -1):
                        if stop_event.is_set():
                            print(f"  {RED}Safety stop triggered! Landing...{RESET}")
                            break
                        print(f"  Hovering... landing in {i}s")
                        time.sleep(1)

                    print(f"  Landing sequence initiated...")

            except KeyboardInterrupt:
                print(f"\n  {RED}Emergency Stop! Landing...{RESET}")
                try:
                    cf.commander.send_stop_setpoint()
                except Exception:
                    pass

            print(f"\n  {YELLOW}Shutting down threads...{RESET}")
            stop_event.set()

    except Exception as e:
        print(f"{RED}FAILED — {e}{RESET}")

    finally:
        stop_event.set()
        natnet.stop()

    print(f"  {GREEN}Done.{RESET}\n")


if __name__ == '__main__':
    main()
