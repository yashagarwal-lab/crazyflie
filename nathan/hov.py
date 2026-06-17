#!/usr/bin/env python3
"""
MoCap-only hover for Crazyflie using OptiTrack + cflib.
No flow deck required — MoCap feeds the firmware EKF via extpos.

Flow:
  1. Stream MoCap → send extpos to firmware EKF at 100 Hz
  2. Firmware EKF uses extpos for velocity estimation
  3. Your PID runs on MoCap position → sends velocity world setpoints
  4. Press Ctrl+C to land
"""

import time
import threading
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from NatNetClient import NatNetClient

logging.basicConfig(level=logging.ERROR)

# ── Config ───────────────────────────────────────────────────────────────────
URI            = "radio://0/80/2M/E7E7E7E701" 
TARGET_X       =  0.0
TARGET_Y       =  0.0
TARGET_Z       =  0.5    # hover height in metres
MAX_SPEED      =  0.3    # m/s
LOOP_HZ        =  50
EXTPOS_HZ      =  100    # must be >= 100 for EKF
TAKEOFF_TIME   =  3.0    # seconds to ramp up to hover height
TAKEOFF_THRUST =  30000  # raw thrust during takeoff ramp (tune per battery)

# ── PID gains ────────────────────────────────────────────────────────────────
KP_XY, KI_XY, KD_XY = 0.6, 0.05, 0.15
KP_Z,  KI_Z,  KD_Z  = 0.8, 0.08, 0.20

# ── Shared MoCap state ───────────────────────────────────────────────────────
class PoseState:
    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.valid = False
        self.frame_count = 0

pose = PoseState()

def labeled_marker_callback(marker_id, pos):
    with pose.lock:
        pose.y, pose.z, pose.x = pos
        pose.frame_count += 1
        pose.valid = True

# ── PID ──────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error, now):
        dt = (now - self._prev_time) if self._prev_time else 0.0
        self._prev_time = now
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit,
                                 self._integral + error * dt))
        deriv = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * deriv

    def reset(self):
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

def clamp(v, lim):
    return max(-lim, min(lim, v))

# ── extpos sender thread ──────────────────────────────────────────────────────
def extpos_thread(scf, stop_event):
    """
    Sends MoCap position into firmware EKF at EXTPOS_HZ.
    This replaces the flow deck as the EKF velocity/position source.
    """
    dt = 1.0 / EXTPOS_HZ
    while not stop_event.is_set():
        t0 = time.time()
        with pose.lock:
            x, y, z  = pose.x, pose.y, pose.z
            valid     = pose.valid
        if valid:
            scf.cf.extpos.send_extpos(x, y, z)
        elapsed = time.time() - t0
        time.sleep(max(0, dt - elapsed))

# ── Takeoff using raw thrust then switch to velocity control ──────────────────
def takeoff(scf, target_z):
    """
    Ramp thrust until MoCap z reaches target_z, then hand off to PID.
    """
    cf = scf.cf
    print(f"[CF]     Taking off to z={target_z}m ...")

    # Unlock motors (send a few zero setpoints first)
    for _ in range(10):
        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
        time.sleep(0.01)

    start = time.time()
    while True:
        with pose.lock:
            cz = pose.z

        if cz >= target_z * 0.90:   # 90% of target height reached
            print(f"[CF]     Reached z={cz:.3f}m — switching to PID hold.")
            break

        if time.time() - start > 6.0:
            print("[CF]     WARNING: Takeoff timeout — continuing anyway.")
            break

        # Positive vz to climb
        cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
        time.sleep(0.02)

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    # 1. Start MoCap
    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.labeledMarkerListener = labeled_marker_callback
    client.run()
    print("[MoCap]  Waiting for first frame...")

    timeout = time.time() + 5.0
    while not pose.valid:
        if time.time() > timeout:
            print("[MoCap]  ERROR: No data. Is Motive streaming?")
            client.stop(); return
        time.sleep(0.05)

    with pose.lock:
        print(f"[MoCap]  First frame: x={pose.x:.3f}  y={pose.y:.3f}  z={pose.z:.3f}")

    # 2. Connect Crazyflie
    cflib.crtp.init_drivers()
    print(f"[CF]     Connecting to {URI} ...")

    pid_x = PID(KP_XY, KI_XY, KD_XY)
    pid_y = PID(KP_XY, KI_XY, KD_XY)
    pid_z = PID(KP_Z,  KI_Z,  KD_Z)

    stop_extpos = threading.Event()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # 3. Set estimator to Kalman (required for extpos)
        cf.param.set_value('stabilizer.estimator', '2')
        time.sleep(0.5)

        # 4. Disable flow deck in firmware so it doesn't interfere
        #    (only needed if deck is physically absent; safe to send either way)
        cf.param.set_value('ring.effect', '0')   # just cosmetic, ignore errors
        try:
            cf.param.set_value('flowdeck.useFlow', '0')
        except Exception:
            pass   # param may not exist if deck is absent — that's fine

        # 5. Start extpos thread
        ep_thread = threading.Thread(
            target=extpos_thread, args=(scf, stop_extpos), daemon=True)
        ep_thread.start()
        print("[EKF]    extpos thread started — feeding MoCap into firmware EKF.")
        time.sleep(1.0)   # let EKF converge on position

        # 6. Takeoff
        takeoff(scf, TARGET_Z)
        pid_x.reset(); pid_y.reset(); pid_z.reset()

        print(f"[CTRL]   Holding at target x={TARGET_X} y={TARGET_Y} z={TARGET_Z}")
        print("         Press Ctrl+C to land.\n")

        dt = 1.0 / LOOP_HZ

        try:
            while True:
                loop_start = time.time()

                with pose.lock:
                    cx, cy, cz = pose.x, pose.y, pose.z
                    got_data   = pose.valid

                if not got_data:
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    time.sleep(dt)
                    continue

                now = time.time()
                ex = TARGET_X - cx
                ey = TARGET_Y - cy
                ez = TARGET_Z - cz

                vx = clamp(pid_x.update(ex, now), MAX_SPEED)
                vy = clamp(pid_y.update(ey, now), MAX_SPEED)
                vz = clamp(pid_z.update(ez, now), MAX_SPEED)

                cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0)

                dist = (ex**2 + ey**2 + ez**2) ** 0.5
                print(f"\r  pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                      f"err=({ex:+.3f},{ey:+.3f},{ez:+.3f})  "
                      f"dist={dist:.3f}m  ",
                      end="", flush=True)

                elapsed = time.time() - loop_start
                time.sleep(max(0, dt - elapsed))

        except KeyboardInterrupt:
            print("\n[CTRL]   Ctrl+C — landing...")

        # Land: descend slowly then kill
        print("[CF]     Descending...")
        with pose.lock:
            cz = pose.z
        while cz > 0.10:
            cf.commander.send_velocity_world_setpoint(0, 0, -0.2, 0)
            time.sleep(0.05)
            with pose.lock:
                cz = pose.z
        cf.commander.send_stop_setpoint()
        time.sleep(0.5)

        stop_extpos.set()

    print("[CF]     Landed and disconnected.")
    client.stop()
    print("[MoCap]  Done.")

if __name__ == '__main__':
    main()