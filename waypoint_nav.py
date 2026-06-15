#!/usr/bin/env python3
"""
MoCap-guided waypoint navigation for Crazyflie using OptiTrack + cflib.

Type waypoints in terminal while drone is flying:
  > x y z        e.g.  > 1.0 0.5 0.5
  > land          to land
  > status        to print current position
  > home          to return to takeoff position
"""

import time
import threading
import logging
import sys
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander
from NatNetClient import NatNetClient

# ── Config ───────────────────────────────────────────────────────────────────
URI            = "radio://0/80/2M/E7E7E7E701"
DEFAULT_HEIGHT =  0.5    # takeoff height (metres)
MAX_SPEED      =  0.3    # m/s cap on all axes
LOOP_HZ        =  50     # controller rate
ARRIVAL_RADIUS =  0.08   # metres — "close enough" to waypoint

# ── PID gains ────────────────────────────────────────────────────────────────
KP_XY, KI_XY, KD_XY = 0.6, 0.05, 0.15
KP_Z,  KI_Z,  KD_Z  = 0.8, 0.08, 0.20

# ── Shared state ─────────────────────────────────────────────────────────────
class PoseState:
    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.valid = False
        self.frame_count = 0

pose = PoseState()

class NavState:
    def __init__(self):
        self.lock = threading.Lock()
        self.target_x   =  0.0
        self.target_y   =  0.0
        self.target_z   =  DEFAULT_HEIGHT
        self.should_land = False
        self.new_waypoint_event = threading.Event()

nav = NavState()


# ── MoCap callback ────────────────────────────────────────────────────────────
def labeled_marker_callback(marker_id, pos):
    with pose.lock:
        pose.y, pose.z, pose.x = pos
        pose.frame_count += 1
        pose.valid = True


# ── PID ───────────────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._prev_time   = None

    def update(self, error, now):
        dt = (now - self._prev_time) if self._prev_time else 0.0
        self._prev_time = now
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral + error * dt))
        deriv = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error
        return self.kp * error + self.ki * self._integral + self.kd * deriv

    def reset(self):
        self._integral = 0.0; self._prev_error = 0.0; self._prev_time = None


def clamp(v, lim):
    return max(-lim, min(lim, v))


# ── Terminal input thread ─────────────────────────────────────────────────────
def input_thread():
    """Runs in background. Reads waypoint commands from terminal."""
    print("\n  Commands:")
    print("    x y z       — fly to position  (e.g.  1.0 0.0 0.5)")
    print("    home        — return to origin (0, 0, takeoff height)")
    print("    status      — print current position")
    print("    land        — land and quit")
    print()

    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            break

        if not raw:
            continue

        if raw.lower() == "land":
            with nav.lock:
                nav.should_land = True
            print("  [NAV]  Landing command sent.")
            break

        if raw.lower() == "home":
            with nav.lock:
                nav.target_x = 0.0
                nav.target_y = 0.0
                nav.target_z = DEFAULT_HEIGHT
                nav.new_waypoint_event.set()
            print(f"  [NAV]  → Home  (0.0, 0.0, {DEFAULT_HEIGHT})")
            continue

        if raw.lower() == "status":
            with pose.lock:
                px, py, pz = pose.x, pose.y, pose.z
            with nav.lock:
                tx, ty, tz = nav.target_x, nav.target_y, nav.target_z
            print(f"  [POS]  Current : ({px:+.3f}, {py:+.3f}, {pz:+.3f})")
            print(f"  [TGT]  Target  : ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            continue

        # Try to parse "x y z"
        parts = raw.split()
        if len(parts) == 3:
            try:
                tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
                with nav.lock:
                    nav.target_x = tx
                    nav.target_y = ty
                    nav.target_z = tz
                    nav.new_waypoint_event.set()
                print(f"  [NAV]  → Waypoint ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            except ValueError:
                print("  [ERR]  Bad input. Use:  x y z  (e.g.  1.0 0.5 0.3)")
        else:
            print("  [ERR]  Unknown command. Type  x y z  or  land / home / status")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.ERROR)

    # 1. Start MoCap
    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.labeledMarkerListener = labeled_marker_callback
    client.run()
    print("[MoCap]  Streaming. Waiting for first frame...")

    timeout = time.time() + 5.0
    while not pose.valid:
        if time.time() > timeout:
            print("[MoCap]  ERROR: No data. Is Motive streaming?")
            client.stop(); return
        time.sleep(0.05)

    with pose.lock:
        home_x, home_y = pose.x, pose.y
        print(f"[MoCap]  First frame: x={pose.x:.3f}  y={pose.y:.3f}  z={pose.z:.3f}")

    # Set initial target = current XY, takeoff height
    with nav.lock:
        nav.target_x = home_x
        nav.target_y = home_y
        nav.target_z = DEFAULT_HEIGHT

    # 2. Connect Crazyflie
    cflib.crtp.init_drivers()
    print(f"[CF]     Connecting to {URI} ...")

    pid_x = PID(KP_XY, KI_XY, KD_XY)
    pid_y = PID(KP_XY, KI_XY, KD_XY)
    pid_z = PID(KP_Z,  KI_Z,  KD_Z)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("[CF]     Connected!")

        with MotionCommander(scf, default_height=DEFAULT_HEIGHT) as mc:
            print(f"[CF]     Took off to {DEFAULT_HEIGHT}m.")
            print("[CF]     Stabilising for 2s...")
            time.sleep(2.0)

            pid_x.reset(); pid_y.reset(); pid_z.reset()

            # Start terminal input in background thread
            t = threading.Thread(target=input_thread, daemon=True)
            t.start()

            dt = 1.0 / LOOP_HZ

            try:
                while True:
                    loop_start = time.time()

                    # Check land flag
                    with nav.lock:
                        if nav.should_land:
                            break
                        tx, ty, tz = nav.target_x, nav.target_y, nav.target_z

                    with pose.lock:
                        cx, cy, cz = pose.x, pose.y, pose.z
                        got_data   = pose.valid

                    if not got_data:
                        mc.stop()
                        time.sleep(dt)
                        continue

                    now = time.time()
                    ex = tx - cx
                    ey = ty - cy
                    ez = tz - cz

                    vx = clamp(pid_x.update(ex, now), MAX_SPEED)
                    vy = clamp(pid_y.update(ey, now), MAX_SPEED)
                    vz = clamp(pid_z.update(ez, now), MAX_SPEED)

                    mc.start_linear_motion(vx, vy, vz, 0.0)

                    # Arrival check
                    dist = (ex**2 + ey**2 + ez**2) ** 0.5
                    arrived = dist < ARRIVAL_RADIUS

                    print(f"\r  pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                          f"tgt=({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                          f"dist={dist:.3f}m  "
                          f"{'[ARRIVED] ' if arrived else '          '}",
                          end="", flush=True)

                    elapsed = time.time() - loop_start
                    time.sleep(max(0, dt - elapsed))

            except KeyboardInterrupt:
                print("\n[CTRL]   Ctrl+C — landing...")

            print("\n[CF]     Landing...")
            # MotionCommander.__exit__ auto-lands

    print("[CF]     Landed and disconnected.")
    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()