#!/usr/bin/env python3
"""
MoCap-only waypoint navigation for Crazyflie using OptiTrack + cflib.
No flow deck required — MoCap feeds the firmware EKF via extpos.

Commands in terminal:
  > x y z       fly to position (e.g.  1.0 0.0 0.5)
  > home        return to takeoff position
  > status      print current position vs target
  > land        land and quit
"""

import time
import threading
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from NatNetClient import NatNetClient

logging.basicConfig(level=logging.ERROR)

# ── Config ───────────────────────────────────────────────────────────────────
URI            = "radio://0/80/2M/E7E7E7E701"
DEFAULT_HEIGHT =  0.5
MAX_SPEED      =  0.3
LOOP_HZ        =  50
EXTPOS_HZ      =  100
ARRIVAL_RADIUS =  0.1   # metres

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
        self.target_x    = 0.0
        self.target_y    = 0.0
        self.target_z    = DEFAULT_HEIGHT
        self.should_land = False

nav = NavState()

# ── MoCap callback ────────────────────────────────────────────────────────────
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
    dt = 1.0 / EXTPOS_HZ
    while not stop_event.is_set():
        t0 = time.time()
        with pose.lock:
            x, y, z = pose.x, pose.y, pose.z
            valid   = pose.valid
        if valid:
            scf.cf.extpos.send_extpos(x, y, z)
        elapsed = time.time() - t0
        time.sleep(max(0, dt - elapsed))

# ── Takeoff ───────────────────────────────────────────────────────────────────
def takeoff(scf, target_z):
    cf = scf.cf
    print(f"[CF]     Taking off to z={target_z}m ...")

    for _ in range(10):
        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
        time.sleep(0.01)

    start = time.time()
    while True:
        with pose.lock:
            cz = pose.z
        if cz >= target_z * 0.90:
            print(f"[CF]     Reached z={cz:.3f}m — switching to PID hold.")
            break
        if time.time() - start > 6.0:
            print("[CF]     WARNING: Takeoff timeout — continuing anyway.")
            break
        cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
        time.sleep(0.02)

# ── Terminal input thread ─────────────────────────────────────────────────────
def input_thread(home_x, home_y):
    print("\n  Commands:")
    print("    x y z    — fly to position  (e.g.  1.0 0.0 0.5)")
    print("    home     — return to takeoff position")
    print("    status   — print current position vs target")
    print("    land     — land and quit")
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
                nav.target_x = home_x
                nav.target_y = home_y
                nav.target_z = DEFAULT_HEIGHT
            print(f"  [NAV]  → Home ({home_x:+.3f}, {home_y:+.3f}, {DEFAULT_HEIGHT})")
            continue

        if raw.lower() == "status":
            with pose.lock:
                px, py, pz = pose.x, pose.y, pose.z
            with nav.lock:
                tx, ty, tz = nav.target_x, nav.target_y, nav.target_z
            print(f"  [POS]  Current : ({px:+.3f}, {py:+.3f}, {pz:+.3f})")
            print(f"  [TGT]  Target  : ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            continue

        parts = raw.split()
        if len(parts) == 3:
            try:
                tx, ty, tz = float(parts[0]), float(parts[1]), float(parts[2])
                with nav.lock:
                    nav.target_x = tx
                    nav.target_y = ty
                    nav.target_z = tz
                print(f"  [NAV]  → Waypoint ({tx:+.3f}, {ty:+.3f}, {tz:+.3f})")
            except ValueError:
                print("  [ERR]  Bad input. Use:  x y z  (e.g.  1.0 0.5 0.3)")
        else:
            print("  [ERR]  Unknown command. Type  x y z  or  land / home / status")

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
        home_x, home_y = pose.x, pose.y
        print(f"[MoCap]  First frame: x={pose.x:.3f}  y={pose.y:.3f}  z={pose.z:.3f}")

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

    stop_extpos = threading.Event()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # 3. Set Kalman estimator + disable flow deck
        cf.param.set_value('stabilizer.estimator', '2')
        time.sleep(0.5)
        try:
            cf.param.set_value('flowdeck.useFlow', '0')
        except Exception:
            pass

        # 4. Start extpos thread
        ep_thread = threading.Thread(
            target=extpos_thread, args=(scf, stop_extpos), daemon=True)
        ep_thread.start()
        print("[EKF]    extpos thread started.")
        time.sleep(1.0)

        # 5. Takeoff
        takeoff(scf, DEFAULT_HEIGHT)
        pid_x.reset(); pid_y.reset(); pid_z.reset()

        # 6. Start input thread
        it = threading.Thread(
            target=input_thread, args=(home_x, home_y), daemon=True)
        it.start()

        print(f"[CTRL]   Flying. Type waypoints in terminal.")
        dt = 1.0 / LOOP_HZ

        try:
            while True:
                loop_start = time.time()

                with nav.lock:
                    if nav.should_land:
                        break
                    tx, ty, tz = nav.target_x, nav.target_y, nav.target_z

                with pose.lock:
                    cx, cy, cz = pose.x, pose.y, pose.z
                    got_data   = pose.valid

                if not got_data:
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    time.sleep(dt)
                    continue

                now = time.time()
                ex = tx - cx
                ey = ty - cy
                ez = tz - cz

                vx = clamp(pid_x.update(ex, now), MAX_SPEED)
                vy = clamp(pid_y.update(ey, now), MAX_SPEED)
                vz = clamp(pid_z.update(ez, now), MAX_SPEED)

                cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0)

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

        # Land
        print("\n[CF]     Descending...")
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