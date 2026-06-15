#!/usr/bin/env python3
"""
2-drone MoCap-only waypoint navigation using OptiTrack + cflib.
No flow deck required — each drone gets extpos from its own marker.

Controls:
  SPACEBAR  — Emergency kill (cuts motors instantly, drone drops)
  Ctrl+C    — Graceful land both drones

Terminal commands (mid-flight):
  1 x y z   — send Drone1 to position  (e.g.  1 1.0 0.0 0.5)
  2 x y z   — send Drone2 to position  (e.g.  2 0.5 0.5 1.0)
  1 home    — send Drone1 home
  2 home    — send Drone2 home
  status    — print both drones current position and target
  land      — graceful land both drones
  land1     — land only Drone1
  land2     — land only Drone2

Safety features:
  - Pre-flight position sanity check (uses remapped coordinates)
  - Hard altitude limit (MAX_ALTITUDE)
  - Kill event propagates to both drones instantly
  - Collision proximity warning

Coordinate system (after remap):
  Raw NatNet pos tuple = (OptiTrack_X, OptiTrack_Y, OptiTrack_Z)
  OptiTrack_Y = up axis (height)
  Remap: cf_x = pos[2], cf_y = pos[0], cf_z = pos[1]
  So scan.py x=+1.098 y=+0.864 z=+0.042 matches flight code exactly.
"""

import time
import threading
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from NatNetClient import NatNetClient
from pynput import keyboard

logging.basicConfig(level=logging.ERROR)

# ── Config ───────────────────────────────────────────────────────────────────
DRONE1_URI      = "radio://0/80/2M/E7E7E7E701"
DRONE2_URI      = "radio://0/80/2M/E7E7E7E702"

# Run scan.py before every flight to confirm these IDs
DRONE1_MARKER   = 50130
DRONE2_MARKER   = 50135

# Initial hover height — XY is auto-set from MoCap ground position at startup
DRONE1_TARGET_Z =  0.25      # 50 cm
DRONE2_TARGET_Z =  1.25      # 100 cm

MAX_SPEED         =  0.3
LOOP_HZ           =  50
EXTPOS_HZ         =  100
MAX_ALTITUDE      =  1.75    # hard ceiling metres — kills both if exceeded
ARRIVAL_RADIUS    =  0.08   # metres — close enough to waypoint
COLLISION_RADIUS  =  0.4    # metres — warn if drones get this close

# ── PID gains ────────────────────────────────────────────────────────────────
KP_XY, KI_XY, KD_XY = 0.6, 0.05, 0.15
KP_Z,  KI_Z,  KD_Z  = 0.8, 0.08, 0.20

# ── Shared events ─────────────────────────────────────────────────────────────
kill_event  = threading.Event()
stop1_event = threading.Event()
stop2_event = threading.Event()

# ── Per-drone pose state ──────────────────────────────────────────────────────
class PoseState:
    def __init__(self, name):
        self.name  = name
        self.lock  = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.valid = False

pose1 = PoseState("Drone1")
pose2 = PoseState("Drone2")

marker_to_pose = {}

# ── Per-drone nav state ───────────────────────────────────────────────────────
class NavState:
    def __init__(self, init_z):
        self.lock        = threading.Lock()
        self.target_x    = 0.0    # overwritten from MoCap before takeoff
        self.target_y    = 0.0
        self.target_z    = init_z
        self.should_land = False

nav1 = NavState(DRONE1_TARGET_Z)
nav2 = NavState(DRONE2_TARGET_Z)

# ── NatNet callback ───────────────────────────────────────────────────────────
def labeled_marker_callback(marker_id, pos):
    """
    Raw NatNet pos = (OptiTrack_X, OptiTrack_Y, OptiTrack_Z)
    OptiTrack_Y is the up axis.
    Remap to Crazyflie frame:
      cf_x = pos[2]  (OptiTrack Z)
      cf_y = pos[0]  (OptiTrack X)
      cf_z = pos[1]  (OptiTrack Y = height)
    """
    if marker_id in marker_to_pose:
        p = marker_to_pose[marker_id]
        with p.lock:
            p.x = pos[2]    # OptiTrack Z → Crazyflie X
            p.y = pos[0]    # OptiTrack X → Crazyflie Y
            p.z = pos[1]    # OptiTrack Y → Crazyflie Z (height)
            p.valid = True

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
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

def clamp(v, lim):
    return max(-lim, min(lim, v))

# ── Emergency kill listener ───────────────────────────────────────────────────
def on_press(key):
    if hasattr(key, 'char') and key.char == '\x18':
        print("\n\n  !! EMERGENCY KILL — CTRL+X PRESSED !!")
        kill_event.set()

def start_kill_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

# ── extpos thread ─────────────────────────────────────────────────────────────
def extpos_thread(scf, pose, stop_event):
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
def takeoff(scf, pose, target_z, name):
    cf = scf.cf
    print(f"[{name}]  Taking off to z={target_z}m ...")
    for _ in range(10):
        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
        time.sleep(0.01)
    start = time.time()
    while True:
        if kill_event.is_set():
            cf.commander.send_stop_setpoint()
            print(f"[{name}]  Kill during takeoff — motors cut.")
            return False
        with pose.lock:
            cz = pose.z
        if cz > MAX_ALTITUDE:
            print(f"[{name}]  ALTITUDE LIMIT HIT DURING TAKEOFF — killing!")
            kill_event.set()
            cf.commander.send_stop_setpoint()
            return False
        if cz >= target_z * 0.90:
            print(f"[{name}]  Reached z={cz:.3f}m — PID taking over.")
            return True
        if time.time() - start > 8.0:
            print(f"[{name}]  WARNING: Takeoff timeout — continuing anyway.")
            return True
        cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
        time.sleep(0.02)

# ── Land ──────────────────────────────────────────────────────────────────────
def land(scf, pose, name):
    cf = scf.cf
    print(f"\n[{name}]  Descending...")
    with pose.lock:
        cz = pose.z
    while cz > 0.10:
        if kill_event.is_set():
            break
        cf.commander.send_velocity_world_setpoint(0, 0, -0.2, 0)
        time.sleep(0.05)
        with pose.lock:
            cz = pose.z
    cf.commander.send_stop_setpoint()
    time.sleep(0.5)
    print(f"[{name}]  Landed.")

# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check():
    """
    Prints remapped Crazyflie-frame coordinates (same as scan.py output)
    and asks user to visually confirm they match physical drone positions.
    """
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         PRE-FLIGHT SANITY CHECK          ║")
    print("  ╚══════════════════════════════════════════╝")
    with pose1.lock:
        p1x, p1y, p1z = pose1.x, pose1.y, pose1.z
    with pose2.lock:
        p2x, p2y, p2z = pose2.x, pose2.y, pose2.z

    print(f"\n  Drone1 marker (ID {DRONE1_MARKER}) — remapped Crazyflie frame:")
    print(f"    x={p1x:+.3f}  y={p1y:+.3f}  z={p1z:+.3f}  (height={p1z:.3f}m)")
    print(f"\n  Drone2 marker (ID {DRONE2_MARKER}) — remapped Crazyflie frame:")
    print(f"    x={p2x:+.3f}  y={p2y:+.3f}  z={p2z:+.3f}  (height={p2z:.3f}m)")
    print()
    print("  These should match your scan.py output exactly.")
    print("  Look at where each drone is physically sitting and confirm.\n")

    confirm = input("  Do these positions match the physical drones? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("\n  [ABORT] Sanity check failed.")
        print("  Run scan.py to get correct IDs, update")
        print("  DRONE1_MARKER and DRONE2_MARKER, then try again.\n")
        return False

    print("  [OK] Sanity check passed — proceeding to flight.\n")
    return True

# ── Terminal input thread ─────────────────────────────────────────────────────
def input_thread(home1_x, home1_y, home2_x, home2_y, status_lock):
    print("\n  ── Waypoint Commands ──────────────────────────────────────")
    print("    1 x y z   — Drone1 to position  (e.g.  1 1.0 0.5 0.5)")
    print("    2 x y z   — Drone2 to position  (e.g.  2 -0.5 0.0 1.0)")
    print("    1 home    — Drone1 return to takeoff spot")
    print("    2 home    — Drone2 return to takeoff spot")
    print("    status    — print positions and targets of both drones")
    print("    land      — graceful land both drones")
    print("    land1     — land Drone1 only")
    print("    land2     — land Drone2 only")
    print("  ────────────────────────────────────────────────────────────\n")

    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            break

        if not raw:
            continue

        # ── land / land1 / land2 ─────────────────────────────────────────────
        if raw.lower() == "land":
            with nav1.lock: nav1.should_land = True
            with nav2.lock: nav2.should_land = True
            print("  [NAV]  Landing both drones.")
            break

        if raw.lower() == "land1":
            with nav1.lock: nav1.should_land = True
            print("  [NAV]  Landing Drone1.")
            continue

        if raw.lower() == "land2":
            with nav2.lock: nav2.should_land = True
            print("  [NAV]  Landing Drone2.")
            continue

        # ── status ───────────────────────────────────────────────────────────
        if raw.lower() == "status":
            with pose1.lock:
                p1x, p1y, p1z = pose1.x, pose1.y, pose1.z
            with pose2.lock:
                p2x, p2y, p2z = pose2.x, pose2.y, pose2.z
            with nav1.lock:
                t1x, t1y, t1z = nav1.target_x, nav1.target_y, nav1.target_z
            with nav2.lock:
                t2x, t2y, t2z = nav2.target_x, nav2.target_y, nav2.target_z
            sep = ((p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2) ** 0.5
            print(f"\n  [Drone1] pos=({p1x:+.3f},{p1y:+.3f},{p1z:+.3f})  "
                  f"tgt=({t1x:+.3f},{t1y:+.3f},{t1z:+.3f})")
            print(f"  [Drone2] pos=({p2x:+.3f},{p2y:+.3f},{p2z:+.3f})  "
                  f"tgt=({t2x:+.3f},{t2y:+.3f},{t2z:+.3f})")
            print(f"  [SEP]    drone separation = {sep:.3f}m\n")
            continue

        parts = raw.split()

        # ── 1 home / 2 home ──────────────────────────────────────────────────
        if len(parts) == 2 and parts[1].lower() == "home":
            if parts[0] == "1":
                with nav1.lock:
                    nav1.target_x = home1_x
                    nav1.target_y = home1_y
                    nav1.target_z = DRONE1_TARGET_Z
                print(f"  [NAV]  Drone1 → Home ({home1_x:+.3f},{home1_y:+.3f},{DRONE1_TARGET_Z})")
            elif parts[0] == "2":
                with nav2.lock:
                    nav2.target_x = home2_x
                    nav2.target_y = home2_y
                    nav2.target_z = DRONE2_TARGET_Z
                print(f"  [NAV]  Drone2 → Home ({home2_x:+.3f},{home2_y:+.3f},{DRONE2_TARGET_Z})")
            else:
                print("  [ERR]  Use  1 home  or  2 home")
            continue

        # ── 1 x y z / 2 x y z ────────────────────────────────────────────────
        if len(parts) == 4:
            drone_id = parts[0]
            try:
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                print("  [ERR]  Bad numbers. Use:  1 x y z  or  2 x y z")
                continue

            # Reject targets above altitude limit
            if tz > MAX_ALTITUDE:
                print(f"  [ERR]  Target z={tz}m exceeds MAX_ALTITUDE={MAX_ALTITUDE}m — rejected.")
                continue

            # Reject negative altitude
            if tz < 0.1:
                print(f"  [ERR]  Target z={tz}m too low — minimum is 0.1m.")
                continue

            # Get other drone position for collision check
            if drone_id == "1":
                with pose2.lock:
                    ox, oy, oz = pose2.x, pose2.y, pose2.z
            elif drone_id == "2":
                with pose1.lock:
                    ox, oy, oz = pose1.x, pose1.y, pose1.z
            else:
                print("  [ERR]  First number must be 1 or 2")
                continue

            # Collision proximity warning
            sep = ((tx-ox)**2 + (ty-oy)**2 + (tz-oz)**2) ** 0.5
            if sep < COLLISION_RADIUS:
                print(f"  [WARN] Target is only {sep:.3f}m from the other drone!")
                confirm = input("         Send anyway? (yes/no): ").strip().lower()
                if confirm != 'yes':
                    print("  [NAV]  Waypoint cancelled.")
                    continue

            if drone_id == "1":
                with nav1.lock:
                    nav1.target_x = tx
                    nav1.target_y = ty
                    nav1.target_z = tz
                print(f"  [NAV]  Drone1 → ({tx:+.3f},{ty:+.3f},{tz:+.3f})")
            else:
                with nav2.lock:
                    nav2.target_x = tx
                    nav2.target_y = ty
                    nav2.target_z = tz
                print(f"  [NAV]  Drone2 → ({tx:+.3f},{ty:+.3f},{tz:+.3f})")
            continue

        print("  [ERR]  Unknown command. Type  1 x y z  /  2 x y z  /  status  /  land")

# ── Per-drone flight loop ─────────────────────────────────────────────────────
def drone_flight_loop(scf, pose, nav, name, stop_event, status_lock):
    cf = scf.cf

    # Kalman estimator + disable flow deck
    cf.param.set_value('stabilizer.estimator', '2')
    time.sleep(0.5)
    try:
        cf.param.set_value('flowdeck.useFlow', '0')
    except Exception:
        pass

    # Start extpos FIRST so MoCap data is flowing before EKF reset
    stop_ep = threading.Event()
    ep = threading.Thread(
        target=extpos_thread, args=(scf, pose, stop_ep), daemon=True)
    ep.start()
    print(f"[{name}]  extpos started. Waiting for MoCap data to establish...")
    time.sleep(1.0)    # let extpos establish before reset

    # NOW reset EKF — it restarts with live MoCap data immediately
    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    print(f"[{name}]  EKF reset. Waiting for convergence...")
    time.sleep(1.5)    # let EKF converge on extpos from clean state

    # Takeoff
    with nav.lock:
        init_z = nav.target_z
    success = takeoff(scf, pose, init_z, name)
    if not success or kill_event.is_set():
        stop_ep.set()
        return

    pid_x = PID(KP_XY, KI_XY, KD_XY)
    pid_y = PID(KP_XY, KI_XY, KD_XY)
    pid_z = PID(KP_Z,  KI_Z,  KD_Z)
    pid_x.reset(); pid_y.reset(); pid_z.reset()

    print(f"[{name}]  Ready for waypoints.")

    dt = 1.0 / LOOP_HZ

    try:
        while not stop_event.is_set() and not kill_event.is_set():
            loop_start = time.time()

            with nav.lock:
                if nav.should_land:
                    break
                tx, ty, tz = nav.target_x, nav.target_y, nav.target_z

            with pose.lock:
                cx, cy, cz = pose.x, pose.y, pose.z
                got_data   = pose.valid

            # Hard altitude limit
            if cz > MAX_ALTITUDE:
                print(f"\n[{name}]  !! ALTITUDE LIMIT {MAX_ALTITUDE}m EXCEEDED "
                      f"(z={cz:.3f}m) — KILLING ALL MOTORS !!")
                kill_event.set()
                break

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

            with status_lock:
                print(f"  [{name}] pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                      f"tgt=({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                      f"dist={dist:.3f}m  "
                      f"{'[ARRIVED]' if arrived else '         '}")

            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

    except Exception as e:
        print(f"[{name}]  Flight loop error: {e}")

    # Exit path
    if kill_event.is_set():
        cf.commander.send_stop_setpoint()
        print(f"[{name}]  Motors killed instantly.")
    else:
        land(scf, pose, name)

    stop_ep.set()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    marker_to_pose[DRONE1_MARKER] = pose1
    marker_to_pose[DRONE2_MARKER] = pose2

    # Start MoCap
    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.labeledMarkerListener = labeled_marker_callback
    client.run()
    print("[MoCap]  Waiting for both markers...")

    timeout = time.time() + 8.0
    while True:
        with pose1.lock: v1 = pose1.valid
        with pose2.lock: v2 = pose2.valid
        if v1 and v2:
            break
        if time.time() > timeout:
            print("[MoCap]  ERROR: Could not see both markers.")
            print("         Run scan.py to check your IDs.")
            client.stop()
            return
        time.sleep(0.05)

    # Sanity check — uses remapped coordinates matching scan.py output
    if not sanity_check():
        client.stop()
        return

    # Save actual home positions from live MoCap (already remapped)
    with pose1.lock:
        home1_x, home1_y = pose1.x, pose1.y
    with pose2.lock:
        home2_x, home2_y = pose2.x, pose2.y

    # Set nav targets to actual ground XY so drone hovers in place on takeoff
    with nav1.lock:
        nav1.target_x = home1_x
        nav1.target_y = home1_y
    with nav2.lock:
        nav2.target_x = home2_x
        nav2.target_y = home2_y

    print(f"[NAV]    Drone1 will hover above x={home1_x:+.3f} y={home1_y:+.3f} z={DRONE1_TARGET_Z}")
    print(f"[NAV]    Drone2 will hover above x={home2_x:+.3f} y={home2_y:+.3f} z={DRONE2_TARGET_Z}")

    cflib.crtp.init_drivers()
    print("[CF]     Connecting to both drones...")

    status_lock = threading.Lock()
    start_kill_listener()

    print("\n  ========================================")
    print("  CTRL+X   = Emergency kill (motors cut)")
    print("  Ctrl+C   = Graceful land both drones  ")
    print(f"  Altitude limit = {MAX_ALTITUDE}m        ")
    print("  ========================================\n")

    try:
        with SyncCrazyflie(DRONE1_URI, cf=Crazyflie(rw_cache='./cache1')) as scf1, \
             SyncCrazyflie(DRONE2_URI, cf=Crazyflie(rw_cache='./cache2')) as scf2:

            print("[CF]     Both drones connected!\n")

            t1 = threading.Thread(
                target=drone_flight_loop,
                args=(scf1, pose1, nav1, "Drone1", stop1_event, status_lock),
                daemon=True)

            t2 = threading.Thread(
                target=drone_flight_loop,
                args=(scf2, pose2, nav2, "Drone2", stop2_event, status_lock),
                daemon=True)

            t1.start()
            time.sleep(3.0)    # Drone1 reaches height before Drone2 lifts
            t2.start()

            # Wait for both to be airborne before opening command terminal
            time.sleep(2.0)
            it = threading.Thread(
                target=input_thread,
                args=(home1_x, home1_y, home2_x, home2_y, status_lock),
                daemon=True)
            it.start()

            try:
                while t1.is_alive() or t2.is_alive():
                    if kill_event.is_set():
                        stop1_event.set()
                        stop2_event.set()
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\n[CTRL]   Ctrl+C — graceful landing both drones...")
                stop1_event.set()
                stop2_event.set()
                t1.join(timeout=10)
                t2.join(timeout=10)

    except Exception as e:
        print(f"[CF]     Connection error: {e}")

    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()