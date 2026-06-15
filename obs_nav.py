#!/usr/bin/env python3
"""
2-drone MoCap-only grid-based waypoint navigation with obstacle avoidance.
No flow deck required — each drone gets extpos from its own marker.

Drones always face +X axis direction.

Controls:
  CTRL+X    — Emergency kill (cuts motors instantly, drone drops)
  Ctrl+C    — Graceful land both drones

Terminal commands (mid-flight):
  1 <grid> <z>  — send Drone1 to grid position at height z  (e.g.  1 13 0.5)
  2 <grid> <z>  — send Drone2 to grid position at height z  (e.g.  2 7 1.0)
  1 home        — Drone1 return to physical takeoff spot
  2 home        — Drone2 return to physical takeoff spot
  status        — print both drones current position and target
  land          — graceful land both drones
  land1         — land only Drone1
  land2         — land only Drone2
  grid          — print the full grid map with coordinates

Obstacle avoidance:
  When a drone's path passes through the other drone and both are at
  similar height, the moving drone automatically:
    Step 1 — climbs to other drone height + 0.5m at current XY
    Step 2 — flies to target XY at avoid height
    Step 3 — descends to target height at target XY
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

# ── Grid map ──────────────────────────────────────────────────────────────────
GRID = {
     1: (-1.0, +1.0),   2: (-0.5, +1.0),   3: ( 0.0, +1.0),   4: (+0.5, +1.0),   5: (+1.0, +1.0),
     6: (-1.0, +0.5),   7: (-0.5, +0.5),   8: ( 0.0, +0.5),   9: (+0.5, +0.5),  10: (+1.0, +0.5),
    11: (-1.0,  0.0),  12: (-0.5,  0.0),  13: ( 0.0,  0.0),  14: (+0.5,  0.0),  15: (+1.0,  0.0),
    16: (-1.0, -0.5),  17: (-0.5, -0.5),  18: ( 0.0, -0.5),  19: (+0.5, -0.5),  20: (+1.0, -0.5),
    21: (-1.0, -1.0),  22: (-0.5, -1.0),  23: ( 0.0, -1.0),  24: (+0.5, -1.0),  25: (+1.0, -1.0),
}

# ── Config ───────────────────────────────────────────────────────────────────
DRONE1_URI      = "radio://0/80/2M/E7E7E7E704"
DRONE2_URI      = "radio://0/80/2M/E7E7E7E702"

DRONE1_MARKER   = 354
DRONE2_MARKER   = 51328

DRONE1_TARGET_Z =  0.5
DRONE2_TARGET_Z =  1.0

MAX_SPEED         =  0.3
LOOP_HZ           =  50
EXTPOS_HZ         =  100
MAX_ALTITUDE      =  1.75
ARRIVAL_RADIUS    =  0.08
COLLISION_RADIUS  =  0.15
HEIGHT_TOLERANCE  =  0.3    # if drones are within this in z, avoidance triggers
AVOID_OFFSET      =  0.5    # climb this much above the obstacle drone

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
        self.lock          = threading.Lock()
        self.target_x      = 0.0
        self.target_y      = 0.0
        self.target_z      = init_z
        self.should_land   = False
        # Waypoint queue — list of (x, y, z) tuples executed in order
        # When queue is empty drone holds at current target
        self.waypoint_queue = []

nav1 = NavState(DRONE1_TARGET_Z)
nav2 = NavState(DRONE2_TARGET_Z)

# ── NatNet callback ───────────────────────────────────────────────────────────
def labeled_marker_callback(marker_id, pos):
    if marker_id in marker_to_pose:
        p = marker_to_pose[marker_id]
        with p.lock:
            p.x = pos[2]
            p.y = pos[0]
            p.z = pos[1]
            p.valid = True

def receiveRigidBodyFrame(id, position, rotation):
    if id in marker_to_pose:
        p = marker_to_pose[id]
        with p.lock:
            p.x = position[2]
            p.y = position[0]
            p.z = position[1]
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

# ── Path conflict check ───────────────────────────────────────────────────────
def path_conflicts(start_x, start_y, target_x, target_y,
                   obs_x, obs_y, radius):
    """
    Check if the straight line from (start_x, start_y) to (target_x, target_y)
    passes within radius of obstacle at (obs_x, obs_y).
    Uses point-to-line-segment distance formula.
    """
    dx = target_x - start_x
    dy = target_y - start_y
    seg_len_sq = dx*dx + dy*dy

    if seg_len_sq == 0:
        # Start and target are the same point
        dist = ((obs_x - start_x)**2 + (obs_y - start_y)**2) ** 0.5
        return dist < radius

    # Project obstacle onto the line segment, clamp to [0,1]
    t = ((obs_x - start_x)*dx + (obs_y - start_y)*dy) / seg_len_sq
    t = max(0.0, min(1.0, t))

    # Closest point on segment to obstacle
    closest_x = start_x + t * dx
    closest_y = start_y + t * dy

    dist = ((obs_x - closest_x)**2 + (obs_y - closest_y)**2) ** 0.5
    return dist < radius

# ── Build waypoint list with avoidance ───────────────────────────────────────
def plan_path(moving_drone_name,
              curr_x, curr_y, curr_z,
              target_x, target_y, target_z,
              obs_x, obs_y, obs_z):
    """
    Returns a list of (x, y, z) waypoints to execute in order.
    If path is clear returns single waypoint [(target_x, target_y, target_z)].
    If conflict and heights similar returns 3-step avoidance path.
    """
    conflict = path_conflicts(curr_x, curr_y, target_x, target_y,
                              obs_x, obs_y, COLLISION_RADIUS)
    height_similar = abs(curr_z - obs_z) < HEIGHT_TOLERANCE

    if not conflict or not height_similar:
        # No avoidance needed
        return [(target_x, target_y, target_z)], False

    # Avoidance needed
    avoid_z = obs_z + AVOID_OFFSET

    # Clamp avoid_z to MAX_ALTITUDE
    avoid_z = min(avoid_z, MAX_ALTITUDE - 0.1)

    waypoints = [
        (curr_x,   curr_y,   avoid_z),    # Step 1: climb at current XY
        (target_x, target_y, avoid_z),    # Step 2: fly to target XY at avoid height
        (target_x, target_y, target_z),   # Step 3: descend to target height
    ]
    return waypoints, True

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
    time.sleep(0.3)
    try:
        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        cf.param.set_value('stabilizer.estimator', '1')
    except Exception:
        pass
    print(f"[{name}]  Landed.")

# ── Grid print ────────────────────────────────────────────────────────────────
def print_grid():
    print("\n  ── Grid Map (origin=13, spacing=0.5m) ──────────────")
    print("       x=-1.0  x=-0.5  x= 0.0  x=+0.5  x=+1.0")
    rows = [
        ("y=+1.0", [1,  2,  3,  4,  5]),
        ("y=+0.5", [6,  7,  8,  9, 10]),
        ("y= 0.0", [11, 12, 13, 14, 15]),
        ("y=-0.5", [16, 17, 18, 19, 20]),
        ("y=-1.0", [21, 22, 23, 24, 25]),
    ]
    for label, nums in rows:
        row_str = "   ".join(f"{n:>2}" for n in nums)
        print(f"  {label}   {row_str}")
    print()

# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check():
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         PRE-FLIGHT SANITY CHECK          ║")
    print("  ╚══════════════════════════════════════════╝")
    with pose1.lock:
        p1x, p1y, p1z = pose1.x, pose1.y, pose1.z
    with pose2.lock:
        p2x, p2y, p2z = pose2.x, pose2.y, pose2.z

    print(f"\n  Drone1 marker (ID {DRONE1_MARKER}):")
    print(f"    x={p1x:+.3f}  y={p1y:+.3f}  z={p1z:+.3f}  (height={p1z:.3f}m)")
    print(f"\n  Drone2 marker (ID {DRONE2_MARKER}):")
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
    print("\n  ── Commands ───────────────────────────────────────────────")
    print("    1 <grid> <z>  — Drone1 to grid pos at height  (e.g. 1 13 0.5)")
    print("    2 <grid> <z>  — Drone2 to grid pos at height  (e.g. 2 7 1.0)")
    print("    1 home        — Drone1 return to takeoff spot")
    print("    2 home        — Drone2 return to takeoff spot")
    print("    status        — print positions and targets")
    print("    grid          — print the grid map")
    print("    land          — land both drones")
    print("    land1         — land Drone1 only")
    print("    land2         — land Drone2 only")
    print("  ────────────────────────────────────────────────────────────\n")

    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            break

        if not raw:
            continue

        if raw.lower() == "grid":
            print_grid()
            continue

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

        if raw.lower() == "status":
            with pose1.lock:
                p1x, p1y, p1z = pose1.x, pose1.y, pose1.z
            with pose2.lock:
                p2x, p2y, p2z = pose2.x, pose2.y, pose2.z
            with nav1.lock:
                t1x, t1y, t1z = nav1.target_x, nav1.target_y, nav1.target_z
                q1 = len(nav1.waypoint_queue)
            with nav2.lock:
                t2x, t2y, t2z = nav2.target_x, nav2.target_y, nav2.target_z
                q2 = len(nav2.waypoint_queue)
            sep = ((p1x-p2x)**2 + (p1y-p2y)**2 + (p1z-p2z)**2) ** 0.5

            def nearest_grid(x, y):
                return min(GRID.items(),
                           key=lambda g: (g[1][0]-x)**2 + (g[1][1]-y)**2)[0]

            # print(f"\n  [Drone1] pos=({p1x:+.3f},{p1y:+.3f},{p1z:+.3f})  "
            #       f"tgt=({t1x:+.3f},{t1y:+.3f},{t1z:+.3f})  "
            #       f"nearest={nearest_grid(p1x,p1y)}  queue={q1}")
            # print(f"  [Drone2] pos=({p2x:+.3f},{p2y:+.3f},{p2z:+.3f})  "
            #       f"tgt=({t2x:+.3f},{t2y:+.3f},{t2z:+.3f})  "
            #       f"nearest={nearest_grid(p2x,p2y)}  queue={q2}")
            # print(f"  [SEP]    drone separation = {sep:.3f}m\n")
            continue

        parts = raw.split()

        if len(parts) == 2 and parts[1].lower() == "home":
            if parts[0] == "1":
                with nav1.lock:
                    nav1.waypoint_queue = []
                    nav1.target_x = home1_x
                    nav1.target_y = home1_y
                    nav1.target_z = DRONE1_TARGET_Z
                print(f"  [NAV]  Drone1 → Home ({home1_x:+.3f},{home1_y:+.3f},{DRONE1_TARGET_Z})")
            elif parts[0] == "2":
                with nav2.lock:
                    nav2.waypoint_queue = []
                    nav2.target_x = home2_x
                    nav2.target_y = home2_y
                    nav2.target_z = DRONE2_TARGET_Z
                print(f"  [NAV]  Drone2 → Home ({home2_x:+.3f},{home2_y:+.3f},{DRONE2_TARGET_Z})")
            else:
                print("  [ERR]  Use  1 home  or  2 home")
            continue

        if len(parts) == 3:
            drone_id = parts[0]
            if drone_id not in ("1", "2"):
                print("  [ERR]  First number must be 1 or 2")
                continue

            try:
                grid_num = int(parts[1])
            except ValueError:
                print("  [ERR]  Grid position must be a whole number 1-25")
                continue

            if grid_num not in GRID:
                print(f"  [ERR]  Grid position {grid_num} not valid — use 1 to 25")
                continue

            try:
                tz = float(parts[2])
            except ValueError:
                print("  [ERR]  Height must be a number  (e.g.  1 13 0.5)")
                continue

            if tz > MAX_ALTITUDE:
                print(f"  [ERR]  Height {tz}m exceeds MAX_ALTITUDE={MAX_ALTITUDE}m — rejected.")
                continue
            if tz < 0.1:
                print(f"  [ERR]  Height {tz}m too low — minimum is 0.1m.")
                continue

            tx, ty = GRID[grid_num]

            # Get moving drone current position and obstacle drone position
            if drone_id == "1":
                with pose1.lock:
                    cx, cy, cz = pose1.x, pose1.y, pose1.z
                with pose2.lock:
                    ox, oy, oz = pose2.x, pose2.y, pose2.z
                moving_name = "Drone1"
            else:
                with pose2.lock:
                    cx, cy, cz = pose2.x, pose2.y, pose2.z
                with pose1.lock:
                    ox, oy, oz = pose1.x, pose1.y, pose1.z
                moving_name = "Drone2"

            # Plan path — avoidance logic lives here
            waypoints, avoided = plan_path(
                moving_name,
                cx, cy, cz,
                tx, ty, tz,
                ox, oy, oz
            )

            if avoided:
                print(f"  [NAV]   {moving_name} → Grid {grid_num} at z={tz}m")
                print(f"  [AVOID] Path conflict — other drone at z={oz:.2f}m is in the way")
                print(f"  [AVOID] Step 1: climb to z={waypoints[0][2]:.2f}m at current position ({cx:+.3f},{cy:+.3f})")
                print(f"  [AVOID] Step 2: fly to Grid {grid_num} ({tx:+.3f},{ty:+.3f}) at z={waypoints[1][2]:.2f}m")
                print(f"  [AVOID] Step 3: descend to z={waypoints[2][2]:.2f}m at Grid {grid_num}")
            else:
                print(f"  [NAV]  {moving_name} → Grid {grid_num:>2} ({tx:+.3f},{ty:+.3f}) at z={tz}m")

            # Push waypoints into queue
            if drone_id == "1":
                with nav1.lock:
                    nav1.waypoint_queue = waypoints[1:]   # remaining after first
                    nav1.target_x = waypoints[0][0]
                    nav1.target_y = waypoints[0][1]
                    nav1.target_z = waypoints[0][2]
            else:
                with nav2.lock:
                    nav2.waypoint_queue = waypoints[1:]
                    nav2.target_x = waypoints[0][0]
                    nav2.target_y = waypoints[0][1]
                    nav2.target_z = waypoints[0][2]
            continue

        print("  [ERR]  Unknown command. Examples:  1 13 0.5  /  2 7 1.0  /  status  /  land")

# ── Per-drone flight loop ─────────────────────────────────────────────────────
def drone_flight_loop(scf, pose, nav, name, stop_event, status_lock):
    cf = scf.cf

    cf.param.set_value('stabilizer.estimator', '2')
    time.sleep(0.5)
    try:
        cf.param.set_value('flowdeck.useFlow', '0')
    except Exception:
        pass

    stop_ep = threading.Event()
    ep = threading.Thread(
        target=extpos_thread, args=(scf, pose, stop_ep), daemon=True)
    ep.start()
    print(f"[{name}]  extpos started. Waiting for MoCap data to establish...")
    time.sleep(1.0)

    cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    cf.param.set_value('kalman.resetEstimation', '0')
    print(f"[{name}]  EKF reset. Waiting for convergence...")
    time.sleep(1.5)

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
                queue_len  = len(nav.waypoint_queue)

            with pose.lock:
                cx, cy, cz = pose.x, pose.y, pose.z
                got_data   = pose.valid

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

            # If arrived at current waypoint and queue has more — advance
            if arrived and queue_len > 0:
                with nav.lock:
                    if nav.waypoint_queue:
                        next_wp = nav.waypoint_queue.pop(0)
                        nav.target_x = next_wp[0]
                        nav.target_y = next_wp[1]
                        nav.target_z = next_wp[2]
                        print(f"\n  [{name}] Waypoint reached — "
                              f"moving to next ({next_wp[0]:+.3f},{next_wp[1]:+.3f},{next_wp[2]:+.3f})")
                        # Reset PIDs for clean transition to next waypoint
                        pid_x.reset(); pid_y.reset(); pid_z.reset()

            nearest = min(GRID.items(),
                          key=lambda g: (g[1][0]-cx)**2 + (g[1][1]-cy)**2)

            with status_lock:
                queue_str = f"  queue={queue_len}" if queue_len > 0 else ""
                print(f"  [{name}] pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                      f"tgt=grid{nearest[0]:>2}({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                      f"dist={dist:.3f}m  "
                      f"{'[ARRIVED]' if arrived and queue_len == 0 else '         '}"
                      f"{queue_str}")

            elapsed = time.time() - loop_start
            time.sleep(max(0, dt - elapsed))

    except Exception as e:
        print(f"[{name}]  Flight loop error: {e}")

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

    print_grid()

    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.labeledMarkerListener = labeled_marker_callback
    client.rigidBodyListener = receiveRigidBodyFrame
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

    if not sanity_check():
        client.stop()
        return

    with pose1.lock:
        home1_x, home1_y = pose1.x, pose1.y
    with pose2.lock:
        home2_x, home2_y = pose2.x, pose2.y

    with nav1.lock:
        nav1.target_x = home1_x
        nav1.target_y = home1_y
    with nav2.lock:
        nav2.target_x = home2_x
        nav2.target_y = home2_y

    print(f"[NAV]    Drone1 home: x={home1_x:+.3f} y={home1_y:+.3f} z={DRONE1_TARGET_Z}")
    print(f"[NAV]    Drone2 home: x={home2_x:+.3f} y={home2_y:+.3f} z={DRONE2_TARGET_Z}")

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
            time.sleep(3.0)
            t2.start()

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
        try:
            with SyncCrazyflie(DRONE1_URI, cf=Crazyflie(rw_cache='./cache1')) as scf1:
                scf1.cf.commander.send_stop_setpoint()
                scf1.cf.param.set_value('stabilizer.estimator', '1')
        except Exception:
            pass
        try:
            with SyncCrazyflie(DRONE2_URI, cf=Crazyflie(rw_cache='./cache2')) as scf2:
                scf2.cf.commander.send_stop_setpoint()
                scf2.cf.param.set_value('stabilizer.estimator', '1')
        except Exception:
            pass

    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()