#!/usr/bin/env python3
"""
N-drone MoCap-only grid-based waypoint navigation with obstacle avoidance.
No flow deck required — each drone gets extpos from its own marker.
Marker IDs are identified at startup via interactive scan.
Auto-land triggered if a drone's marker is lost for more than 500ms.

Drones always face +X axis direction.

Controls:
  CTRL+X        — Emergency kill all drones (motors cut instantly)
  Ctrl+C        — Graceful land all drones

Terminal commands (mid-flight):
  <id> <grid> <z>  — send drone to grid position at height z (e.g. 1 13 0.5)
  <id> home        — drone returns to physical takeoff spot
  status           — print all drones current position and target
  land             — graceful land all drones
  land <id>        — land specific drone
  grid             — print the full grid map

Obstacle avoidance:
  When a drone's path passes through any other drone at similar height,
  the moving drone automatically:
    Step 1 — climbs above the highest conflicting drone + AVOID_OFFSET
    Step 2 — flies to target XY at avoid height
    Step 3 — descends to target height at target XY

Auto-land on marker loss:
  If a drone's MoCap marker is not seen for more than MARKER_TIMEOUT seconds
  that drone automatically lands. Other drones continue flying normally.
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
DEFAULT_HOVER_HEIGHT  =  0.5     # all drones take off to this height
MAX_SPEED             =  0.3
LOOP_HZ               =  50
EXTPOS_HZ             =  100
MAX_ALTITUDE          =  2.05
ARRIVAL_RADIUS        =  0.08
COLLISION_RADIUS      =  0.15
HEIGHT_TOLERANCE      =  0.3    # vertical diff below this triggers avoidance
AVOID_OFFSET          =  0.5    # climb this much above highest conflicting drone
MARKER_TIMEOUT        =  0.5    # seconds before auto-land triggered
TAKEOFF_STAGGER       =  3.0    # seconds between each drone takeoff

# ── PID gains ────────────────────────────────────────────────────────────────
KP_XY, KI_XY, KD_XY = 0.6, 0.05, 0.15
KP_Z,  KI_Z,  KD_Z  = 0.8, 0.08, 0.20

# ── Shared kill event ─────────────────────────────────────────────────────────
kill_event = threading.Event()

# ── Per-drone state classes ───────────────────────────────────────────────────
class PoseState:
    def __init__(self, name):
        self.name         = name
        self.lock         = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.valid        = False
        self.last_update  = 0.0    # timestamp of last MoCap frame

class NavState:
    def __init__(self):
        self.lock           = threading.Lock()
        self.target_x       = 0.0
        self.target_y       = 0.0
        self.target_z       = DEFAULT_HOVER_HEIGHT
        self.should_land    = False
        self.waypoint_queue = []

# ── Runtime drone registry — populated after scan ─────────────────────────────
# drones[i] = {
#   'name': 'Drone1',
#   'uri':  'radio://0/80/2M/E7E7E7E701',
#   'marker': 50424,
#   'pose': PoseState,
#   'nav':  NavState,
#   'stop': threading.Event,
#   'home_x': float,
#   'home_y': float,
# }
drones = []
marker_to_pose = {}

# ── Scan state ────────────────────────────────────────────────────────────────
class ScanState:
    def __init__(self):
        self.lock    = threading.Lock()
        self.markers = {}   # marker_id → (cf_x, cf_y, cf_z)

scan_state = ScanState()

# ── Scan callback ─────────────────────────────────────────────────────────────
def scan_callback(marker_id, pos):
    cf_x = pos[2]
    cf_y = pos[0]
    cf_z = pos[1]
    with scan_state.lock:
        scan_state.markers[marker_id] = (cf_x, cf_y, cf_z)

# ── Flight callback ───────────────────────────────────────────────────────────
def labeled_marker_callback(marker_id, pos):
    if marker_id in marker_to_pose:
        p = marker_to_pose[marker_id]
        with p.lock:
            p.x = pos[2]
            p.y = pos[0]
            p.z = pos[1]
            p.valid      = True
            p.last_update = time.time()

# ── Scan and assign markers ───────────────────────────────────────────────────
def scan_and_assign_markers(client, n):
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         MARKER SCAN & ASSIGNMENT         ║")
    print("  ╚══════════════════════════════════════════╝")
    print("\n  Scanning for markers — move each drone slightly to identify it...")
    print("  Scanning for 5 seconds...\n")

    client.labeledMarkerListener = scan_callback
    time.sleep(5.0)

    with scan_state.lock:
        found = dict(scan_state.markers)

    if len(found) == 0:
        print("  [ERROR] No markers found. Is Motive streaming?")
        raise SystemExit(1)

    print(f"  Found {len(found)} marker(s):\n")
    for mid, (x, y, z) in sorted(found.items()):
        print(f"    Marker ID: {mid}   "
              f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  (height={z:.3f}m)")
    print()

    assigned_ids = []
    for i in range(1, n+1):
        while True:
            try:
                raw = input(f"  Enter marker ID for Drone{i}: ").strip()
                mid = int(raw)
                if mid not in found:
                    print(f"  [ERR] {mid} not in scan results. Try again.")
                    continue
                if mid in assigned_ids:
                    print(f"  [ERR] {mid} already assigned to another drone.")
                    continue
                assigned_ids.append(mid)
                break
            except ValueError:
                print("  [ERR] Please enter a valid integer marker ID.")

    # Build drone registry
    for i, mid in enumerate(assigned_ids):
        idx   = i + 1
        name  = f"Drone{idx}"
        uri   = f"radio://0/80/2M/E7E7E7E7{idx:02d}"
        pos   = found[mid]

        pose = PoseState(name)
        nav  = NavState()

        with pose.lock:
            pose.x, pose.y, pose.z = pos
            pose.valid       = True
            pose.last_update = time.time()

        drones.append({
            'name':   name,
            'uri':    uri,
            'marker': mid,
            'pose':   pose,
            'nav':    nav,
            'stop':   threading.Event(),
            'home_x': pos[0],
            'home_y': pos[1],
        })
        marker_to_pose[mid] = pose

    # Switch to flight callback
    client.labeledMarkerListener = labeled_marker_callback

    print()
    for d in drones:
        with d['pose'].lock:
            x, y, z = d['pose'].x, d['pose'].y, d['pose'].z
        print(f"  [OK] {d['name']} → marker {d['marker']}  "
              f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  uri={d['uri']}")
    print()

# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check():
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         PRE-FLIGHT SANITY CHECK          ║")
    print("  ╚══════════════════════════════════════════╝\n")

    for d in drones:
        with d['pose'].lock:
            x, y, z = d['pose'].x, d['pose'].y, d['pose'].z
        print(f"  {d['name']} (marker {d['marker']}):")
        print(f"    x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  (height={z:.3f}m)")

    print()
    print("  Look at where each drone is physically sitting and confirm.\n")
    confirm = input("  Do these positions match the physical drones? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("\n  [ABORT] Sanity check failed — re-run and re-enter marker IDs.\n")
        return False
    print("  [OK] Sanity check passed — proceeding to flight.\n")
    return True

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
def path_conflicts(start_x, start_y, target_x, target_y, obs_x, obs_y, radius):
    dx = target_x - start_x
    dy = target_y - start_y
    seg_len_sq = dx*dx + dy*dy

    if seg_len_sq == 0:
        return ((obs_x-start_x)**2 + (obs_y-start_y)**2)**0.5 < radius

    t = ((obs_x-start_x)*dx + (obs_y-start_y)*dy) / seg_len_sq
    t = max(0.0, min(1.0, t))
    closest_x = start_x + t*dx
    closest_y = start_y + t*dy
    return ((obs_x-closest_x)**2 + (obs_y-closest_y)**2)**0.5 < radius

# ── Plan path with avoidance against all other drones ────────────────────────
def plan_path(moving_idx, curr_x, curr_y, curr_z,
              target_x, target_y, target_z):
    """
    Checks path against all other drones.
    If any conflict at similar height, climbs above the highest conflicting
    drone + AVOID_OFFSET, flies over, then descends.
    Returns (waypoints_list, avoided_bool, conflicting_drone_names).
    """
    conflicting   = []
    highest_obs_z = 0.0

    for i, d in enumerate(drones):
        if i == moving_idx:
            continue
        with d['pose'].lock:
            ox, oy, oz = d['pose'].x, d['pose'].y, d['pose'].z

        conflict       = path_conflicts(curr_x, curr_y, target_x, target_y,
                                        ox, oy, COLLISION_RADIUS)
        height_similar = abs(curr_z - oz) < HEIGHT_TOLERANCE

        if conflict and height_similar:
            conflicting.append(d['name'])
            if oz > highest_obs_z:
                highest_obs_z = oz

    if not conflicting:
        return [(target_x, target_y, target_z)], False, []

    avoid_z = min(highest_obs_z + AVOID_OFFSET, MAX_ALTITUDE - 0.1)

    waypoints = [
        (curr_x,   curr_y,   avoid_z),
        (target_x, target_y, avoid_z),
        (target_x, target_y, target_z),
    ]
    return waypoints, True, conflicting

# ── Emergency kill ────────────────────────────────────────────────────────────
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
            x, y, z  = pose.x, pose.y, pose.z
            valid     = pose.valid
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
            print(f"[{name}]  ALTITUDE LIMIT HIT — killing!")
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

# ── Per-drone flight loop ─────────────────────────────────────────────────────
def drone_flight_loop(scf, drone_dict, status_lock):
    pose  = drone_dict['pose']
    nav   = drone_dict['nav']
    name  = drone_dict['name']
    stop  = drone_dict['stop']
    cf    = scf.cf

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
        while not stop.is_set() and not kill_event.is_set():
            loop_start = time.time()

            with nav.lock:
                if nav.should_land:
                    break
                tx, ty, tz = nav.target_x, nav.target_y, nav.target_z
                queue_len  = len(nav.waypoint_queue)

            with pose.lock:
                cx, cy, cz    = pose.x, pose.y, pose.z
                got_data      = pose.valid
                last_update   = pose.last_update

            # ── Marker loss check — auto-land if not seen for MARKER_TIMEOUT ──
            if got_data and (time.time() - last_update) > MARKER_TIMEOUT:
                print(f"\n[{name}]  !! MARKER LOST for >{MARKER_TIMEOUT}s — AUTO LANDING !!")
                break

            # ── Hard altitude limit ───────────────────────────────────────────
            if cz > MAX_ALTITUDE:
                print(f"\n[{name}]  !! ALTITUDE LIMIT {MAX_ALTITUDE}m EXCEEDED — KILLING !!")
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

            dist    = (ex**2 + ey**2 + ez**2) ** 0.5
            arrived = dist < ARRIVAL_RADIUS

            if arrived and queue_len > 0:
                with nav.lock:
                    if nav.waypoint_queue:
                        next_wp = nav.waypoint_queue.pop(0)
                        nav.target_x = next_wp[0]
                        nav.target_y = next_wp[1]
                        nav.target_z = next_wp[2]
                        print(f"\n  [{name}] Waypoint reached — "
                              f"moving to next ({next_wp[0]:+.3f},{next_wp[1]:+.3f},{next_wp[2]:+.3f})")
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

# ── Terminal input thread ─────────────────────────────────────────────────────
def input_thread(status_lock):
    n = len(drones)
    print("\n  ── Commands ───────────────────────────────────────────────")
    print(f"    <id> <grid> <z>  — drone to grid pos  (e.g. 1 13 0.5)  id=1-{n}")
    print(f"    <id> home        — drone return to takeoff spot")
    print(f"    status           — print all drone positions")
    print(f"    grid             — print the grid map")
    print(f"    land             — land all drones")
    print(f"    land <id>        — land specific drone")
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
            for d in drones:
                with d['nav'].lock:
                    d['nav'].should_land = True
            print("  [NAV]  Landing all drones.")
            break

        if raw.lower() == "status":
            print()
            for d in drones:
                with d['pose'].lock:
                    px, py, pz = d['pose'].x, d['pose'].y, d['pose'].z
                    age = time.time() - d['pose'].last_update
                with d['nav'].lock:
                    tx, ty, tz = d['nav'].target_x, d['nav'].target_y, d['nav'].target_z
                    ql = len(d['nav'].waypoint_queue)
                nearest = min(GRID.items(),
                              key=lambda g: (g[1][0]-px)**2 + (g[1][1]-py)**2)[0]
                lost_str = f"  [MARKER AGE {age:.2f}s]" if age > 0.2 else ""
                print(f"  [{d['name']}] pos=({px:+.3f},{py:+.3f},{pz:+.3f})  "
                      f"tgt=({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                      f"nearest={nearest}  queue={ql}{lost_str}")
            print()
            continue

        parts = raw.split()

        # land <id>
        if len(parts) == 2 and parts[0].lower() == "land":
            try:
                idx = int(parts[1]) - 1
                if 0 <= idx < len(drones):
                    with drones[idx]['nav'].lock:
                        drones[idx]['nav'].should_land = True
                    print(f"  [NAV]  Landing {drones[idx]['name']}.")
                else:
                    print(f"  [ERR]  Drone ID must be 1 to {n}")
            except ValueError:
                print("  [ERR]  Use:  land <id>  (e.g.  land 2)")
            continue

        # <id> home
        if len(parts) == 2 and parts[1].lower() == "home":
            try:
                idx = int(parts[0]) - 1
                if 0 <= idx < len(drones):
                    d = drones[idx]
                    with d['nav'].lock:
                        d['nav'].waypoint_queue = []
                        d['nav'].target_x = d['home_x']
                        d['nav'].target_y = d['home_y']
                        d['nav'].target_z = DEFAULT_HOVER_HEIGHT
                    print(f"  [NAV]  {d['name']} → Home "
                          f"({d['home_x']:+.3f},{d['home_y']:+.3f},{DEFAULT_HOVER_HEIGHT})")
                else:
                    print(f"  [ERR]  Drone ID must be 1 to {n}")
            except ValueError:
                print("  [ERR]  Use:  <id> home  (e.g.  1 home)")
            continue

        # <id> <grid> <z>
        if len(parts) == 3:
            try:
                drone_idx = int(parts[0]) - 1
            except ValueError:
                print("  [ERR]  First value must be a drone number")
                continue

            if not (0 <= drone_idx < len(drones)):
                print(f"  [ERR]  Drone ID must be 1 to {n}")
                continue

            try:
                grid_num = int(parts[1])
            except ValueError:
                print("  [ERR]  Grid position must be a whole number 1-25")
                continue

            if grid_num not in GRID:
                print(f"  [ERR]  Grid {grid_num} not valid — use 1 to 25")
                continue

            try:
                tz = float(parts[2])
            except ValueError:
                print("  [ERR]  Height must be a number  (e.g. 0.5)")
                continue

            if tz > MAX_ALTITUDE:
                print(f"  [ERR]  Height {tz}m exceeds MAX_ALTITUDE={MAX_ALTITUDE}m")
                continue
            if tz < 0.1:
                print(f"  [ERR]  Height {tz}m too low — minimum 0.1m")
                continue

            tx, ty  = GRID[grid_num]
            d       = drones[drone_idx]
            name    = d['name']

            with d['pose'].lock:
                cx, cy, cz = d['pose'].x, d['pose'].y, d['pose'].z

            waypoints, avoided, conflicting = plan_path(
                drone_idx, cx, cy, cz, tx, ty, tz)

            if avoided:
                print(f"  [NAV]   {name} → Grid {grid_num} at z={tz}m")
                print(f"  [AVOID] Path conflict with {', '.join(conflicting)}")
                print(f"  [AVOID] Step 1: climb to z={waypoints[0][2]:.2f}m at ({cx:+.3f},{cy:+.3f})")
                print(f"  [AVOID] Step 2: fly to Grid {grid_num} ({tx:+.3f},{ty:+.3f}) at z={waypoints[1][2]:.2f}m")
                print(f"  [AVOID] Step 3: descend to z={waypoints[2][2]:.2f}m at Grid {grid_num}")
            else:
                print(f"  [NAV]  {name} → Grid {grid_num:>2} ({tx:+.3f},{ty:+.3f}) at z={tz}m")

            with d['nav'].lock:
                d['nav'].waypoint_queue = waypoints[1:]
                d['nav'].target_x = waypoints[0][0]
                d['nav'].target_y = waypoints[0][1]
                d['nav'].target_z = waypoints[0][2]
            continue

        print("  [ERR]  Unknown command. Examples:  1 13 0.5  /  2 home  /  status  /  land")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print_grid()

    # Ask how many drones
    while True:
        try:
            n = int(input("  How many drones are you flying today? : ").strip())
            if n < 1:
                print("  [ERR]  Must be at least 1.")
                continue
            break
        except ValueError:
            print("  [ERR]  Please enter a whole number.")

    print(f"\n  [OK]  Setting up for {n} drone(s).\n")

    # Start NatNet
    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.run()
    print("[MoCap]  Connected.\n")

    # Scan and assign
    try:
        scan_and_assign_markers(client, n)
    except SystemExit:
        client.stop()
        return

    # Sanity check
    if not sanity_check():
        client.stop()
        return

    # Set nav home targets
    for d in drones:
        with d['nav'].lock:
            d['nav'].target_x = d['home_x']
            d['nav'].target_y = d['home_y']
            d['nav'].target_z = DEFAULT_HOVER_HEIGHT
        print(f"[NAV]    {d['name']} home: "
              f"x={d['home_x']:+.3f}  y={d['home_y']:+.3f}  z={DEFAULT_HOVER_HEIGHT}  "
              f"uri={d['uri']}")

    cflib.crtp.init_drivers()
    print("\n[CF]     Connecting to all drones...")

    status_lock = threading.Lock()
    start_kill_listener()

    print("\n  ========================================")
    print("  CTRL+X   = Emergency kill all drones  ")
    print("  Ctrl+C   = Graceful land all drones   ")
    print(f"  Altitude limit = {MAX_ALTITUDE}m       ")
    print(f"  Marker timeout = {MARKER_TIMEOUT}s (auto-land)")
    print("  ========================================\n")

    # Open all SyncCrazyflie connections dynamically
    def open_connections(drone_list, idx, scf_list, status_lock):
        """Recursively open SyncCrazyflie connections for all drones."""
        if idx == len(drone_list):
            # All connected — start flight loops
            threads = []
            for i, d in enumerate(drone_list):
                t = threading.Thread(
                    target=drone_flight_loop,
                    args=(scf_list[i], d, status_lock),
                    daemon=True)
                threads.append(t)

            for i, t in enumerate(threads):
                t.start()
                print(f"[CF]     {drone_list[i]['name']} flight loop started.")
                if i < len(threads) - 1:
                    time.sleep(TAKEOFF_STAGGER)

            # Start input thread after all drones airborne
            time.sleep(2.0)
            it = threading.Thread(
                target=input_thread,
                args=(status_lock,),
                daemon=True)
            it.start()

            try:
                while any(t.is_alive() for t in threads):
                    if kill_event.is_set():
                        for d in drone_list:
                            d['stop'].set()
                    time.sleep(0.2)
            except KeyboardInterrupt:
                print("\n[CTRL]   Ctrl+C — graceful landing all drones...")
                for d in drone_list:
                    with d['nav'].lock:
                        d['nav'].should_land = True
                for t in threads:
                    t.join(timeout=15)
            return

        d = drone_list[idx]
        try:
            with SyncCrazyflie(d['uri'],
                               cf=Crazyflie(rw_cache=f"./cache{idx+1}")) as scf:
                scf_list.append(scf)
                print(f"[CF]     {d['name']} connected via {d['uri']}")
                open_connections(drone_list, idx+1, scf_list, status_lock)
        except Exception as e:
            print(f"[CF]     Failed to connect {d['name']} at {d['uri']}: {e}")
            # Cleanup already connected
            for d2 in drone_list[:idx]:
                try:
                    d2['stop'].set()
                except Exception:
                    pass

    try:
        open_connections(drones, 0, [], status_lock)
    except Exception as e:
        print(f"[CF]     Connection error: {e}")

    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()