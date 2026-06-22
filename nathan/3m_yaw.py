#!/usr/bin/env python3
"""
N-drone MoCap rigid body waypoint navigation with yaw hold.
No flow deck required — position and heading from OptiTrack rigid bodies.

HOW TO USE:
  1. Place drones in any orientation — each holds its initial heading automatically.
  2. In the ACTIVE DRONES section below, comment/uncomment which drones to fly.
  3. Run the script — it auto-detects how many are active.
  4. Each drone's PID gains can be tuned individually in its Drone(...) definition.

Controls:
  CTRL+X         — Emergency kill ALL drones instantly
  Ctrl+C         — Graceful land ALL drones

Terminal commands (mid-flight):
  <n> <grid> <z> — send drone N to grid position at height z  (e.g. 1 13 0.5)
  <n> home       — drone N return to its physical takeoff spot
  status         — print all drones position, heading and target
  grid           — print the full grid map
  land           — graceful land all drones
  land <n>       — graceful land drone N only

Obstacle avoidance:
  When a drone path passes through any other drone at similar height,
  the moving drone automatically:
    Step 1 — climbs to blocking drone height + AVOID_OFFSET at current XY
    Step 2 — flies to target XY at avoid height
    Step 3 — descends to target height at target XY
"""

import math
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

# ── Global flight parameters ──────────────────────────────────────────────────
MAX_SPEED        = 0.3
MAX_YAW_RATE     = 100.0   # deg/s — max yaw correction rate
LOOP_HZ          = 50
EXTPOS_HZ        = 100
ARRIVAL_RADIUS   = 0.08
COLLISION_RADIUS = 0.15
HEIGHT_TOLERANCE = 0.3     # z diff below which avoidance triggers
AVOID_OFFSET     = 0.5     # climb this much above the blocking drone

# ── Flight volume clamps — edit these 6 values to resize the allowed cube ─────
# Commanded waypoints are clamped to this volume before being sent to the drone.
# A drift safety net in the flight loop triggers graceful land if the drone
# physically drifts outside this volume despite clamped targets.
CLAMP_X_MIN = -1.8
CLAMP_X_MAX = +1.8
CLAMP_Y_MIN = -1.8
CLAMP_Y_MAX = +1.8
CLAMP_Z_MIN =  0.1   # keep above floor
CLAMP_Z_MAX =  1.8   # max altitude

# ── Shared kill event ─────────────────────────────────────────────────────────
kill_event = threading.Event()

# ── PID controller ────────────────────────────────────────────────────────────
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

def clamp_target(x, y, z):
    """Clamp a commanded waypoint to the configured flight volume."""
    return (max(CLAMP_X_MIN, min(CLAMP_X_MAX, x)),
            max(CLAMP_Y_MIN, min(CLAMP_Y_MAX, y)),
            max(CLAMP_Z_MIN, min(CLAMP_Z_MAX, z)))

def quat_to_yaw(qx, qy, qz, qw):
    """
    Extract yaw (rotation around world Z axis) from quaternion.
    Returns yaw in degrees, range -180..+180.
    NatNet quaternion convention: (qx, qy, qz, qw).
    """
    # Remap NatNet axes to world frame same as position:
    # NatNet x->world y, NatNet y->world z, NatNet z->world x
    # For yaw we only need the world-Z component of rotation
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw_rad   = math.atan2(siny_cosp, cosy_cosp)
    return math.degrees(yaw_rad)

def wrap_angle(angle):
    """Wrap angle to -180..+180 degrees."""
    while angle >  180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle

# ── Drone class ───────────────────────────────────────────────────────────────
class Drone:
    def __init__(self,
                 number,
                 marker_id,
                 default_z,
                 kp_xy=0.6,  ki_xy=0.05,  kd_xy=0.15,
                 kp_z=0.8,   ki_z=0.08,   kd_z=0.20,
                 kp_yaw=2.0, ki_yaw=0.05, kd_yaw=0.1):
        """
        number    : drone number (1-based, matches terminal command prefix)
        marker_id : Motive rigid body streaming ID  (350 + number by convention)
        default_z : hover height in metres
        kp/ki/kd  : PID gains — tune per drone if needed
        kp/ki/kd_yaw : yaw PID — output is deg/s yaw rate sent to drone
        """
        self.number    = number
        self.name      = f"Drone{number}"
        self.uri       = f"radio://0/80/2M/E7E7E7E70{number}"
        self.marker_id = marker_id
        self.default_z = default_z
        self.cache     = f"./cache{number}"

        # ── Pose (filled by NatNet callback) ──────────────────────────────────
        self.pose_lock  = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.yaw        = 0.0    # degrees, world frame
        self.qx = self.qy = self.qz = 0.0
        self.qw         = 1.0
        self.pose_valid = False

        # ── Nav state ─────────────────────────────────────────────────────────
        self.nav_lock       = threading.Lock()
        self.target_x       = 0.0
        self.target_y       = 0.0
        self.target_z       = default_z
        self.target_yaw     = None   # set to home yaw on takeoff
        self.should_land    = False
        self.waypoint_queue = []     # list of (x, y, z)

        self.home_x   = 0.0
        self.home_y   = 0.0
        self.home_yaw = None         # captured at startup, held throughout

        # ── Per-drone PID controllers ─────────────────────────────────────────
        self.pid_x   = PID(kp_xy,  ki_xy,  kd_xy)
        self.pid_y   = PID(kp_xy,  ki_xy,  kd_xy)
        self.pid_z   = PID(kp_z,   ki_z,   kd_z)
        self.pid_yaw = PID(kp_yaw, ki_yaw, kd_yaw, integral_limit=30.0)

        # ── Control events ────────────────────────────────────────────────────
        self.stop_event = threading.Event()

    # ── Pose update (called from NatNet thread) ───────────────────────────────
    def update_pose(self, position, rotation):
        """
        NatNet streams position as (y, z, x) and rotation as quaternion (qx, qy, qz, qw).
        Remap position to world frame and extract yaw from quaternion.
        """
        qx, qy, qz, qw = rotation
        yaw = quat_to_yaw(qx, qy, qz, qw)
        with self.pose_lock:
            self.x  = position[2]
            self.y  = position[0]
            self.z  = position[1]
            self.qx, self.qy, self.qz, self.qw = qx, qy, qz, qw
            self.yaw        = yaw
            self.pose_valid = True

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.z, self.yaw, self.pose_valid

    def get_orientation(self):
        """Return full quaternion for extpose."""
        with self.pose_lock:
            return self.qx, self.qy, self.qz, self.qw

    # ── Nav helpers ───────────────────────────────────────────────────────────
    def set_target(self, x, y, z, queue=None):
        with self.nav_lock:
            self.target_x = x
            self.target_y = y
            self.target_z = z
            self.waypoint_queue = queue if queue else []

    def get_nav(self):
        with self.nav_lock:
            return (self.target_x, self.target_y, self.target_z,
                    self.target_yaw, self.should_land, list(self.waypoint_queue))

    def advance_waypoint(self):
        """Pop next waypoint from queue and set as target. Returns new target or None."""
        with self.nav_lock:
            if self.waypoint_queue:
                wp = self.waypoint_queue.pop(0)
                self.target_x, self.target_y, self.target_z = wp
                return wp
        return None

    def reset_pids(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.pid_z.reset()
        self.pid_yaw.reset()

    # ── extpose sender (position + orientation) ───────────────────────────────
    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, _, valid = self.get_pose()
            qx, qy, qz, qw   = self.get_orientation()
            if valid:
                # send_extpose gives the EKF both position AND orientation
                # so it knows the drone's heading from the start
                scf.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    # ── Takeoff ───────────────────────────────────────────────────────────────
    def takeoff(self, scf):
        cf = scf.cf
        print(f"[{self.name}]  Taking off to z={self.default_z}m  "
              f"(holding yaw={self.home_yaw:.1f}deg) ...")
        for _ in range(10):
            cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
            time.sleep(0.01)
        start = time.time()
        while True:
            if kill_event.is_set():
                cf.commander.send_stop_setpoint()
                return False
            _, _, cz, _, _ = self.get_pose()
            if cz > CLAMP_Z_MAX:
                print(f"[{self.name}]  ALTITUDE LIMIT HIT DURING TAKEOFF — killing!")
                kill_event.set()
                cf.commander.send_stop_setpoint()
                return False
            if cz >= self.default_z * 0.90:
                print(f"[{self.name}]  Reached z={cz:.3f}m — PID taking over.")
                return True
            if time.time() - start > 8.0:
                print(f"[{self.name}]  WARNING: Takeoff timeout — continuing anyway.")
                return True
            cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
            time.sleep(0.02)

    # ── Land ──────────────────────────────────────────────────────────────────
    def land(self, scf):
        cf = scf.cf
        print(f"\n[{self.name}]  Descending...")
        _, _, cz, _, _ = self.get_pose()
        while cz > 0.10:
            if kill_event.is_set():
                break
            cf.commander.send_velocity_world_setpoint(0, 0, -0.2, 0)
            time.sleep(0.05)
            _, _, cz, _, _ = self.get_pose()
        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        try:
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            cf.param.set_value('stabilizer.estimator', '1')
        except Exception:
            pass
        print(f"[{self.name}]  Landed.")

    # ── Flight loop ───────────────────────────────────────────────────────────
    def flight_loop(self, scf, all_drones, status_lock):
        cf = scf.cf

        cf.param.set_value('stabilizer.estimator', '2')
        time.sleep(0.5)
        try:
            cf.param.set_value('flowdeck.useFlow', '0')
        except Exception:
            pass

        stop_ep = threading.Event()
        ep = threading.Thread(
            target=self.run_extpos, args=(scf, stop_ep), daemon=True)
        ep.start()
        print(f"[{self.name}]  extpose started. Waiting for MoCap data...")
        time.sleep(1.0)

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        print(f"[{self.name}]  EKF reset. Waiting for convergence...")
        time.sleep(1.5)

        if not self.takeoff(scf) or kill_event.is_set():
            stop_ep.set()
            return

        self.reset_pids()
        print(f"[{self.name}]  Ready for waypoints.")
        dt = 1.0 / LOOP_HZ

        try:
            while not self.stop_event.is_set() and not kill_event.is_set():
                loop_start = time.time()

                tx, ty, tz, target_yaw, should_land, queue = self.get_nav()
                if should_land:
                    break

                cx, cy, cz, cyaw, got_data = self.get_pose()
                queue_len = len(queue)

                # ── Drift safety net — graceful land if drone escapes volume ──
                # Targets are pre-clamped; this only catches unexpected physical drift.
                out_of_bounds = (
                    cx < CLAMP_X_MIN - 0.1 or cx > CLAMP_X_MAX + 0.1 or
                    cy < CLAMP_Y_MIN - 0.1 or cy > CLAMP_Y_MAX + 0.1 or
                    cz < CLAMP_Z_MIN - 0.1 or cz > CLAMP_Z_MAX + 0.1
                )
                if out_of_bounds:
                    print(f"\n[{self.name}]  !! OUT OF BOUNDS "
                          f"pos=({cx:+.3f},{cy:+.3f},{cz:+.3f}) — graceful landing!")
                    with self.nav_lock:
                        self.should_land = True
                    break

                if not got_data:
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    time.sleep(dt)
                    continue

                now = time.time()
                ex, ey, ez = tx - cx, ty - cy, tz - cz

                vx = clamp(self.pid_x.update(ex, now), MAX_SPEED)
                vy = clamp(self.pid_y.update(ey, now), MAX_SPEED)
                vz = clamp(self.pid_z.update(ez, now), MAX_SPEED)

                # ── Yaw PID — hold home heading ───────────────────────────────
                # wrap_angle ensures we always take the shortest rotation path
                yaw_error    = wrap_angle(target_yaw - cyaw)
                yaw_rate_cmd = clamp(self.pid_yaw.update(yaw_error, now), MAX_YAW_RATE)

                cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_rate_cmd)

                dist    = (ex**2 + ey**2 + ez**2) ** 0.5
                arrived = dist < ARRIVAL_RADIUS

                if arrived and queue_len > 0:
                    wp = self.advance_waypoint()
                    if wp:
                        print(f"\n  [{self.name}] Waypoint reached — "
                              f"next ({wp[0]:+.3f},{wp[1]:+.3f},{wp[2]:+.3f})")
                        self.reset_pids()

                nearest = min(GRID.items(),
                              key=lambda g: (g[1][0]-cx)**2 + (g[1][1]-cy)**2)
                queue_str = f"  queue={queue_len}" if queue_len > 0 else ""
                with status_lock:
                    print(f"  [{self.name}] pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                          f"yaw={cyaw:+.1f}deg  "
                          f"tgt=grid{nearest[0]:>2}({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                          f"dist={dist:.3f}m  "
                          f"{'[ARRIVED]' if arrived and queue_len == 0 else '         '}"
                          f"{queue_str}")

                elapsed = time.time() - loop_start
                time.sleep(max(0, dt - elapsed))

        except Exception as e:
            print(f"[{self.name}]  Flight loop error: {e}")

        if kill_event.is_set():
            cf.commander.send_stop_setpoint()
            print(f"[{self.name}]  Motors killed instantly.")
        else:
            self.land(scf)

        stop_ep.set()


# ══════════════════════════════════════════════════════════════════════════════
#
#   ACTIVE DRONES — comment out any drone you don't want to fly
#
#   Drone(number, marker_id, default_z,
#         kp_xy, ki_xy, kd_xy,
#         kp_z,  ki_z,  kd_z,
#         kp_yaw, ki_yaw, kd_yaw)
#
#   number    : 1-8  (also used to build URI: E7E7E7E70<number>)
#   marker_id : Motive streaming ID  (351 for Drone1, 352 for Drone2, etc.)
#   default_z : takeoff and hover height in metres
#   PID gains : tune per drone if needed — leave at defaults for identical hardware
#   yaw PID   : kp_yaw output is deg/s — increase kp_yaw if heading drifts slowly,
#               decrease if it oscillates
#
# ══════════════════════════════════════════════════════════════════════════════
ACTIVE_DRONES = [

    Drone(number=1, marker_id=351, default_z=0.5),
    # Drone(number=2, marker_id=352, default_z=0.5),
    # Drone(number=3, marker_id=353, default_z=0.5),
    # Drone(number=4, marker_id=354, default_z=0.5),
    # Drone(number=5, marker_id=355, default_z=0.5),
    # Drone(number=6, marker_id=356, default_z=0.5),
    # Drone(number=7, marker_id=357, default_z=0.5),
    # Drone(number=8, marker_id=358, default_z=0.5),

    # ── Custom PID example ────────────────────────────────────────────────────
    # Drone(number=4, marker_id=354, default_z=0.5,
    #       kp_xy=0.8, ki_xy=0.06, kd_xy=0.18,
    #       kp_z=1.0,  ki_z=0.10,  kd_z=0.25,
    #       kp_yaw=2.5, ki_yaw=0.05, kd_yaw=0.1),

]
# ══════════════════════════════════════════════════════════════════════════════


# ── NatNet rigid body callback ────────────────────────────────────────────────
marker_to_drone = {}

def receiveRigidBodyFrame(rb_id, position, rotation):
    """rotation is (qx, qy, qz, qw) from NatNet."""
    if rb_id in marker_to_drone:
        marker_to_drone[rb_id].update_pose(position, rotation)

# ── Path conflict check ───────────────────────────────────────────────────────
def path_conflicts(sx, sy, tx, ty, ox, oy, radius):
    dx, dy = tx - sx, ty - sy
    seg_sq = dx*dx + dy*dy
    if seg_sq == 0:
        return ((ox-sx)**2 + (oy-sy)**2) ** 0.5 < radius
    t  = max(0.0, min(1.0, ((ox-sx)*dx + (oy-sy)*dy) / seg_sq))
    cx = sx + t*dx
    cy = sy + t*dy
    return ((ox-cx)**2 + (oy-cy)**2) ** 0.5 < radius

# ── Plan path with avoidance against all other active drones ─────────────────
def plan_path(moving_drone, target_x, target_y, target_z):
    cx, cy, cz, _, _ = moving_drone.get_pose()
    worst_z = None

    for drone in ACTIVE_DRONES:
        if drone is moving_drone:
            continue
        ox, oy, oz, _, _ = drone.get_pose()
        conflict       = path_conflicts(cx, cy, target_x, target_y,
                                        ox, oy, COLLISION_RADIUS)
        height_similar = abs(cz - oz) < HEIGHT_TOLERANCE
        if conflict and height_similar:
            if worst_z is None or oz > worst_z:
                worst_z = oz

    if worst_z is None:
        return [(target_x, target_y, target_z)], False

    avoid_z = min(worst_z + AVOID_OFFSET, CLAMP_Z_MAX - 0.1)
    return [
        (cx,       cy,       avoid_z),
        (target_x, target_y, avoid_z),
        (target_x, target_y, target_z),
    ], True

# ── Emergency kill listener ───────────────────────────────────────────────────
def on_press(key):
    if hasattr(key, 'char') and key.char == '\x18':
        print("\n\n  !! EMERGENCY KILL — CTRL+X PRESSED !!")
        kill_event.set()

def start_kill_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

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
    print("  ╚══════════════════════════════════════════╝\n")
    for drone in ACTIVE_DRONES:
        x, y, z, yaw, _ = drone.get_pose()
        print(f"  {drone.name} (marker {drone.marker_id})  "
              f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  yaw={yaw:+.1f}deg")
    print()
    print("  Z values should be close to 0.0 (floor level).")
    print("  Yaw shows the current heading each drone will hold during flight.")
    print("  Confirm positions match where each drone is physically sitting.\n")
    confirm = input("  Do ALL positions match the physical drones? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("\n  [ABORT] Sanity check failed. Check marker IDs in ACTIVE_DRONES.\n")
        return False
    print("  [OK] Sanity check passed — proceeding to flight.\n")
    return True

# ── Terminal input thread ─────────────────────────────────────────────────────
def input_thread(status_lock):
    numbers = [d.number for d in ACTIVE_DRONES]

    print("\n  ── Commands ────────────────────────────────────────────────────")
    print(f"    <n> <grid> <z>  — drone N to grid at height  (e.g. 1 13 0.5)")
    print(f"    <n> home        — drone N return to takeoff spot")
    print("    status          — print all positions and headings")
    print("    grid            — print the grid map")
    print("    land            — land ALL drones")
    print("    land <n>        — land drone N only")
    print(f"    Active drones: {numbers}")
    print(f"    Volume: x=[{CLAMP_X_MIN},{CLAMP_X_MAX}]  "
          f"y=[{CLAMP_Y_MIN},{CLAMP_Y_MAX}]  "
          f"z=[{CLAMP_Z_MIN},{CLAMP_Z_MAX}]")
    print("  ────────────────────────────────────────────────────────────────\n")

    drone_by_num = {d.number: d for d in ACTIVE_DRONES}

    while True:
        try:
            raw = input("  > ").strip()
        except EOFError:
            break
        if not raw:
            continue

        parts = raw.split()

        if parts[0].lower() == "grid":
            print_grid()
            continue

        # land  or  land <n>
        if parts[0].lower() == "land":
            if len(parts) == 1:
                for drone in ACTIVE_DRONES:
                    with drone.nav_lock:
                        drone.should_land = True
                print("  [NAV]  Landing ALL drones.")
                break
            elif len(parts) == 2:
                try:
                    dn = int(parts[1])
                    if dn in drone_by_num:
                        with drone_by_num[dn].nav_lock:
                            drone_by_num[dn].should_land = True
                        print(f"  [NAV]  Landing Drone{dn}.")
                    else:
                        print(f"  [ERR]  No active drone {dn}. Active: {numbers}")
                except ValueError:
                    print("  [ERR]  Usage: land  or  land <n>")
            continue

        if parts[0].lower() == "status":
            print()
            for drone in ACTIVE_DRONES:
                cx, cy, cz, cyaw, _ = drone.get_pose()
                with drone.nav_lock:
                    tx, ty, tz = drone.target_x, drone.target_y, drone.target_z
                    q = len(drone.waypoint_queue)
                dist    = ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2) ** 0.5
                nearest = min(GRID.items(),
                              key=lambda g: (g[1][0]-cx)**2 + (g[1][1]-cy)**2)
                print(f"  [{drone.name}] pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                      f"yaw={cyaw:+.1f}deg (tgt={drone.home_yaw:+.1f})  "
                      f"tgt=grid{nearest[0]:>2}({tx:+.3f},{ty:+.3f},{tz:+.3f})  "
                      f"dist={dist:.3f}m  queue={q}")
            print()
            continue

        # <n> home
        if len(parts) == 2 and parts[1].lower() == "home":
            try:
                dn = int(parts[0])
            except ValueError:
                print("  [ERR]  Usage: <n> home")
                continue
            if dn not in drone_by_num:
                print(f"  [ERR]  No active drone {dn}. Active: {numbers}")
                continue
            d = drone_by_num[dn]
            d.set_target(d.home_x, d.home_y, d.default_z)
            print(f"  [NAV]  Drone{dn} -> Home ({d.home_x:+.3f},{d.home_y:+.3f},{d.default_z})")
            continue

        # <n> <grid> <z>
        if len(parts) == 3:
            try:
                dn = int(parts[0])
            except ValueError:
                print("  [ERR]  First value must be a drone number")
                continue
            if dn not in drone_by_num:
                print(f"  [ERR]  No active drone {dn}. Active: {numbers}")
                continue

            try:
                grid_num = int(parts[1])
            except ValueError:
                print("  [ERR]  Grid must be a whole number 1-25")
                continue
            if grid_num not in GRID:
                print(f"  [ERR]  Grid {grid_num} not valid — use 1 to 25")
                continue

            try:
                tz = float(parts[2])
            except ValueError:
                print("  [ERR]  Height must be a number  (e.g. 1 13 0.5)")
                continue

            tx, ty = GRID[grid_num]
            # Clamp the commanded target to the allowed flight volume
            orig = (tx, ty, tz)
            tx, ty, tz = clamp_target(tx, ty, tz)
            if (tx, ty, tz) != orig:
                print(f"  [CLAMP] Target clamped: "
                      f"({orig[0]:+.3f},{orig[1]:+.3f},{orig[2]:.3f}) "
                      f"-> ({tx:+.3f},{ty:+.3f},{tz:.3f})")

            moving   = drone_by_num[dn]
            waypoints, avoided = plan_path(moving, tx, ty, tz)

            if avoided:
                print(f"  [NAV]   Drone{dn} -> Grid {grid_num} at z={tz}m")
                print(f"  [AVOID] Path conflict detected")
                print(f"  [AVOID] Step 1: climb to z={waypoints[0][2]:.2f}m at current XY")
                print(f"  [AVOID] Step 2: fly to Grid {grid_num} at z={waypoints[1][2]:.2f}m")
                print(f"  [AVOID] Step 3: descend to z={waypoints[2][2]:.2f}m")
            else:
                print(f"  [NAV]  Drone{dn} -> Grid {grid_num:>2} ({tx:+.3f},{ty:+.3f}) at z={tz}m")

            moving.set_target(waypoints[0][0], waypoints[0][1], waypoints[0][2],
                              queue=waypoints[1:])
            continue

        print("  [ERR]  Unknown command. Examples:  1 13 0.5  /  2 home  /  status  /  land")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    n = len(ACTIVE_DRONES)
    print(f"\n[INIT]   {n} drone(s) active: {[d.name for d in ACTIVE_DRONES]}")

    for drone in ACTIVE_DRONES:
        marker_to_drone[drone.marker_id] = drone

    print_grid()

    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.rigidBodyListener = receiveRigidBodyFrame
    client.run()
    print("[MoCap]  Waiting for all rigid bodies...")

    timeout = time.time() + 10.0
    while True:
        missing = [d.name for d in ACTIVE_DRONES if not d.pose_valid]
        if not missing:
            break
        if time.time() > timeout:
            print(f"[MoCap]  ERROR: Cannot see: {missing}")
            print("         Check Motive streaming and marker IDs in ACTIVE_DRONES.")
            client.stop()
            return
        time.sleep(0.05)

    print("[MoCap]  All rigid bodies found!")

    if not sanity_check():
        client.stop()
        return

    # Save home positions and capture initial yaw as the heading to hold
    for drone in ACTIVE_DRONES:
        x, y, _, yaw, _ = drone.get_pose()
        drone.home_x   = x
        drone.home_y   = y
        drone.home_yaw = yaw
        drone.set_target(x, y, drone.default_z)
        with drone.nav_lock:
            drone.target_yaw = yaw   # hold this heading throughout flight
        print(f"[NAV]    {drone.name} home: x={x:+.3f} y={y:+.3f} z={drone.default_z}  "
              f"yaw={yaw:+.1f}deg (will hold this heading)")

    cflib.crtp.init_drivers()
    print("[CF]     Connecting to all drones...")

    status_lock = threading.Lock()
    start_kill_listener()

    print("\n  ========================================")
    print("  CTRL+X   = Emergency kill ALL drones  ")
    print("  Ctrl+C   = Graceful land ALL drones   ")
    print(f"  Drones   = {n}                        ")
    print(f"  Volume   = +/-{CLAMP_X_MAX}m XY, {CLAMP_Z_MAX}m Z")
    print("  Each drone holds its own initial heading")
    print("  ========================================\n")

    scf_list = []
    try:
        for drone in ACTIVE_DRONES:
            scf = SyncCrazyflie(drone.uri, cf=Crazyflie(rw_cache=drone.cache))
            scf.open_link()
            scf_list.append(scf)
        print(f"[CF]     All {n} drones connected!\n")

        flight_threads = []
        for drone, scf in zip(ACTIVE_DRONES, scf_list):
            t = threading.Thread(
                target=drone.flight_loop,
                args=(scf, ACTIVE_DRONES, status_lock),
                daemon=True)
            flight_threads.append(t)
            t.start()
            time.sleep(3.0)   # stagger takeoffs

        time.sleep(2.0)
        it = threading.Thread(
            target=input_thread, args=(status_lock,), daemon=True)
        it.start()

        try:
            while any(t.is_alive() for t in flight_threads):
                if kill_event.is_set():
                    for drone in ACTIVE_DRONES:
                        drone.stop_event.set()
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[CTRL]   Ctrl+C — graceful landing all drones...")
            for drone in ACTIVE_DRONES:
                drone.stop_event.set()
            for t in flight_threads:
                t.join(timeout=12)

    except Exception as e:
        print(f"[CF]     Connection error: {e}")
        for scf in scf_list:
            try:
                scf.cf.commander.send_stop_setpoint()
                scf.cf.param.set_value('stabilizer.estimator', '1')
            except Exception:
                pass

    finally:
        for scf in scf_list:
            try:
                scf.close_link()
            except Exception:
                pass

    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()