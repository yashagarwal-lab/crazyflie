#!/usr/bin/env python3
"""
N-drone MoCap rigid body waypoint navigation — curses TUI edition.
No flow deck required — position and orientation from OptiTrack rigid bodies.

Controls (curses UI):
  Arrow keys       — navigate PID table (select drone row / gain column)
  + / -            — nudge selected PID gain up / down
  : (colon)        — open command bar (same commands as before)
  CTRL+X           — emergency kill ALL drones instantly
  q                — graceful land all and quit

Command bar commands (press : to open):
  <n> <grid> <z>   — send drone N to grid position at height z  (e.g. 1 13 0.5)
  <n> home         — drone N return to its physical takeoff spot
  land             — graceful land all drones
  land <n>         — graceful land drone N only
  status           — flash status to log panel
  grid             — print grid map to log panel
"""

import math
import time
import threading
import logging
import curses
import curses.ascii

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
LOOP_HZ          = 50
EXTPOS_HZ        = 100
ARRIVAL_RADIUS   = 0.08
COLLISION_RADIUS = 0.15
HEIGHT_TOLERANCE = 0.3
AVOID_OFFSET     = 0.5

# ── Flight volume clamps ───────────────────────────────────────────────────────
CLAMP_X_MIN = -1.8
CLAMP_X_MAX = +1.8
CLAMP_Y_MIN = -1.8
CLAMP_Y_MAX = +1.8
CLAMP_Z_MIN =  0.1
CLAMP_Z_MAX =  1.8

# ── Shared kill event ─────────────────────────────────────────────────────────
kill_event = threading.Event()

# ── Log buffer (replaces print() for mid-flight messages) ────────────────────
_log_lock   = threading.Lock()
_log_lines  = []          # list of strings, newest last
LOG_MAX     = 200

def log(msg):
    """Thread-safe append to the in-memory log buffer."""
    with _log_lock:
        _log_lines.append(msg)
        if len(_log_lines) > LOG_MAX:
            _log_lines.pop(0)

def get_log_tail(n):
    with _log_lock:
        return list(_log_lines[-n:])


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
    return (max(CLAMP_X_MIN, min(CLAMP_X_MAX, x)),
            max(CLAMP_Y_MIN, min(CLAMP_Y_MAX, y)),
            max(CLAMP_Z_MIN, min(CLAMP_Z_MAX, z)))


# ── Drone class ───────────────────────────────────────────────────────────────
class Drone:
    # PID gain names in display order
    GAIN_NAMES = ['kp_xy', 'ki_xy', 'kd_xy', 'kp_z', 'ki_z', 'kd_z']
    # Nudge step per gain
    GAIN_STEP  = {'kp_xy': 0.05, 'ki_xy': 0.005, 'kd_xy': 0.01,
                  'kp_z' : 0.05, 'ki_z' : 0.005, 'kd_z' : 0.01}
    GAIN_MIN   = 0.0
    GAIN_MAX   = 5.0

    def __init__(self,
                 number,
                 marker_id,
                 default_z,
                 kp_xy=0.6, ki_xy=0.05, kd_xy=0.15,
                 kp_z=0.8,  ki_z=0.08,  kd_z=0.20):
        self.number    = number
        self.name      = f"Drone{number}"
        self.uri       = f"radio://0/80/2M/E7E7E7E70{number}"
        self.marker_id = marker_id
        self.default_z = default_z
        self.cache     = f"./cache{number}"

        # ── Pose ──────────────────────────────────────────────────────────────
        self.pose_lock  = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.qx = self.qy = self.qz = 0.0
        self.qw = 1.0
        self.pose_valid = False
        self.last_valid_time = 0.0

        # ── Nav state ─────────────────────────────────────────────────────────
        self.nav_lock       = threading.Lock()
        self.target_x       = 0.0
        self.target_y       = 0.0
        self.target_z       = default_z
        self.should_land    = False
        self.waypoint_queue = []

        self.home_x = 0.0
        self.home_y = 0.0

        # ── PID controllers ───────────────────────────────────────────────────
        self.pid_x = PID(kp_xy, ki_xy, kd_xy)
        self.pid_y = PID(kp_xy, ki_xy, kd_xy)
        self.pid_z = PID(kp_z,  ki_z,  kd_z)

        # Live-editable gain values (read by TUI, written by nudge)
        self._gain_lock = threading.Lock()
        self._gains = {
            'kp_xy': kp_xy, 'ki_xy': ki_xy, 'kd_xy': kd_xy,
            'kp_z' : kp_z,  'ki_z' : ki_z,  'kd_z' : kd_z,
        }

        self.stop_event = threading.Event()

    # ── Gain live-edit ────────────────────────────────────────────────────────
    def get_gains(self):
        with self._gain_lock:
            return dict(self._gains)

    def nudge_gain(self, name, direction):
        """Adjust a PID gain live; propagates to the running PID objects."""
        step = self.GAIN_STEP[name] * direction
        with self._gain_lock:
            new_val = max(self.GAIN_MIN,
                          min(self.GAIN_MAX, self._gains[name] + step))
            self._gains[name] = new_val
        # Push to actual PID objects
        with self.pose_lock:  # reuse pose_lock is fine here; tiny critical section
            pass
        kp_xy = self._gains['kp_xy']
        ki_xy = self._gains['ki_xy']
        kd_xy = self._gains['kd_xy']
        kp_z  = self._gains['kp_z']
        ki_z  = self._gains['ki_z']
        kd_z  = self._gains['kd_z']
        self.pid_x.kp = kp_xy; self.pid_x.ki = ki_xy; self.pid_x.kd = kd_xy
        self.pid_y.kp = kp_xy; self.pid_y.ki = ki_xy; self.pid_y.kd = kd_xy
        self.pid_z.kp = kp_z;  self.pid_z.ki = ki_z;  self.pid_z.kd = kd_z
        log(f"[PID]  {self.name} {name} = {self._gains[name]:.4f}")

    # ── Pose update ───────────────────────────────────────────────────────────
    def update_pose(self, position, rotation):
        nqx, nqy, nqz, nqw = rotation
        with self.pose_lock:
            self.x = position[2]
            self.y = position[0]
            self.z = position[1]
            self.qx = nqz
            self.qy = nqx
            self.qz = nqy
            self.qw = nqw
            self.pose_valid = True
            self.last_valid_time = time.time()

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.z, self.pose_valid

    def get_extpose(self):
        with self.pose_lock:
            return (self.x, self.y, self.z,
                    self.qx, self.qy, self.qz, self.qw,
                    self.pose_valid)

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
                    self.should_land, list(self.waypoint_queue))

    def advance_waypoint(self):
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

    # ── extpose sender ────────────────────────────────────────────────────────
    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, qx, qy, qz, qw, valid = self.get_extpose()
            if valid:
                scf.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    # ── Takeoff ───────────────────────────────────────────────────────────────
    def takeoff(self, scf):
        cf = scf.cf
        log(f"[{self.name}]  Taking off to z={self.default_z}m ...")
        for _ in range(10):
            cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
            time.sleep(0.01)
        start = time.time()
        while True:
            if kill_event.is_set():
                cf.commander.send_stop_setpoint()
                return False
            _, _, cz, _ = self.get_pose()
            if cz > CLAMP_Z_MAX:
                log(f"[{self.name}]  ALTITUDE LIMIT HIT DURING TAKEOFF — killing!")
                kill_event.set()
                cf.commander.send_stop_setpoint()
                return False
            if cz >= self.default_z * 0.90:
                log(f"[{self.name}]  Reached z={cz:.3f}m — PID taking over.")
                return True
            if time.time() - start > 8.0:
                log(f"[{self.name}]  WARNING: Takeoff timeout — continuing anyway.")
                return True
            cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
            time.sleep(0.02)

    # ── Land ──────────────────────────────────────────────────────────────────
    def land(self, scf):
        cf = scf.cf
        log(f"[{self.name}]  Descending...")
        _, _, cz, _ = self.get_pose()
        while cz > 0.10:
            if kill_event.is_set():
                break
            cf.commander.send_velocity_world_setpoint(0, 0, -0.2, 0)
            time.sleep(0.05)
            _, _, cz, _ = self.get_pose()
        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        try:
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            cf.param.set_value('stabilizer.estimator', '1')
        except Exception:
            pass
        log(f"[{self.name}]  Landed.")

    # ── Flight loop ───────────────────────────────────────────────────────────
    def flight_loop(self, scf, all_drones):
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
        log(f"[{self.name}]  extpose started. Waiting for MoCap data...")
        time.sleep(1.0)

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        log(f"[{self.name}]  EKF reset. Waiting for convergence...")
        time.sleep(1.5)

        if not self.takeoff(scf) or kill_event.is_set():
            stop_ep.set()
            return

        self.reset_pids()
        log(f"[{self.name}]  Ready for waypoints.")
        dt = 1.0 / LOOP_HZ

        try:
            while not self.stop_event.is_set() and not kill_event.is_set():
                loop_start = time.time()

                tx, ty, tz, should_land, queue = self.get_nav()
                if should_land:
                    break

                cx, cy, cz, got_data = self.get_pose()
                queue_len = len(queue)

                out_of_bounds = (
                    cx < CLAMP_X_MIN - 0.1 or cx > CLAMP_X_MAX + 0.1 or
                    cy < CLAMP_Y_MIN - 0.1 or cy > CLAMP_Y_MAX + 0.1 or
                    cz < CLAMP_Z_MIN - 0.1 or cz > CLAMP_Z_MAX + 0.1
                )
                if out_of_bounds:
                    log(f"[{self.name}]  !! OUT OF BOUNDS "
                        f"pos=({cx:+.3f},{cy:+.3f},{cz:+.3f}) — graceful landing!")
                    with self.nav_lock:
                        self.should_land = True
                    break

                with self.pose_lock:
                    last_t = self.last_valid_time
                if last_t > 0 and (time.time() - last_t) > 0.005:
                    log(f"[{self.name}]  !! MOCAP SIGNAL LOST >0.05s — graceful landing!")
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

                try:
                    cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0)
                except Exception as radio_err:
                    log(f"[{self.name}]  !! RADIO CONNECTION LOST ({radio_err}) — graceful landing!")
                    with self.nav_lock:
                        self.should_land = True
                    break

                dist    = (ex**2 + ey**2 + ez**2) ** 0.5
                arrived = dist < ARRIVAL_RADIUS

                if arrived and queue_len > 0:
                    wp = self.advance_waypoint()
                    if wp:
                        log(f"[{self.name}] Waypoint reached — "
                            f"next ({wp[0]:+.3f},{wp[1]:+.3f},{wp[2]:+.3f})")
                        self.reset_pids()

                elapsed = time.time() - loop_start
                time.sleep(max(0, dt - elapsed))

        except Exception as e:
            log(f"[{self.name}]  Flight loop error: {e}")

        if kill_event.is_set():
            cf.commander.send_stop_setpoint()
            log(f"[{self.name}]  Motors killed instantly.")
        else:
            self.land(scf)

        stop_ep.set()


# ══════════════════════════════════════════════════════════════════════════════
#   ACTIVE DRONES
# ══════════════════════════════════════════════════════════════════════════════
ACTIVE_DRONES = [

    # Drone(number=1, marker_id=351, default_z=0.5),
    # Drone(number=2, marker_id=352, default_z=0.5),
    # Drone(number=3, marker_id=353, default_z=0.5),
    Drone(number=4, marker_id=354, default_z=0.5),
    # Drone(number=5, marker_id=355, default_z=0.5),
    # Drone(number=6, marker_id=356, default_z=0.5),
    # Drone(number=7, marker_id=357, default_z=0.5),
    # Drone(number=8, marker_id=358, default_z=0.5),

]
# ══════════════════════════════════════════════════════════════════════════════


# ── NatNet callback ───────────────────────────────────────────────────────────
marker_to_drone = {}

def receiveRigidBodyFrame(rb_id, position, rotation):
    if rb_id in marker_to_drone:
        marker_to_drone[rb_id].update_pose(position, rotation)

# ── Path conflict / avoidance ─────────────────────────────────────────────────
def path_conflicts(sx, sy, tx, ty, ox, oy, radius):
    dx, dy = tx - sx, ty - sy
    seg_sq = dx*dx + dy*dy
    if seg_sq == 0:
        return ((ox-sx)**2 + (oy-sy)**2) ** 0.5 < radius
    t  = max(0.0, min(1.0, ((ox-sx)*dx + (oy-sy)*dy) / seg_sq))
    cx = sx + t*dx; cy = sy + t*dy
    return ((ox-cx)**2 + (oy-cy)**2) ** 0.5 < radius

def plan_path(moving_drone, target_x, target_y, target_z):
    cx, cy, cz, _ = moving_drone.get_pose()
    worst_z = None
    for drone in ACTIVE_DRONES:
        if drone is moving_drone:
            continue
        ox, oy, oz, _ = drone.get_pose()
        if (path_conflicts(cx, cy, target_x, target_y, ox, oy, COLLISION_RADIUS)
                and abs(cz - oz) < HEIGHT_TOLERANCE):
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

# ── Emergency kill (pynput, works even inside curses) ─────────────────────────
def on_press(key):
    if hasattr(key, 'char') and key.char == '\x18':
        log("\n!! EMERGENCY KILL — CTRL+X PRESSED !!")
        kill_event.set()

def start_kill_listener():
    listener = keyboard.Listener(on_press=on_press)
    listener.daemon = True
    listener.start()

# ── Sanity check (runs before curses, uses normal print/input) ────────────────
def sanity_check():
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║         PRE-FLIGHT SANITY CHECK          ║")
    print("  ╚══════════════════════════════════════════╝\n")
    for drone in ACTIVE_DRONES:
        x, y, z, _ = drone.get_pose()
        with drone.pose_lock:
            qx, qy, qz, qw = drone.qx, drone.qy, drone.qz, drone.qw
        yaw = math.degrees(math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz)))
        print(f"  {drone.name} (marker {drone.marker_id})  "
              f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  yaw={yaw:+.1f}deg")
    print()
    print("  Z values should all be close to 0.0 (floor level).")
    print("  Yaw is shown for reference — drones can face any direction.\n")
    confirm = input("  Do ALL positions match the physical drones? (yes/no): ").strip().lower()
    if confirm != 'yes':
        print("\n  [ABORT] Sanity check failed. Check marker IDs in ACTIVE_DRONES.\n")
        return False
    print("  [OK] Sanity check passed — proceeding to flight.\n")
    return True


# ══════════════════════════════════════════════════════════════════════════════
#   CURSES TUI
# ══════════════════════════════════════════════════════════════════════════════

# Column labels for the PID table
PID_COLS = Drone.GAIN_NAMES   # ['kp_xy','ki_xy','kd_xy','kp_z','ki_z','kd_z']

def _safe_addstr(win, row, col, text, attr=0):
    """addstr that ignores out-of-bounds writes."""
    max_y, max_x = win.getmaxyx()
    if row < 0 or row >= max_y:
        return
    avail = max_x - col
    if avail <= 0:
        return
    try:
        win.addstr(row, col, text[:avail], attr)
    except curses.error:
        pass

def draw_ui(stdscr, sel_row, sel_col, cmd_mode, cmd_buf):
    """Redraw the entire curses screen."""
    stdscr.erase()
    max_y, max_x = stdscr.getmaxyx()
    n = len(ACTIVE_DRONES)

    row = 0

    # ── Title bar ─────────────────────────────────────────────────────────────
    title = " DRONE CONTROL  |  : command  |  arrows+/-: PID  |  q: quit  |  CTRL+X: KILL"
    _safe_addstr(stdscr, row, 0, title.ljust(max_x), curses.A_REVERSE)
    row += 1

    # ── Status table ──────────────────────────────────────────────────────────
    hdr = f"  {'Drone':<8}  {'X':>7}  {'Y':>7}  {'Z':>7}  {'tX':>7}  {'tY':>7}  {'tZ':>7}  {'Dist':>6}  {'Queue':>5}  Status"
    _safe_addstr(stdscr, row, 0, hdr, curses.A_BOLD)
    row += 1
    _safe_addstr(stdscr, row, 0, "  " + "─" * min(max_x - 3, 85))
    row += 1

    for drone in ACTIVE_DRONES:
        cx, cy, cz, valid = drone.get_pose()
        tx, ty, tz, sl, wq = drone.get_nav()
        dist    = ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2) ** 0.5
        arrived = dist < ARRIVAL_RADIUS and len(wq) == 0
        status  = "LAND" if sl else ("ARRIVED" if arrived else f"q={len(wq)}")
        color   = curses.A_NORMAL
        if sl:
            color = curses.color_pair(2)  # yellow
        elif arrived:
            color = curses.color_pair(1)  # green
        line = (f"  {drone.name:<8}  {cx:+7.3f}  {cy:+7.3f}  {cz:+7.3f}"
                f"  {tx:+7.3f}  {ty:+7.3f}  {tz:+7.3f}"
                f"  {dist:6.3f}  {len(wq):>5}  {status}")
        _safe_addstr(stdscr, row, 0, line, color)
        row += 1

    row += 1

    # ── PID table ─────────────────────────────────────────────────────────────
    col_w = 9
    pid_hdr = f"  {'Drone':<8}" + "".join(f"  {g:>{col_w}}" for g in PID_COLS)
    _safe_addstr(stdscr, row, 0, pid_hdr, curses.A_BOLD)
    row += 1
    _safe_addstr(stdscr, row, 0, "  " + "─" * min(max_x - 3, 70))
    row += 1

    for r_idx, drone in enumerate(ACTIVE_DRONES):
        gains = drone.get_gains()
        line_parts = [f"  {drone.name:<8}"]
        for c_idx, gname in enumerate(PID_COLS):
            val_str = f"{gains[gname]:.4f}"
            if r_idx == sel_row and c_idx == sel_col:
                line_parts.append(f"  [{val_str:>{col_w-2}}]")
            else:
                line_parts.append(f"  {val_str:>{col_w}}")
        line = "".join(line_parts)
        _safe_addstr(stdscr, row, 0, line)
        # Highlight selected cell separately with reverse video
        if r_idx == sel_row:
            # Calculate column pixel offset for the selected cell
            base_off = 2 + 8 + 2   # "  " + drone_name + "  "
            cell_off = base_off + sel_col * (col_w + 2)
            cell_str = f"[{gains[PID_COLS[sel_col]]:.4f}]"
            _safe_addstr(stdscr, row, cell_off, cell_str, curses.A_REVERSE)
        row += 1

    row += 1
    _safe_addstr(stdscr, row, 0,
        "  PID nav: ↑↓ drone  ←→ gain  +/- nudge", curses.A_DIM)
    row += 1

    # ── Log panel ─────────────────────────────────────────────────────────────
    log_area_start = row + 1
    log_lines_available = max(0, max_y - log_area_start - 3)
    _safe_addstr(stdscr, row, 0, "  ── Log " + "─" * min(max_x - 10, 60), curses.A_DIM)
    row += 1

    tail = get_log_tail(log_lines_available)
    for msg in tail:
        _safe_addstr(stdscr, row, 2, msg[:max_x - 3], curses.A_DIM)
        row += 1
        if row >= max_y - 2:
            break

    # ── Command bar ───────────────────────────────────────────────────────────
    cmd_row = max_y - 1
    if cmd_mode:
        bar = f": {cmd_buf}"
        _safe_addstr(stdscr, cmd_row, 0, bar.ljust(max_x), curses.A_BOLD)
        # Position cursor after typed text
        try:
            stdscr.move(cmd_row, min(len(bar), max_x - 1))
        except curses.error:
            pass
    else:
        hint = "  Press : to enter a command"
        _safe_addstr(stdscr, cmd_row, 0, hint.ljust(max_x), curses.A_DIM)

    stdscr.refresh()


def handle_command(raw, drone_by_num):
    """Parse and execute a command string. Same logic as original input_thread."""
    parts = raw.strip().split()
    if not parts:
        return

    numbers = list(drone_by_num.keys())

    if parts[0].lower() == "grid":
        rows = [
            ("y=+1.0", [1,  2,  3,  4,  5]),
            ("y=+0.5", [6,  7,  8,  9, 10]),
            ("y= 0.0", [11, 12, 13, 14, 15]),
            ("y=-0.5", [16, 17, 18, 19, 20]),
            ("y=-1.0", [21, 22, 23, 24, 25]),
        ]
        log("  Grid: x=-1.0  -0.5   0.0  +0.5  +1.0")
        for label, nums in rows:
            log(f"  {label}  " + "  ".join(f"{n:>2}" for n in nums))
        return

    if parts[0].lower() == "land":
        if len(parts) == 1:
            for drone in ACTIVE_DRONES:
                with drone.nav_lock:
                    drone.should_land = True
            log("  [NAV]  Landing ALL drones.")
        elif len(parts) == 2:
            try:
                dn = int(parts[1])
                if dn in drone_by_num:
                    with drone_by_num[dn].nav_lock:
                        drone_by_num[dn].should_land = True
                    log(f"  [NAV]  Landing Drone{dn}.")
                else:
                    log(f"  [ERR]  No active drone {dn}. Active: {numbers}")
            except ValueError:
                log("  [ERR]  Usage: land  or  land <n>")
        return

    if parts[0].lower() == "status":
        for drone in ACTIVE_DRONES:
            cx, cy, cz, _ = drone.get_pose()
            with drone.nav_lock:
                tx, ty, tz = drone.target_x, drone.target_y, drone.target_z
                q = len(drone.waypoint_queue)
            dist = ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2) ** 0.5
            log(f"  [{drone.name}] pos=({cx:+.3f},{cy:+.3f},{cz:+.3f})  "
                f"tgt=({tx:+.3f},{ty:+.3f},{tz:+.3f})  dist={dist:.3f}  q={q}")
        return

    if len(parts) == 2 and parts[1].lower() == "home":
        try:
            dn = int(parts[0])
        except ValueError:
            log("  [ERR]  Usage: <n> home")
            return
        if dn not in drone_by_num:
            log(f"  [ERR]  No active drone {dn}. Active: {numbers}")
            return
        d = drone_by_num[dn]
        d.set_target(d.home_x, d.home_y, d.default_z)
        log(f"  [NAV]  Drone{dn} -> Home ({d.home_x:+.3f},{d.home_y:+.3f},{d.default_z})")
        return

    if len(parts) == 3:
        try:
            dn = int(parts[0])
        except ValueError:
            log("  [ERR]  First value must be a drone number")
            return
        if dn not in drone_by_num:
            log(f"  [ERR]  No active drone {dn}. Active: {numbers}")
            return
        try:
            grid_num = int(parts[1])
        except ValueError:
            log("  [ERR]  Grid must be a whole number 1-25")
            return
        if grid_num not in GRID:
            log(f"  [ERR]  Grid {grid_num} not valid — use 1-25")
            return
        try:
            tz = float(parts[2])
        except ValueError:
            log("  [ERR]  Height must be a number  (e.g. 1 13 0.5)")
            return

        tx, ty = GRID[grid_num]
        orig = (tx, ty, tz)
        tx, ty, tz = clamp_target(tx, ty, tz)
        if (tx, ty, tz) != orig:
            log(f"  [CLAMP] ({orig[0]:+.3f},{orig[1]:+.3f},{orig[2]:.3f}) "
                f"-> ({tx:+.3f},{ty:+.3f},{tz:.3f})")

        moving = drone_by_num[dn]
        waypoints, avoided = plan_path(moving, tx, ty, tz)
        if avoided:
            log(f"  [NAV]   Drone{dn} -> Grid {grid_num} at z={tz}m (avoid)")
            log(f"  [AVOID] Step1: climb to z={waypoints[0][2]:.2f}m")
            log(f"  [AVOID] Step2: fly to Grid {grid_num} at z={waypoints[1][2]:.2f}m")
            log(f"  [AVOID] Step3: descend to z={waypoints[2][2]:.2f}m")
        else:
            log(f"  [NAV]  Drone{dn} -> Grid {grid_num:>2} ({tx:+.3f},{ty:+.3f}) at z={tz}m")

        moving.set_target(waypoints[0][0], waypoints[0][1], waypoints[0][2],
                          queue=waypoints[1:])
        return

    log("  [ERR]  Unknown command. Examples:  1 13 0.5  /  2 home  /  land  /  status")


def curses_ui(stdscr):
    """Main curses loop — replaces input_thread."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.keypad(True)

    # Colour pairs
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN,  -1)   # arrived
    curses.init_pair(2, curses.COLOR_YELLOW, -1)   # landing

    n = len(ACTIVE_DRONES)
    drone_by_num = {d.number: d for d in ACTIVE_DRONES}

    sel_row = 0          # selected drone row in PID table
    sel_col = 0          # selected gain column in PID table
    cmd_mode = False     # True when command bar is active
    cmd_buf  = ""        # characters typed into command bar

    REFRESH_INTERVAL = 0.1   # seconds between full redraws
    last_draw = 0.0

    log("[UI]  Curses TUI active. Press : for commands, q to quit.")

    while not kill_event.is_set():
        now = time.time()
        if now - last_draw >= REFRESH_INTERVAL:
            draw_ui(stdscr, sel_row, sel_col, cmd_mode, cmd_buf)
            last_draw = now

        try:
            ch = stdscr.getch()
        except curses.error:
            ch = -1

        if ch == -1:
            time.sleep(0.02)
            continue

        # ── Command bar mode ──────────────────────────────────────────────────
        if cmd_mode:
            if ch in (curses.KEY_ENTER, ord('\n'), ord('\r')):
                handle_command(cmd_buf, drone_by_num)
                cmd_buf  = ""
                cmd_mode = False
            elif ch in (curses.KEY_BACKSPACE, 127, 8):
                cmd_buf = cmd_buf[:-1]
            elif ch == 27:          # ESC — cancel
                cmd_buf  = ""
                cmd_mode = False
            elif 32 <= ch < 127:    # printable ASCII
                cmd_buf += chr(ch)
            continue

        # ── Normal navigation mode ────────────────────────────────────────────
        if ch == ord(';'):
            cmd_mode = True
            cmd_buf  = ""

        elif ch == ord('q'):
            log("[UI]  q pressed — landing all drones.")
            for drone in ACTIVE_DRONES:
                with drone.nav_lock:
                    drone.should_land = True
            break

        elif ch == curses.KEY_UP:
            sel_row = (sel_row - 1) % n

        elif ch == curses.KEY_DOWN:
            sel_row = (sel_row + 1) % n

        elif ch == curses.KEY_LEFT:
            sel_col = (sel_col - 1) % len(PID_COLS)

        elif ch == curses.KEY_RIGHT:
            sel_col = (sel_col + 1) % len(PID_COLS)

        elif ch in (ord('+'), ord('=')):
            ACTIVE_DRONES[sel_row].nudge_gain(PID_COLS[sel_col], +1)

        elif ch in (ord('-'), ord('_')):
            ACTIVE_DRONES[sel_row].nudge_gain(PID_COLS[sel_col], -1)

    # Final redraw to show state before exit
    draw_ui(stdscr, sel_row, sel_col, False, "")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    n = len(ACTIVE_DRONES)
    print(f"\n[INIT]   {n} drone(s) active: {[d.name for d in ACTIVE_DRONES]}")

    for drone in ACTIVE_DRONES:
        marker_to_drone[drone.marker_id] = drone

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

    for drone in ACTIVE_DRONES:
        x, y, _, _ = drone.get_pose()
        drone.home_x = x
        drone.home_y = y
        drone.set_target(x, y, drone.default_z)
        print(f"[NAV]    {drone.name} home: x={x:+.3f} y={y:+.3f} z={drone.default_z}")

    cflib.crtp.init_drivers()
    print("[CF]     Connecting to all drones...")

    start_kill_listener()

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
                args=(scf, ACTIVE_DRONES),
                daemon=True)
            flight_threads.append(t)
            t.start()
            time.sleep(3.0)

        time.sleep(2.0)

        # ── Hand control to curses ─────────────────────────────────────────────
        curses.wrapper(curses_ui)

        # ── After curses exits, wait for all drones to finish ─────────────────
        print("\n[CTRL]  Waiting for drones to land...")
        try:
            while any(t.is_alive() for t in flight_threads):
                if kill_event.is_set():
                    for drone in ACTIVE_DRONES:
                        drone.stop_event.set()
                time.sleep(0.2)
        except KeyboardInterrupt:
            print("\n[CTRL]   Ctrl+C — kill all.")
            kill_event.set()
            for drone in ACTIVE_DRONES:
                drone.stop_event.set()

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

    # Dump log to terminal after curses closes
    print("\n── Flight log ─────────────────────────────────────────────────")
    for line in get_log_tail(LOG_MAX):
        print(line)
    print("── End ─────────────────────────────────────────────────────────\n")
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()