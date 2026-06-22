#!/usr/bin/env python3
"""
N-drone MoCap rigid body waypoint navigation with CBF safety.

This module provides the `SafeSwarmController` class, which handles:
  - Connection to OptiTrack (NatNetClient).
  - Individual drone initialization and communication via `cflib`.
  - Control Barrier Function (CBF) safety filtering for workspace bounds and inter-drone separation.
  - Interactive terminal mode for manual testing using a static curses UI.
"""

import time
import threading
import logging
import curses
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from NatNetClient import NatNetClient
from pynput import keyboard

from core.pid import PID, clamp
from core.grid import GRID, print_grid
from core.cbf import CBFSafetyFilter
import math

def quat_to_yaw(qx, qy, qz, qw):
    siny_cosp = 2.0 * (qw * qy + qz * qx)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.degrees(math.atan2(siny_cosp, cosy_cosp))

def wrap_angle(angle):
    while angle >  180.0: angle -= 360.0
    while angle < -180.0: angle += 360.0
    return angle

logging.basicConfig(level=logging.ERROR)

# ── Global flight parameters ──────────────────────────────────────────────────
MAX_SPEED        = 0.3
LOOP_HZ          = 50
EXTPOS_HZ        = 100
MAX_ALTITUDE     = 1.8
ARRIVAL_RADIUS   = 0.08
COLLISION_RADIUS = 0.15
HEIGHT_TOLERANCE = 0.3    
AVOID_OFFSET     = 0.5    

# ── Cylindrical workspace (CBF enforced) ──────────────────────────────────────
WORKSPACE_RADIUS = 1.25   
WORKSPACE_Z_MAX  = 1.8    
WORKSPACE_Z_MIN  = 0.05   
CBF_D_MIN        = 0.3    
CBF_ALPHA_BOUND  = 1.0    
CBF_ALPHA_SEP    = 0.8    
MARKER_TIMEOUT   = 0.5    
SOFT_RADIUS      = 1.0    

# ── Hard emergency kill radii ─────────────────────────────────────────────────
HARD_KILL_R      = 1.55   
HARD_KILL_Z      = 2.0    


class Drone:
    def __init__(self, number, marker_id, default_z, controller, kp_xy=0.6, ki_xy=0.05, kd_xy=0.15, kp_z=0.8, ki_z=0.08, kd_z=0.20):
        self.number    = number
        self.name      = f"Drone{number}"
        self.uri       = f"radio://0/80/2M/E7E7E7E70{number}"
        self.marker_id = marker_id
        self.default_z = default_z
        self.cache     = f"./cache{number}"
        self.controller = controller

        self.pose_lock   = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.qx = self.qy = self.qz = 0.0
        self.qw = 1.0
        self.yaw = 0.0
        self.pose_valid  = False
        self.last_update = 0.0

        self.nav_lock       = threading.Lock()
        self.target_x       = 0.0
        self.target_y       = 0.0
        self.target_z       = default_z
        self.target_yaw     = 0.0
        self.should_land    = False
        self.waypoint_queue = []

        self.home_x = 0.0
        self.home_y = 0.0
        self.home_yaw = 0.0

        self.pid_x = PID(kp_xy, ki_xy, kd_xy)
        self.pid_y = PID(kp_xy, ki_xy, kd_xy)
        self.pid_z = PID(kp_z,  ki_z,  kd_z)
        self.pid_yaw = PID(2.0, 0.05, 0.1, integral_limit=30.0)

        self.stop_event = threading.Event()

    def update_pose(self, position, rotation):
        qx, qy, qz, qw = rotation
        yaw = quat_to_yaw(qx, qy, qz, qw)
        with self.pose_lock:
            self.x = position[2]
            self.y = position[0]
            self.z = position[1]
            self.qx = qx
            self.qy = qy
            self.qz = qz
            self.qw = qw
            self.yaw = yaw
            self.pose_valid  = True
            self.last_update = time.time()

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.z, self.yaw, self.pose_valid

    def get_orientation(self):
        with self.pose_lock:
            return self.qx, self.qy, self.qz, self.qw

    def get_pose_age(self):
        with self.pose_lock:
            return time.time() - self.last_update if self.last_update > 0 else float('inf')

    def set_target(self, x, y, z, queue=None):
        with self.nav_lock:
            self.target_x = x
            self.target_y = y
            self.target_z = z
            self.waypoint_queue = queue if queue else []

    def get_nav(self):
        with self.nav_lock:
            return (self.target_x, self.target_y, self.target_z, self.target_yaw, self.should_land, list(self.waypoint_queue))

    def advance_waypoint(self):
        with self.nav_lock:
            if self.waypoint_queue:
                wp = self.waypoint_queue.pop(0)
                self.target_x, self.target_y, self.target_z = wp
                return wp
        return None

    def reset_pids(self):
        self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset(); self.pid_yaw.reset()

    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, _, valid = self.get_pose()
            qx, qy, qz, qw = self.get_orientation()
            if valid:
                scf.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    def takeoff(self, scf):
        cf = scf.cf
        _, _, cz, cyaw, _ = self.get_pose()
        with self.nav_lock:
            self.target_yaw = cyaw
        self.controller.log(f"[{self.name}] Taking off to z={self.default_z}m, yaw={cyaw:.1f}deg...")
        for _ in range(10):
            cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
            time.sleep(0.01)
        start = time.time()
        while True:
            if self.controller.kill_event.is_set():
                cf.commander.send_stop_setpoint()
                return False
            _, _, cz, _, _ = self.get_pose()
            if cz > MAX_ALTITUDE:
                self.controller.log(f"[{self.name}] ALTITUDE LIMIT HIT DURING TAKEOFF — killing!")
                self.controller.kill_event.set()
                cf.commander.send_stop_setpoint()
                return False
            if cz >= self.default_z * 0.90:
                self.controller.log(f"[{self.name}] Reached z={cz:.3f}m — PID taking over.")
                return True
            if time.time() - start > 8.0:
                self.controller.log(f"[{self.name}] WARNING: Takeoff timeout — continuing anyway.")
                return True
            cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
            time.sleep(0.02)

    def land(self, scf):
        cf = scf.cf
        self.controller.log(f"[{self.name}] Descending...")
        _, _, cz, _, _ = self.get_pose()
        while cz > 0.10:
            if self.controller.kill_event.is_set():
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
        self.controller.log(f"[{self.name}] Landed.")

    def flight_loop(self, scf):
        cf = scf.cf

        cf.param.set_value('stabilizer.estimator', '2')
        time.sleep(0.5)
        try: cf.param.set_value('flowdeck.useFlow', '0')
        except Exception: pass

        stop_ep = threading.Event()
        ep = threading.Thread(target=self.run_extpos, args=(scf, stop_ep), daemon=True)
        ep.start()
        self.controller.log(f"[{self.name}] extpos started. Waiting for MoCap data...")
        time.sleep(1.0)

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        self.controller.log(f"[{self.name}] EKF reset. Waiting for convergence...")
        time.sleep(1.5)

        if not self.takeoff(scf) or self.controller.kill_event.is_set():
            stop_ep.set()
            return

        self.reset_pids()
        self.controller.log(f"[{self.name}] Ready for waypoints.")
        dt = 1.0 / LOOP_HZ

        try:
            while not self.stop_event.is_set() and not self.controller.kill_event.is_set():
                loop_start = time.time()

                tx, ty, tz, target_yaw, should_land, queue = self.get_nav()
                if should_land:
                    break

                cx, cy, cz, cyaw, got_data = self.get_pose()
                queue_len = len(queue)

                if got_data and self.get_pose_age() > MARKER_TIMEOUT:
                    self.controller.log(f"[{self.name}] !! MARKER LOST for >{MARKER_TIMEOUT}s — AUTO LANDING !!")
                    with self.nav_lock:
                        self.should_land = True
                    break

                if (cx**2 + cy**2) > HARD_KILL_R**2 or cz > HARD_KILL_Z:
                    self.controller.log(f"[{self.name}] !! HARD BOUNDARY BREACH pos=({cx:+.3f},{cy:+.3f},{cz:+.3f}) — KILLING !!")
                    self.controller.kill_event.set()
                    break

                if not got_data:
                    cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    time.sleep(dt)
                    continue

                # SOFT_RADIUS 'return home' logic removed.
                # The CBF filter will now naturally hold the drone at the boundary constraint.

                now = time.time()
                ex, ey, ez = tx - cx, ty - cy, tz - cz

                vx = clamp(self.pid_x.update(ex, now), MAX_SPEED)
                vy = clamp(self.pid_y.update(ey, now), MAX_SPEED)
                vz = clamp(self.pid_z.update(ez, now), MAX_SPEED)

                yaw_error = wrap_angle(target_yaw - cyaw)
                yaw_rate_cmd = clamp(self.pid_yaw.update(yaw_error, now), 100.0)

                other_poses = []
                for d in self.controller.drones:
                    if d is not self:
                        ox, oy, oz, _, _ = d.get_pose()
                        other_poses.append((ox, oy, oz))
                        
                vx, vy, vz = self.controller.cbf_filter.filter(
                    pos=(cx, cy, cz),
                    v_des=(vx, vy, vz),
                    other_positions=other_poses,
                )

                cf.commander.send_velocity_world_setpoint(vx, vy, vz, yaw_rate_cmd)

                dist    = (ex**2 + ey**2 + ez**2) ** 0.5
                arrived = dist < ARRIVAL_RADIUS

                if arrived and queue_len > 0:
                    wp = self.advance_waypoint()
                    if wp:
                        self.controller.log(f"[{self.name}] Waypoint reached — next ({wp[0]:+.3f},{wp[1]:+.3f},{wp[2]:+.3f})")
                        self.reset_pids()

                elapsed = time.time() - loop_start
                time.sleep(max(0, dt - elapsed))

        except Exception as e:
            self.controller.log(f"[{self.name}] Flight loop error: {e}")

        if self.controller.kill_event.is_set():
            cf.commander.send_stop_setpoint()
            self.controller.log(f"[{self.name}] Motors killed instantly.")
        else:
            self.land(scf)

        stop_ep.set()


class SafeSwarmController:
    def __init__(self, drone_configs):
        self.kill_event = threading.Event()
        self.drones = []
        self.drone_by_num = {}
        self.marker_to_drone = {}
        self.messages = []
        self.msg_lock = threading.Lock()
        
        self.cbf_filter = CBFSafetyFilter(
            radius=WORKSPACE_RADIUS,
            z_max=WORKSPACE_Z_MAX,
            z_min=WORKSPACE_Z_MIN,
            d_min=CBF_D_MIN,
            alpha_boundary=CBF_ALPHA_BOUND,
            alpha_separation=CBF_ALPHA_SEP,
            max_speed=MAX_SPEED,
        )

        for cfg in drone_configs:
            d = Drone(
                number=cfg['number'],
                marker_id=cfg['marker_id'],
                default_z=cfg['default_z'],
                controller=self,
                kp_xy=cfg.get('kp_xy', 0.6), ki_xy=cfg.get('ki_xy', 0.05), kd_xy=cfg.get('kd_xy', 0.15),
                kp_z=cfg.get('kp_z', 0.8), ki_z=cfg.get('ki_z', 0.08), kd_z=cfg.get('kd_z', 0.20)
            )
            self.drones.append(d)
            self.drone_by_num[d.number] = d
            self.marker_to_drone[d.marker_id] = d

        self.scf_list = []
        self.flight_threads = []
        self.mocap_client = None

    def log(self, msg):
        """Store logs for the curses UI."""
        with self.msg_lock:
            self.messages.append(msg)
            if len(self.messages) > 10:
                self.messages.pop(0)

    def receive_rigid_body_frame(self, rb_id, position, rotation):
        if rb_id in self.marker_to_drone:
            self.marker_to_drone[rb_id].update_pose(position, rotation)

    def _path_conflicts(self, sx, sy, tx, ty, ox, oy, radius):
        dx, dy = tx - sx, ty - sy
        seg_sq = dx*dx + dy*dy
        if seg_sq == 0:
            return ((ox-sx)**2 + (oy-sy)**2) ** 0.5 < radius
        t  = max(0.0, min(1.0, ((ox-sx)*dx + (oy-sy)*dy) / seg_sq))
        cx = sx + t*dx
        cy = sy + t*dy
        return ((ox-cx)**2 + (oy-cy)**2) ** 0.5 < radius

    def _plan_path(self, moving_drone, target_x, target_y, target_z):
        cx, cy, cz, _, _ = moving_drone.get_pose()
        worst_z = None

        for drone in self.drones:
            if drone is moving_drone:
                continue
            ox, oy, oz, _, _ = drone.get_pose()
            conflict       = self._path_conflicts(cx, cy, target_x, target_y, ox, oy, COLLISION_RADIUS)
            height_similar = abs(cz - oz) < HEIGHT_TOLERANCE
            if conflict and height_similar:
                if worst_z is None or oz > worst_z:
                    worst_z = oz

        if worst_z is None:
            return [(target_x, target_y, target_z)], False

        avoid_z = min(worst_z + AVOID_OFFSET, MAX_ALTITUDE - 0.1)
        return [
            (cx,       cy,       avoid_z),
            (target_x, target_y, avoid_z),
            (target_x, target_y, target_z),
        ], True

    def on_press(self, key):
        if hasattr(key, 'char') and key.char == '\x18':
            self.log("!! EMERGENCY KILL — CTRL+X PRESSED !!")
            self.kill_event.set()

    def start_kill_listener(self):
        listener = keyboard.Listener(on_press=self.on_press)
        listener.daemon = True
        listener.start()

    def sanity_check(self):
        print("\n  ╔══════════════════════════════════════════╗")
        print("  ║         PRE-FLIGHT SANITY CHECK          ║")
        print("  ╚══════════════════════════════════════════╝\n")
        for drone in self.drones:
            x, y, z, yaw, _ = drone.get_pose()
            print(f"  {drone.name} (marker {drone.marker_id})  "
                  f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  yaw={yaw:+.1f}deg")
        print()
        print("  Z values should all be close to 0.0 (floor level).")
        print("  Confirm each position matches where that drone is physically sitting.\n")
        confirm = input("  Do ALL positions match the physical drones? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("\n  [ABORT] Sanity check failed. Check marker IDs.\n")
            return False
        print("  [OK] Sanity check passed — proceeding to flight.\n")
        return True

    def start(self, interactive=False):
        n = len(self.drones)
        print(f"\n[INIT]   {n} drone(s) initialized.")
        if interactive:
            print_grid()

        print("[MoCap]  Connecting to Motive NatNet...")
        self.mocap_client = NatNetClient()
        self.mocap_client.rigidBodyListener = self.receive_rigid_body_frame
        self.mocap_client.run()
        print("[MoCap]  Waiting for all rigid bodies...")

        timeout = time.time() + 10.0
        while True:
            missing = [d.name for d in self.drones if not d.pose_valid]
            if not missing:
                break
            if time.time() > timeout:
                print(f"[MoCap]  ERROR: Cannot see: {missing}")
                self.mocap_client.stop()
                return False
            time.sleep(0.05)
        print("[MoCap]  All rigid bodies found!")

        if interactive and not self.sanity_check():
            self.mocap_client.stop()
            return False

        for drone in self.drones:
            x, y, _, yaw, _ = drone.get_pose()
            drone.home_x = x
            drone.home_y = y
            drone.home_yaw = yaw
            drone.set_target(x, y, drone.default_z)
            with drone.nav_lock:
                drone.target_yaw = yaw
            print(f"[NAV]    {drone.name} home: x={x:+.3f} y={y:+.3f} z={drone.default_z} yaw={yaw:+.1f}deg")

        cflib.crtp.init_drivers()
        print("[CF]     Connecting to all drones...")
        self.start_kill_listener()

        try:
            for drone in self.drones:
                scf = SyncCrazyflie(drone.uri, cf=Crazyflie(rw_cache=drone.cache))
                scf.open_link()
                self.scf_list.append(scf)
            print(f"[CF]     All {n} drones connected!\n")

            for drone, scf in zip(self.drones, self.scf_list):
                t = threading.Thread(target=drone.flight_loop, args=(scf,), daemon=True)
                self.flight_threads.append(t)
                t.start()
                time.sleep(3.0)

            return True

        except Exception as e:
            print(f"[CF]     Connection error: {e}")
            self.stop_all()
            return False

    def stop_all(self):
        self.kill_event.set()
        for scf in self.scf_list:
            try:
                scf.cf.commander.send_stop_setpoint()
                scf.cf.param.set_value('stabilizer.estimator', '1')
            except Exception: pass
        for scf in self.scf_list:
            try: scf.close_link()
            except Exception: pass
        if self.mocap_client:
            self.mocap_client.stop()

    def land_all(self):
        for drone in self.drones:
            with drone.nav_lock:
                drone.should_land = True
        self.log("[NAV] Landing ALL drones.")

    def land(self, drone_number):
        if drone_number in self.drone_by_num:
            with self.drone_by_num[drone_number].nav_lock:
                self.drone_by_num[drone_number].should_land = True
            self.log(f"[NAV] Landing Drone{drone_number}.")

    def goto_grid(self, drone_number, grid_number, z):
        if drone_number not in self.drone_by_num: return False
        if grid_number not in GRID: return False
        tx, ty = GRID[grid_number]
        return self.goto_point(drone_number, tx, ty, z, grid_num=grid_number)

    def goto_point(self, drone_number, x, y, z, grid_num=None):
        if drone_number not in self.drone_by_num: return False
        if z > MAX_ALTITUDE or z < 0.1: return False

        moving = self.drone_by_num[drone_number]
        waypoints, avoided = self._plan_path(moving, x, y, z)

        if avoided:
            self.log(f"[AVOID] Drone{drone_number} path conflict detected")
            self.log(f"[AVOID] Steps: climb to z={waypoints[0][2]:.2f}m, then fly, then descend.")
        
        target_desc = f"Grid {grid_num}" if grid_num else f"Point ({x:+.3f},{y:+.3f})"
        self.log(f"[NAV] Drone{drone_number} → {target_desc} at z={z}m")

        moving.set_target(waypoints[0][0], waypoints[0][1], waypoints[0][2], queue=waypoints[1:])
        return True

    def go_home(self, drone_number):
        if drone_number in self.drone_by_num:
            d = self.drone_by_num[drone_number]
            d.set_target(d.home_x, d.home_y, d.default_z)
            self.log(f"[NAV] Drone{drone_number} → Home ({d.home_x:+.3f},{d.home_y:+.3f},{d.default_z})")

    def process_command(self, raw):
        parts = raw.split()
        if not parts: return
        
        if parts[0].lower() == "land":
            if len(parts) == 1: self.land_all()
            elif len(parts) == 2:
                try: self.land(int(parts[1]))
                except ValueError: self.log("[ERR] Usage: land or land <n>")
            return

        if len(parts) == 2 and parts[1].lower() == "home":
            try: self.go_home(int(parts[0]))
            except ValueError: self.log("[ERR] Usage: <n> home")
            return

        if len(parts) == 3:
            try:
                dn = int(parts[0])
                grid_num = int(parts[1])
                tz = float(parts[2])
                if not self.goto_grid(dn, grid_num, tz):
                    self.log("[ERR] Invalid command. Check drone number, grid number, or height.")
            except ValueError:
                self.log("[ERR] Usage: <n> <grid> <z> (e.g. 1 13 0.5)")
            return

        self.log(f"[ERR] Unknown command: {raw}")

    def _curses_ui(self, stdscr):
        curses.curs_set(1)  # Show cursor for input
        stdscr.nodelay(True)
        stdscr.timeout(100) # 10Hz refresh
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_GREEN, -1)
        curses.init_pair(2, curses.COLOR_RED, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_CYAN, -1)

        input_str = ""

        while True:
            # Check exit conditions
            if self.kill_event.is_set():
                self.log("Emergency kill engaged! Exiting interactive terminal.")
                time.sleep(1.0)
                break
                
            all_landed = all(not t.is_alive() for t in self.flight_threads)
            if all_landed and len(self.flight_threads) > 0:
                self.log("All drones landed. Exiting interactive terminal.")
                time.sleep(1.0)
                break

            stdscr.erase()
            h, w = stdscr.getmaxyx()

            # --- HEADER ---
            stdscr.addstr(0, 0, "=== Safe Swarm Interactive Terminal ===", curses.color_pair(1) | curses.A_BOLD)
            stdscr.addstr(1, 0, "Commands:  <n> <grid> <z> | <n> home | land | land <n> | Ctrl+C to quit")
            
            # --- DRONE STATUS ---
            row = 3
            for drone in self.drones:
                cx, cy, cz, cyaw, valid = drone.get_pose()
                with drone.nav_lock:
                    tx, ty, tz = drone.target_x, drone.target_y, drone.target_z
                    q = len(drone.waypoint_queue)
                dist = ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2) ** 0.5
                nearest = min(GRID.items(), key=lambda g: (g[1][0]-cx)**2 + (g[1][1]-cy)**2)[0]
                
                status_color = curses.color_pair(1) if valid else curses.color_pair(2)
                pos_str = f"({cx:+.2f}, {cy:+.2f}, {cz:+.2f})" if valid else "( NO MOCAP )"
                tgt_str = f"Grid{nearest:>2} ({tx:+.2f}, {ty:+.2f}, {tz:+.2f})"
                
                info = f"[{drone.name}] Pos: {pos_str:20} Tgt: {tgt_str:20} Dist: {dist:.2f}m  Q: {q}"
                if row < h - 4:
                    stdscr.addstr(row, 0, info, status_color)
                row += 1

            # --- LOGS ---
            row += 1
            if row < h - 4:
                stdscr.addstr(row, 0, "--- Event Log ---", curses.color_pair(3))
                row += 1
                with self.msg_lock:
                    # show last messages fitting in screen
                    msgs = self.messages[-(h - row - 3):] if h - row - 3 > 0 else []
                    for msg in msgs:
                        stdscr.addstr(row, 0, msg[:w-1])
                        row += 1

            # --- INPUT PROMPT ---
            prompt_row = h - 1
            stdscr.addstr(prompt_row, 0, f"> {input_str}")
            
            stdscr.move(prompt_row, 2 + len(input_str))
            stdscr.refresh()

            # Handle Input
            try:
                key = stdscr.getch()
                if key != -1:
                    if key in (curses.KEY_ENTER, 10, 13):
                        if input_str.strip():
                            self.process_command(input_str.strip())
                        input_str = ""
                    elif key in (curses.KEY_BACKSPACE, 127, 8):
                        input_str = input_str[:-1]
                    elif 32 <= key <= 126:
                        input_str += chr(key)
            except KeyboardInterrupt:
                break
            except Exception:
                pass

    def run_interactive(self):
        try:
            curses.wrapper(self._curses_ui)
        except Exception as e:
            print(f"Curses UI failed: {e}")
        
        # Wait for flight threads to finish landing
        print("\n[CTRL] Graceful landing all drones...")
        for drone in self.drones:
            drone.stop_event.set()
            
        try:
            while any(t.is_alive() for t in self.flight_threads):
                if self.kill_event.is_set():
                    break
                time.sleep(0.2)
        except KeyboardInterrupt:
            self.kill_event.set()

        for t in self.flight_threads:
            t.join(timeout=5)

        self.stop_all()


if __name__ == '__main__':
    drone_configs = [
        {'number': 1, 'marker_id': 351, 'default_z': 0.5},
        # {'number': 2, 'marker_id': 352, 'default_z': 0.5},
    ]

    controller = SafeSwarmController(drone_configs)
    
    if controller.start(interactive=True):
        controller.run_interactive()