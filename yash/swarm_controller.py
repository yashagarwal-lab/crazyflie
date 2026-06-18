#!/usr/bin/env python3
"""
Backend Swarm Controller API — Modular Controller Architecture.

Supports configurable onboard controllers via drone_configs:
  'pid'         (1) — Default. Velocity setpoints + offboard PID.
  'mellinger'   (2) — Position setpoints + interpolation.
  'indi'        (3) — Velocity setpoints + offboard PID.
  'brescianini' (4) — Position setpoints + interpolation.
"""

import time
import math
import threading
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from NatNetClient import NatNetClient
from pynput import keyboard
from core.pid import PID, clamp

logging.basicConfig(level=logging.ERROR)

# ── Flight parameters ─────────────────────────────────────────────────────────
MAX_SPEED        = 0.3
LOOP_HZ          = 50
EXTPOS_HZ        = 100
BATTERY_THRESHOLD = 3.2
MOCAP_TIMEOUT    = 1.5

# ── Workspace limits ──────────────────────────────────────────────────────────
WORKSPACE_RADIUS = 1.25
WORKSPACE_Z_MAX  = 1.8
WORKSPACE_Z_MIN  = 0.05
HARD_KILL_R      = 1.55
HARD_KILL_Z      = 2.0

# ── Controller registry ──────────────────────────────────────────────────────
CONTROLLERS = {
    'pid':          {'param': '1', 'mode': 'velocity'},
    'mellinger':    {'param': '2', 'mode': 'position'},
    'indi':         {'param': '3', 'mode': 'velocity'},
    'brescianini':  {'param': '4', 'mode': 'position'},
}


class Drone:
    def __init__(self, config, controller):
        self.number    = config['number']
        self.name      = f"Drone{self.number}"
        self.uri       = config.get('uri', f"radio://0/80/2M/E7E7E7E70{self.number}")
        self.marker_id = config['marker_id']
        self.default_z = config['default_z']
        self.cache     = f"./cache{self.number}"
        self.controller = controller

        # Controller mode
        ctrl_name = config.get('controller', 'pid')
        ctrl_info = CONTROLLERS.get(ctrl_name, CONTROLLERS['pid'])
        self.ctrl_param = ctrl_info['param']
        self.ctrl_mode  = ctrl_info['mode']  # 'velocity' or 'position'
        self.ctrl_name  = ctrl_name

        # State
        self.pose_lock   = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.pose_valid  = False
        self.last_update = 0.0
        self.battery     = 4.2
        self.state       = "INIT"

        self.nav_lock    = threading.Lock()
        self.target_x    = 0.0
        self.target_y    = 0.0
        self.target_z    = self.default_z
        self.should_land = False

        # Position mode: interpolated setpoint
        self.setpoint_x = 0.0
        self.setpoint_y = 0.0
        self.setpoint_z = 0.0

        # Velocity mode: PID controllers
        self.pid_x = PID(0.6, 0.05, 0.15)
        self.pid_y = PID(0.6, 0.05, 0.15)
        self.pid_z = PID(0.8, 0.08, 0.20)

        self.stop_event = threading.Event()

    # ── Pose ──────────────────────────────────────────────────────────────
    def update_pose(self, position):
        with self.pose_lock:
            self.x, self.y, self.z = position[2], position[0], position[1]
            self.pose_valid  = True
            self.last_update = time.time()

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.z, self.pose_valid

    def get_pose_age(self):
        with self.pose_lock:
            return time.time() - self.last_update if self.last_update > 0 else float('inf')

    # ── Nav ───────────────────────────────────────────────────────────────
    def set_target(self, x, y, z):
        with self.nav_lock:
            dist = ((self.target_x-x)**2 + (self.target_y-y)**2 + (self.target_z-z)**2)**0.5
            if dist > 0.2:
                self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()
            self.target_x, self.target_y, self.target_z = x, y, z

    def get_nav(self):
        with self.nav_lock:
            return (self.target_x, self.target_y, self.target_z, self.should_land)

    def is_arrived(self, radius=0.08):
        cx, cy, cz, valid = self.get_pose()
        with self.nav_lock:
            tx, ty, tz = self.target_x, self.target_y, self.target_z
        # Clamp target to workspace for arrival check
        r = math.sqrt(tx**2 + ty**2)
        if r > WORKSPACE_RADIUS: tx *= WORKSPACE_RADIUS/r; ty *= WORKSPACE_RADIUS/r
        tz = max(WORKSPACE_Z_MIN, min(tz, WORKSPACE_Z_MAX))
        if not valid: return False
        return ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2)**0.5 < radius

    # ── Battery ───────────────────────────────────────────────────────────
    def _battery_callback(self, timestamp, data, logconf):
        voltage = data.get('pm.vbat', 4.2)
        self.battery = float(voltage)
        if self.battery < BATTERY_THRESHOLD and self.state == "FLYING":
            self.controller.log(f"[{self.name}] LOW BATTERY ({self.battery:.2f}V) - Auto Landing!")
            self.controller.fire_event("low_battery", self.number, {"voltage": self.battery})
            with self.nav_lock:
                self.should_land = True

    # ── Extpos ────────────────────────────────────────────────────────────
    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, valid = self.get_pose()
            if valid and self.get_pose_age() < 0.2:
                scf.cf.extpos.send_extpos(x, y, z)
            time.sleep(max(0, dt - (time.time() - t0)))

    # ── Takeoff ───────────────────────────────────────────────────────────
    def takeoff(self, scf):
        cf = scf.cf
        self.state = "TAKEOFF"
        self.controller.log(f"[{self.name}] Taking off ({self.ctrl_name}) to z={self.default_z}m ...")
        cx, cy, cz, _ = self.get_pose()
        ramp_dt = 0.02
        start = time.time()

        if self.ctrl_mode == 'position':
            self.setpoint_x, self.setpoint_y = cx, cy
            self.setpoint_z = max(0.05, cz)
            for _ in range(10):
                cf.commander.send_position_setpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, 0)
                time.sleep(ramp_dt)
            while self.setpoint_z < self.default_z:
                if self.controller.kill_event.is_set():
                    cf.commander.send_stop_setpoint(); return False
                _, _, az, _ = self.get_pose()
                if az > HARD_KILL_Z:
                    self.controller.kill_event.set(); cf.commander.send_stop_setpoint(); return False
                if time.time() - start > 8.0: break
                self.setpoint_z = min(self.setpoint_z + MAX_SPEED * ramp_dt, self.default_z)
                cf.commander.send_position_setpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, 0)
                time.sleep(ramp_dt)
        else:  # velocity mode
            for _ in range(10):
                cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                time.sleep(0.01)
            while True:
                if self.controller.kill_event.is_set():
                    cf.commander.send_stop_setpoint(); return False
                _, _, cz, _ = self.get_pose()
                if cz > HARD_KILL_Z:
                    self.controller.kill_event.set(); cf.commander.send_stop_setpoint(); return False
                if cz >= self.default_z * 0.90: break
                if time.time() - start > 8.0: break
                cf.commander.send_velocity_world_setpoint(0, 0, 0.3, 0)
                time.sleep(ramp_dt)

        self.controller.log(f"[{self.name}] Takeoff complete.")
        return True

    # ── Land ──────────────────────────────────────────────────────────────
    def land(self, scf):
        cf = scf.cf
        self.state = "LANDING"
        self.controller.log(f"[{self.name}] Descending...")
        ramp_dt = 0.02

        if self.ctrl_mode == 'position':
            while self.setpoint_z > 0.05:
                if self.controller.kill_event.is_set(): break
                cx, cy, _, _ = self.get_pose()
                self.setpoint_x, self.setpoint_y = cx, cy
                self.setpoint_z = max(self.setpoint_z - 0.2 * ramp_dt, 0.0)
                cf.commander.send_position_setpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, 0)
                time.sleep(ramp_dt)
        else:
            _, _, cz, _ = self.get_pose()
            while cz > 0.10:
                if self.controller.kill_event.is_set(): break
                cf.commander.send_velocity_world_setpoint(0, 0, -0.2, 0)
                time.sleep(0.05)
                _, _, cz, _ = self.get_pose()

        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        try:
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            cf.param.set_value('stabilizer.controller', '1')
            cf.param.set_value('stabilizer.estimator', '1')
        except Exception: pass
        self.state = "LANDED"
        self.controller.log(f"[{self.name}] Landed.")

    # ── Flight Loop ───────────────────────────────────────────────────────
    def flight_loop(self, scf):
        cf = scf.cf

        # Battery logging
        log_conf = LogConfig(name='Battery', period_in_ms=1000)
        log_conf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._battery_callback)
        log_conf.start()

        # Setup estimator + controller
        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('stabilizer.controller', self.ctrl_param)
        time.sleep(0.5)
        try: cf.param.set_value('flowdeck.useFlow', '0')
        except Exception: pass

        stop_ep = threading.Event()
        threading.Thread(target=self.run_extpos, args=(scf, stop_ep), daemon=True).start()
        self.controller.log(f"[{self.name}] extpos started ({self.ctrl_name} mode={self.ctrl_mode})")
        time.sleep(1.0)

        cf.param.set_value('kalman.resetEstimation', '1')
        time.sleep(0.1)
        cf.param.set_value('kalman.resetEstimation', '0')
        time.sleep(1.5)

        if not self.takeoff(scf) or self.controller.kill_event.is_set():
            stop_ep.set(); return

        self.state = "FLYING"
        self.controller.log(f"[{self.name}] Ready — {self.ctrl_name} active.")
        dt = 1.0 / LOOP_HZ
        was_arrived = False
        self.pid_x.reset(); self.pid_y.reset(); self.pid_z.reset()

        try:
            while not self.stop_event.is_set() and not self.controller.kill_event.is_set():
                loop_start = time.time()
                tx, ty, tz, should_land = self.get_nav()
                if should_land: break

                cx, cy, cz, got_data = self.get_pose()
                age = self.get_pose_age()

                if age > MOCAP_TIMEOUT:
                    self.controller.log(f"[{self.name}] MOCAP LOST — AUTO LANDING")
                    self.controller.fire_event("mocap_lost", self.number, {})
                    with self.nav_lock: self.should_land = True
                    break

                if (cx**2+cy**2) > HARD_KILL_R**2 or cz > HARD_KILL_Z:
                    self.controller.log(f"[{self.name}] HARD BOUNDARY — KILLING")
                    self.controller.kill_event.set(); break

                if age > 0.2:
                    if self.ctrl_mode == 'position':
                        cf.commander.send_position_setpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, 0)
                    else:
                        cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                    time.sleep(dt); continue

                if self.ctrl_mode == 'position':
                    # Interpolate setpoint toward target
                    ex, ey, ez = tx - self.setpoint_x, ty - self.setpoint_y, tz - self.setpoint_z
                    dist = math.sqrt(ex**2 + ey**2 + ez**2)
                    step = MAX_SPEED * dt
                    if dist > step:
                        self.setpoint_x += (ex/dist)*step
                        self.setpoint_y += (ey/dist)*step
                        self.setpoint_z += (ez/dist)*step
                    else:
                        self.setpoint_x, self.setpoint_y, self.setpoint_z = tx, ty, tz

                    # Anti-windup clamp
                    dd = math.sqrt((self.setpoint_x-cx)**2+(self.setpoint_y-cy)**2+(self.setpoint_z-cz)**2)
                    if dd > 0.3:
                        s = 0.3/dd
                        self.setpoint_x = cx+(self.setpoint_x-cx)*s
                        self.setpoint_y = cy+(self.setpoint_y-cy)*s
                        self.setpoint_z = cz+(self.setpoint_z-cz)*s

                    # Boundary clamp
                    r = math.sqrt(self.setpoint_x**2+self.setpoint_y**2)
                    if r > WORKSPACE_RADIUS:
                        self.setpoint_x *= WORKSPACE_RADIUS/r
                        self.setpoint_y *= WORKSPACE_RADIUS/r
                    self.setpoint_z = max(WORKSPACE_Z_MIN, min(self.setpoint_z, WORKSPACE_Z_MAX))

                    cf.commander.send_position_setpoint(self.setpoint_x, self.setpoint_y, self.setpoint_z, 0)

                else:  # velocity mode
                    now = time.time()
                    ex, ey, ez = tx - cx, ty - cy, tz - cz
                    vx = clamp(self.pid_x.update(ex, now), MAX_SPEED)
                    vy = clamp(self.pid_y.update(ey, now), MAX_SPEED)
                    vz = clamp(self.pid_z.update(ez, now), MAX_SPEED)

                    # Boundary clamp on velocity
                    r = math.sqrt(cx**2+cy**2)
                    if r > WORKSPACE_RADIUS * 0.95:
                        dot = cx*vx + cy*vy
                        if dot > 0: vx, vy = 0.0, 0.0
                    if cz > WORKSPACE_Z_MAX * 0.95 and vz > 0: vz = 0.0
                    if cz < WORKSPACE_Z_MIN + 0.02 and vz < 0: vz = 0.0

                    cf.commander.send_velocity_world_setpoint(vx, vy, vz, 0)

                # Arrival event
                currently_arrived = self.is_arrived()
                if currently_arrived and not was_arrived:
                    self.controller.fire_event("arrived", self.number, {"pos": (cx, cy, cz)})
                was_arrived = currently_arrived

                time.sleep(max(0, dt - (time.time() - loop_start)))

        except Exception as e:
            self.controller.log(f"[{self.name}] Flight loop error: {e}")

        if self.controller.kill_event.is_set():
            cf.commander.send_stop_setpoint()
            self.state = "KILLED"
        else:
            self.land(scf)

        log_conf.stop()
        stop_ep.set()


class SwarmController:
    def __init__(self, drone_configs, logging_callback=None, event_callback=None):
        self.kill_event = threading.Event()
        self.drones = []
        self.drone_by_num = {}
        self.marker_to_drone = {}
        self.logging_callback = logging_callback
        self.event_callback = event_callback

        for cfg in drone_configs:
            d = Drone(config=cfg, controller=self)
            self.drones.append(d)
            self.drone_by_num[d.number] = d
            self.marker_to_drone[d.marker_id] = d

        self.scf_list = []
        self.flight_threads = []
        self.mocap_client = None

    def log(self, msg):
        if self.logging_callback: self.logging_callback(msg)
        else: print(msg)

    def fire_event(self, event_name, drone_number, data):
        if self.event_callback: self.event_callback(event_name, drone_number, data)

    def receive_rigid_body_frame(self, rb_id, position, rotation):
        if rb_id in self.marker_to_drone:
            self.marker_to_drone[rb_id].update_pose(position)

    def on_press(self, key):
        if hasattr(key, 'char') and key.char == '\x18':
            self.log("!! EMERGENCY KILL — CTRL+X !!")
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
            x, y, z, _ = drone.get_pose()
            print(f"  {drone.name} ({drone.ctrl_name})  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}")
        print("\n  Z should be ~0.0 (floor). Confirm positions match physical drones.\n")
        if input("  All correct? (yes/no): ").strip().lower() != 'yes':
            print("\n  [ABORT]\n"); return False
        print("  [OK]\n"); return True

    def start(self, interactive=True):
        n = len(self.drones)
        self.log(f"\n[INIT] {n} drone(s)")
        self.log("[MoCap] Connecting...")
        self.mocap_client = NatNetClient()
        self.mocap_client.rigidBodyListener = self.receive_rigid_body_frame
        self.mocap_client.run()

        timeout = time.time() + 10.0
        while True:
            missing = [d.name for d in self.drones if not d.pose_valid]
            if not missing: break
            if time.time() > timeout:
                self.log(f"[MoCap] ERROR: Cannot see: {missing}")
                self.mocap_client.stop(); return False
            time.sleep(0.05)
        self.log("[MoCap] All bodies found!")

        if interactive and not self.sanity_check():
            self.mocap_client.stop(); return False

        for drone in self.drones:
            x, y, _, _ = drone.get_pose()
            drone.set_target(x, y, drone.default_z)

        cflib.crtp.init_drivers()
        self.start_kill_listener()

        try:
            for drone in self.drones:
                connected = False
                for attempt in range(3):
                    try:
                        scf = SyncCrazyflie(drone.uri, cf=Crazyflie(rw_cache=drone.cache))
                        scf.open_link()
                        self.scf_list.append(scf); connected = True; break
                    except Exception as e:
                        self.log(f"[CF] {drone.name} attempt {attempt+1} failed: {e}")
                        time.sleep(1.0)
                if not connected:
                    raise Exception(f"Failed to connect {drone.name}")

            self.log(f"[CF] All {n} connected!\n")
            for drone, scf in zip(self.drones, self.scf_list):
                t = threading.Thread(target=drone.flight_loop, args=(scf,), daemon=True)
                self.flight_threads.append(t); t.start(); time.sleep(3.0)
            return True
        except Exception as e:
            self.log(f"[CF] Fatal: {e}"); self.stop_all(); return False

    def stop_all(self):
        self.kill_event.set()
        for scf in self.scf_list:
            try:
                scf.cf.commander.send_stop_setpoint()
                scf.cf.param.set_value('stabilizer.controller', '1')
                scf.cf.param.set_value('stabilizer.estimator', '1')
            except Exception: pass
        for scf in self.scf_list:
            try: scf.close_link()
            except Exception: pass
        if self.mocap_client: self.mocap_client.stop()

    def land_all(self):
        for d in self.drones:
            with d.nav_lock: d.should_land = True
        self.log("[NAV] Landing ALL.")

    def land(self, drone_number):
        if drone_number in self.drone_by_num:
            with self.drone_by_num[drone_number].nav_lock:
                self.drone_by_num[drone_number].should_land = True

    def goto_point(self, drone_number, x, y, z):
        if drone_number not in self.drone_by_num: return False
        if z > WORKSPACE_Z_MAX or z < 0.1: return False
        self.drone_by_num[drone_number].set_target(x, y, z)
        return True

    def is_arrived(self, drone_number, radius=0.08):
        if drone_number not in self.drone_by_num: return False
        return self.drone_by_num[drone_number].is_arrived(radius)

    def get_state(self):
        state = []
        for d in self.drones:
            cx, cy, cz, valid = d.get_pose()
            with d.nav_lock: tx, ty, tz = d.target_x, d.target_y, d.target_z
            state.append({
                'number': d.number, 'name': d.name,
                'pos': (cx, cy, cz) if valid else None,
                'target': (tx, ty, tz), 'battery': d.battery,
                'state': d.state, 'arrived': d.is_arrived(),
                'controller': d.ctrl_name,
            })
        return state

    def all_landed(self):
        return len(self.flight_threads) > 0 and all(not t.is_alive() for t in self.flight_threads)

    def wait_for_landing(self):
        for d in self.drones: d.stop_event.set()
        try:
            while any(t.is_alive() for t in self.flight_threads):
                if self.kill_event.is_set(): break
                time.sleep(0.2)
        except KeyboardInterrupt: self.kill_event.set()
        for t in self.flight_threads: t.join(timeout=5)
        self.stop_all()
