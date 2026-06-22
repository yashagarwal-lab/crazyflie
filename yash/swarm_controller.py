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
from core.state import DroneState
from core.pipeline import MellingerController, CBFSafetyWrapper, CBFSafetyFilter

logging.basicConfig(level=logging.ERROR)

# ── Flight parameters ─────────────────────────────────────────────────────────
MAX_SPEED        = 0.3
LOOP_HZ          = 50
EXTPOS_HZ        = 100
BATTERY_THRESHOLD = 3.2
MOCAP_TIMEOUT    = 1.5

# ── Workspace limits ──────────────────────────────────────────────────────────
WORKSPACE_RADIUS = 1.5
WORKSPACE_Z_MAX  = 2.0
WORKSPACE_Z_MIN  = 0.05
HARD_KILL_R      = 1.7
HARD_KILL_Z      = 2.2

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
        ctrl_name = config.get('controller', 'mellinger')
        ctrl_info = CONTROLLERS.get(ctrl_name, CONTROLLERS['pid'])
        self.ctrl_param = ctrl_info['param']
        self.ctrl_mode  = ctrl_info['mode']  # 'velocity' or 'position'
        self.ctrl_name  = ctrl_name

        # State (Modular)
        self.drone_state = DroneState()

        # Navigation state
        self.nav_lock    = threading.Lock()
        self.target_x    = 0.0
        self.target_y    = 0.0
        self.target_z    = self.default_z
        self.target_yaw  = 0.0
        self.should_land = False

        # Flight Pipeline
        # By default, use Mellinger Controller
        base_controller = MellingerController(max_speed=MAX_SPEED)
        
        # If config requests CBF, wrap it
        if config.get('use_cbf', False):
            cbf_filter = CBFSafetyFilter(
                v_max=MAX_SPEED, 
                r_safe=0.3, # Collision radius
                workspace_radius=WORKSPACE_RADIUS,
                z_min=WORKSPACE_Z_MIN,
                z_max=WORKSPACE_Z_MAX
            )
            self.pipeline = CBFSafetyWrapper(
                base_controller, 
                cbf_filter, 
                lambda: [d.drone_state for d in self.controller.drones]
            )
        else:
            self.pipeline = base_controller

        self.stop_event = threading.Event()

    # ── Pose ──────────────────────────────────────────────────────────────
    def update_pose(self, position, rotation):
        self.drone_state.update_pose(position, rotation)

    def get_pose(self):
        return self.drone_state.get_pose()

    def get_orientation(self):
        with self.drone_state.lock:
            return self.drone_state.qx, self.drone_state.qy, self.drone_state.qz, self.drone_state.qw

    def get_pose_age(self):
        return self.drone_state.get_pose_age()

    # ── Nav ───────────────────────────────────────────────────────────────
    def set_target(self, x, y, z):
        with self.nav_lock:
            self.target_x, self.target_y, self.target_z = x, y, z

    def get_nav(self):
        with self.nav_lock:
            return (self.target_x, self.target_y, self.target_z, self.should_land)

    def is_arrived(self, radius=0.08):
        cx, cy, cz, _, _, _, _, valid = self.get_pose()
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
        self.drone_state.battery = float(voltage)
        if self.drone_state.battery < BATTERY_THRESHOLD and self.drone_state.status == "FLYING":
            self.controller.log(f"[{self.name}] LOW BATTERY ({self.drone_state.battery:.2f}V) - Auto Landing!")
            self.controller.fire_event("low_battery", self.number, {"voltage": self.drone_state.battery})
            with self.nav_lock:
                self.should_land = True

    # ── Extpos ────────────────────────────────────────────────────────────
    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, qx, qy, qz, qw, valid = self.get_pose()
            if valid and self.get_pose_age() < 0.2:
                scf.cf.extpos.send_extpose(x, y, z, qx, qy, qz, qw)
            time.sleep(max(0, dt - (time.time() - t0)))

    # ── Takeoff ───────────────────────────────────────────────────────────
    def takeoff(self, scf):
        cf = scf.cf
        self.drone_state.status = "TAKEOFF"
        cx, cy, cz, _, _, _, _, _ = self.get_pose()
        cyaw = self.drone_state.get_yaw()
        with self.nav_lock:
            self.target_yaw = cyaw
        self.controller.log(f"[{self.name}] Taking off ({self.ctrl_name}) to z={self.default_z}m, yaw={cyaw:.1f}deg...")
        ramp_dt = 0.02
        start = time.time()

        if self.ctrl_mode == 'position':
            self.pipeline.setpoint_x, self.pipeline.setpoint_y = cx, cy
            self.pipeline.setpoint_z = max(0.05, cz)
            for _ in range(10):
                cf.commander.send_position_setpoint(self.pipeline.setpoint_x, self.pipeline.setpoint_y, self.pipeline.setpoint_z, 0)
                time.sleep(ramp_dt)
            while self.pipeline.setpoint_z < self.default_z:
                if self.controller.kill_event.is_set():
                    cf.commander.send_stop_setpoint(); return False
                _, _, az, _, _, _, _, _ = self.get_pose()
                if az > HARD_KILL_Z:
                    self.controller.kill_event.set(); cf.commander.send_stop_setpoint(); return False
                if time.time() - start > 8.0: break
                self.pipeline.setpoint_z = min(self.pipeline.setpoint_z + MAX_SPEED * ramp_dt, self.default_z)
                cf.commander.send_position_setpoint(self.pipeline.setpoint_x, self.pipeline.setpoint_y, self.pipeline.setpoint_z, 0)
                time.sleep(ramp_dt)
        else:  # velocity mode
            for _ in range(10):
                cf.commander.send_velocity_world_setpoint(0, 0, 0, 0)
                time.sleep(0.01)
            while True:
                if self.controller.kill_event.is_set():
                    cf.commander.send_stop_setpoint(); return False
                _, _, cz, _, _, _, _, _ = self.get_pose()
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
        self.drone_state.status = "LANDING"
        self.controller.log(f"[{self.name}] Descending...")
        ramp_dt = 0.02

        _, _, cz, _, _, _, _, _ = self.get_pose()
        while cz > 0.10:
            if self.controller.kill_event.is_set(): break
            
            self.pipeline.setpoint_z = max(self.pipeline.setpoint_z - 0.2 * ramp_dt, 0.0)
            sz = self.pipeline.setpoint_z
            sx = self.pipeline.setpoint_x
            sy = self.pipeline.setpoint_y
            cf.commander.send_position_setpoint(sx, sy, sz, 0)
            time.sleep(0.05)
            _, _, cz, _, _, _, _, _ = self.get_pose()

        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        try:
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            cf.param.set_value('stabilizer.controller', '1')
            cf.param.set_value('stabilizer.estimator', '1')
        except Exception: pass
        self.drone_state.status = "LANDED"
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

        self.drone_state.status = "FLYING"
        self.controller.log(f"[{self.name}] Ready — {self.ctrl_name} active.")
        dt = 1.0 / LOOP_HZ
        was_arrived = False

        try:
            while not self.stop_event.is_set() and not self.controller.kill_event.is_set():
                loop_start = time.time()
                tx, ty, tz, should_land = self.get_nav()
                if should_land: break

                cx, cy, cz, qx, qy, qz, qw, got_data = self.get_pose()
                age = self.get_pose_age()

                if age > MOCAP_TIMEOUT:
                    self.controller.log(f"[{self.name}] MoCap timeout ({age:.1f}s) -> HOVERING")
                    cf.commander.send_hover_setpoint(0, 0, 0, cz)
                    self.controller.fire_event("mocap_lost", self.number, {'age': age})
                    time.sleep(dt); continue

                if (cx**2+cy**2) > HARD_KILL_R**2 or cz > HARD_KILL_Z:
                    self.controller.log(f"[{self.name}] HARD BOUNDARY — KILLING")
                    self.controller.kill_event.set(); break

                if age > 0.2:
                    cf.commander.send_hover_setpoint(0, 0, 0, cz)
                    time.sleep(dt); continue

                # Use Modular Pipeline
                setpoint = self.pipeline.compute(self.drone_state, (tx, ty, tz), self.target_yaw, dt)
                
                # Boundary clamp on setpoint
                r = math.sqrt(setpoint.x**2 + setpoint.y**2)
                if r > WORKSPACE_RADIUS:
                    setpoint.x *= WORKSPACE_RADIUS/r
                    setpoint.y *= WORKSPACE_RADIUS/r
                setpoint.z = max(WORKSPACE_Z_MIN, min(setpoint.z, WORKSPACE_Z_MAX))
                
                cf.commander.send_position_setpoint(setpoint.x, setpoint.y, setpoint.z, setpoint.yaw)

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
            self.marker_to_drone[rb_id].update_pose(position, rotation)

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
            x, y, z, _, _, _, _, _ = drone.get_pose()
            yaw = drone.drone_state.get_yaw()
            print(f"  {drone.name} ({drone.ctrl_name})  x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  yaw={yaw:+.1f}deg")
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
            x, y, _, _, _, _, _, _ = drone.get_pose()
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
            cx, cy, cz, qx, qy, qz, qw, valid = d.get_pose()
            cyaw = d.drone_state.get_yaw()
            with d.nav_lock: tx, ty, tz = d.target_x, d.target_y, d.target_z
            state.append({
                'number': d.number, 'name': d.name,
                'pos': (cx, cy, cz, cyaw) if valid else None,
                'target': (tx, ty, tz), 'battery': d.drone_state.battery,
                'state': d.drone_state.status, 'arrived': d.is_arrived(),
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
