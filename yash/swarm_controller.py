#!/usr/bin/env python3
"""
Backend Swarm Controller API.

This module provides the `SwarmController` class, which acts strictly as a hardware 
and safety backend. It handles:
  - Connection to OptiTrack (NatNetClient).
  - Individual drone initialization and communication via `cflib`.
  - Control Barrier Function (CBF) safety filtering for absolute workspace bounds.

Architecture Note:
  With the Mellinger controller, we send POSITION setpoints (not velocity).
  Mellinger has its own internal high-gain position tracker, so we do NOT
  need an offboard PID loop. Instead, the CBF filter clamps the TARGET
  POSITION to stay inside the workspace boundary.
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

from core.cbf import CBFSafetyFilter

logging.basicConfig(level=logging.ERROR)

# ── Global flight parameters ──────────────────────────────────────────────────
LOOP_HZ          = 50
EXTPOS_HZ        = 100
MAX_ALTITUDE     = 1.8
BATTERY_THRESHOLD = 3.2    # Auto-land if voltage drops below this
MOCAP_TIMEOUT    = 1.5     # Seconds to hover without MoCap before landing

# ── Cylindrical workspace (CBF enforced) ──────────────────────────────────────
WORKSPACE_RADIUS = 1.25   
WORKSPACE_Z_MAX  = 1.8    
WORKSPACE_Z_MIN  = 0.05   
CBF_D_MIN        = 0.3    

# ── Hard emergency kill radii ─────────────────────────────────────────────────
HARD_KILL_R      = 1.55   
HARD_KILL_Z      = 2.0    


def clamp_to_workspace(x, y, z, other_positions=None, radius=WORKSPACE_RADIUS,
                        z_min=WORKSPACE_Z_MIN, z_max=WORKSPACE_Z_MAX, d_min=CBF_D_MIN):
    """
    Clamp a target position to stay inside the cylindrical workspace
    and maintain minimum separation from other drones.
    
    This replaces the velocity-based CBF QP. Since Mellinger tracks positions
    directly, we simply project the target back inside the safe region.
    """
    # Clamp Z
    z = max(z_min, min(z, z_max))
    
    # Clamp XY to cylinder
    r = math.sqrt(x**2 + y**2)
    if r > radius:
        scale = radius / r
        x *= scale
        y *= scale

    # Push away from other drones (simple repulsion)
    if other_positions:
        for ox, oy, oz in other_positions:
            dx, dy, dz = x - ox, y - oy, z - oz
            dist = math.sqrt(dx**2 + dy**2 + dz**2)
            if dist < d_min and dist > 0.001:
                # Push target outward to maintain d_min separation
                push = (d_min - dist) / dist
                x += dx * push
                y += dy * push
                z += dz * push
                # Re-clamp after push
                z = max(z_min, min(z, z_max))
                r = math.sqrt(x**2 + y**2)
                if r > radius:
                    scale = radius / r
                    x *= scale
                    y *= scale

    return x, y, z


class Drone:
    def __init__(self, config, controller):
        self.number    = config['number']
        self.name      = f"Drone{self.number}"
        self.uri       = config.get('uri', f"radio://0/80/2M/E7E7E7E70{self.number}")
        self.marker_id = config['marker_id']
        self.default_z = config['default_z']
        self.cache     = f"./cache{self.number}"
        self.controller = controller

        # State Variables
        self.pose_lock   = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.pose_valid  = False
        self.last_update = 0.0
        self.battery     = 4.2
        self.state       = "INIT"

        self.nav_lock       = threading.Lock()
        self.target_x       = 0.0
        self.target_y       = 0.0
        self.target_z       = self.default_z
        self.should_land    = False

        self.stop_event = threading.Event()

    def update_pose(self, position):
        with self.pose_lock:
            self.x = position[2]
            self.y = position[0]
            self.z = position[1]
            self.pose_valid  = True
            self.last_update = time.time()

    def get_pose(self):
        with self.pose_lock:
            return self.x, self.y, self.z, self.pose_valid

    def get_pose_age(self):
        with self.pose_lock:
            return time.time() - self.last_update if self.last_update > 0 else float('inf')

    def set_target(self, x, y, z):
        with self.nav_lock:
            self.target_x = x
            self.target_y = y
            self.target_z = z

    def get_nav(self):
        with self.nav_lock:
            return (self.target_x, self.target_y, self.target_z, self.should_land)

    def is_arrived(self, radius=0.08):
        with self.pose_lock:
            cx, cy, cz = self.x, self.y, self.z
            valid = self.pose_valid
        with self.nav_lock:
            tx, ty, tz = self.target_x, self.target_y, self.target_z
        if not valid: return False
        return ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2)**0.5 < radius

    def _battery_callback(self, timestamp, data, logconf):
        voltage = data.get('pm.vbat', 4.2)
        self.battery = float(voltage)
        if self.battery < BATTERY_THRESHOLD and self.state == "FLYING":
            self.controller.log(f"[{self.name}] LOW BATTERY ({self.battery:.2f}V) - Auto Landing!")
            self.controller.fire_event("low_battery", self.number, {"voltage": self.battery})
            with self.nav_lock:
                self.should_land = True

    def run_extpos(self, scf, stop_ep):
        dt = 1.0 / EXTPOS_HZ
        while not stop_ep.is_set():
            t0 = time.time()
            x, y, z, valid = self.get_pose()
            age = self.get_pose_age()
            
            # Only send extpos if we have fresh MoCap data. 
            # If stale, we stop sending so the EKF relies purely on IMU.
            if valid and age < 0.2:
                scf.cf.extpos.send_extpos(x, y, z)
            
            elapsed = time.time() - t0
            time.sleep(max(0, dt - elapsed))

    def takeoff(self, scf):
        """Ramp position setpoint from ground to default_z."""
        cf = scf.cf
        self.state = "TAKEOFF"
        self.controller.log(f"[{self.name}] Taking off to z={self.default_z}m ...")

        cx, cy, _, _ = self.get_pose()
        
        # Warm up the commander with current position at ground level
        for _ in range(10):
            cf.commander.send_position_setpoint(cx, cy, 0.0, 0)
            time.sleep(0.02)

        # Ramp Z upward smoothly
        ramp_rate = 0.3  # m/s
        ramp_dt = 0.02
        current_z = 0.05
        start = time.time()
        
        while current_z < self.default_z:
            if self.controller.kill_event.is_set():
                cf.commander.send_stop_setpoint()
                return False
            
            _, _, actual_z, _ = self.get_pose()
            if actual_z > MAX_ALTITUDE:
                self.controller.log(f"[{self.name}] ALTITUDE LIMIT HIT DURING TAKEOFF — killing!")
                self.controller.kill_event.set()
                cf.commander.send_stop_setpoint()
                return False
            
            if time.time() - start > 8.0:
                self.controller.log(f"[{self.name}] WARNING: Takeoff timeout — continuing anyway.")
                break
            
            current_z += ramp_rate * ramp_dt
            current_z = min(current_z, self.default_z)
            cf.commander.send_position_setpoint(cx, cy, current_z, 0)
            time.sleep(ramp_dt)
        
        self.controller.log(f"[{self.name}] Takeoff complete.")
        return True

    def land(self, scf):
        """Ramp position setpoint down to ground."""
        cf = scf.cf
        self.state = "LANDING"
        self.controller.log(f"[{self.name}] Descending...")
        
        cx, cy, cz, _ = self.get_pose()
        ramp_rate = 0.2  # m/s descent
        ramp_dt = 0.02
        current_z = cz
        
        while current_z > 0.05:
            if self.controller.kill_event.is_set():
                break
            cx, cy, _, _ = self.get_pose()  # Track XY during descent
            current_z -= ramp_rate * ramp_dt
            current_z = max(current_z, 0.0)
            cf.commander.send_position_setpoint(cx, cy, current_z, 0)
            time.sleep(ramp_dt)
        
        cf.commander.send_stop_setpoint()
        time.sleep(0.3)
        try:
            cf.param.set_value('kalman.resetEstimation', '1')
            time.sleep(0.1)
            cf.param.set_value('kalman.resetEstimation', '0')
            cf.param.set_value('stabilizer.controller', '1')
            cf.param.set_value('stabilizer.estimator', '1')
        except Exception:
            pass
        self.state = "LANDED"
        self.controller.log(f"[{self.name}] Landed.")

    def flight_loop(self, scf):
        cf = scf.cf

        # 1. Setup Battery Logging
        log_conf = LogConfig(name='Battery', period_in_ms=1000)
        log_conf.add_variable('pm.vbat', 'float')
        scf.cf.log.add_config(log_conf)
        log_conf.data_received_cb.add_callback(self._battery_callback)
        log_conf.start()

        # 2. Setup Estimator + Mellinger Controller
        cf.param.set_value('stabilizer.estimator', '2')   # EKF
        cf.param.set_value('stabilizer.controller', '2')  # Mellinger
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

        # 3. Takeoff
        if not self.takeoff(scf) or self.controller.kill_event.is_set():
            stop_ep.set()
            return

        self.state = "FLYING"
        self.controller.log(f"[{self.name}] Ready — Mellinger position control active.")
        
        dt = 1.0 / LOOP_HZ
        was_arrived = False

        # 4. Main Flight Loop — send POSITION setpoints, not velocity
        try:
            while not self.stop_event.is_set() and not self.controller.kill_event.is_set():
                loop_start = time.time()

                tx, ty, tz, should_land = self.get_nav()
                if should_land:
                    break

                cx, cy, cz, got_data = self.get_pose()
                age = self.get_pose_age()

                # Handle MoCap Occlusion — hold last known position
                if age > MOCAP_TIMEOUT:
                    self.controller.log(f"[{self.name}] !! MOCAP LOST > {MOCAP_TIMEOUT}s — AUTO LANDING !!")
                    self.controller.fire_event("mocap_lost", self.number, {})
                    with self.nav_lock:
                        self.should_land = True
                    break

                # Hard Boundary Failsafe
                if (cx**2 + cy**2) > HARD_KILL_R**2 or cz > HARD_KILL_Z:
                    self.controller.log(f"[{self.name}] !! HARD BOUNDARY BREACH pos=({cx:+.3f},{cy:+.3f},{cz:+.3f}) — KILLING !!")
                    self.controller.kill_event.set()
                    break

                # Short Occlusion: Keep sending last target, EKF holds via IMU
                if age > 0.2:
                    cf.commander.send_position_setpoint(tx, ty, tz, 0)
                    time.sleep(dt)
                    continue

                # Clamp target to workspace boundary + inter-drone separation
                other_poses = []
                for d in self.controller.drones:
                    if d is not self:
                        ox, oy, oz, _ = d.get_pose()
                        other_poses.append((ox, oy, oz))

                safe_x, safe_y, safe_z = clamp_to_workspace(
                    tx, ty, tz, other_positions=other_poses
                )

                # Send position setpoint — Mellinger handles the tracking
                cf.commander.send_position_setpoint(safe_x, safe_y, safe_z, 0)

                # Fire event once upon reaching target
                currently_arrived = self.is_arrived()
                if currently_arrived and not was_arrived:
                    self.controller.fire_event("arrived", self.number, {"pos": (cx, cy, cz)})
                was_arrived = currently_arrived

                elapsed = time.time() - loop_start
                time.sleep(max(0, dt - elapsed))

        except Exception as e:
            self.controller.log(f"[{self.name}] Flight loop error: {e}")

        # 5. Landing / Kill
        if self.controller.kill_event.is_set():
            cf.commander.send_stop_setpoint()
            self.state = "KILLED"
            self.controller.log(f"[{self.name}] Motors killed instantly.")
        else:
            self.land(scf)

        log_conf.stop()
        stop_ep.set()


class SwarmController:
    def __init__(self, drone_configs, logging_callback=None, event_callback=None):
        """
        drone_configs: List of dicts specifying drone configs.
        logging_callback: function(str) to pipe hardware logs back to your UI planner.
        event_callback: function(event_name, drone_number, data)
        """
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
        if self.logging_callback:
            self.logging_callback(msg)
        else:
            print(msg)

    def fire_event(self, event_name, drone_number, data):
        if self.event_callback:
            self.event_callback(event_name, drone_number, data)

    def receive_rigid_body_frame(self, rb_id, position, rotation):
        if rb_id in self.marker_to_drone:
            self.marker_to_drone[rb_id].update_pose(position)

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
            x, y, z, _ = drone.get_pose()
            print(f"  {drone.name} (marker {drone.marker_id})  "
                  f"x={x:+.3f}  y={y:+.3f}  z={z:+.3f}  (height={z:.3f}m)")
        print()
        print("  Z values should all be close to 0.0 (floor level).")
        print("  Confirm each position matches where that drone is physically sitting.\n")
        confirm = input("  Do ALL positions match the physical drones? (yes/no): ").strip().lower()
        if confirm != 'yes':
            print("\n  [ABORT] Sanity check failed. Check marker IDs.\n")
            return False
        print("  [OK] Sanity check passed — proceeding to flight.\n")
        return True

    def start(self, interactive=True):
        n = len(self.drones)
        self.log(f"\n[INIT]   {n} drone(s) initialized.")

        self.log("[MoCap]  Connecting to Motive NatNet...")
        self.mocap_client = NatNetClient()
        self.mocap_client.rigidBodyListener = self.receive_rigid_body_frame
        self.mocap_client.run()
        self.log("[MoCap]  Waiting for all rigid bodies...")

        timeout = time.time() + 10.0
        while True:
            missing = [d.name for d in self.drones if not d.pose_valid]
            if not missing:
                break
            if time.time() > timeout:
                self.log(f"[MoCap]  ERROR: Cannot see: {missing}")
                self.mocap_client.stop()
                return False
            time.sleep(0.05)
        self.log("[MoCap]  All rigid bodies found!")

        if interactive and not self.sanity_check():
            self.mocap_client.stop()
            return False

        for drone in self.drones:
            x, y, _, _ = drone.get_pose()
            drone.set_target(x, y, drone.default_z)
            self.log(f"[NAV]    {drone.name} targeting home: x={x:+.3f} y={y:+.3f} z={drone.default_z}")

        cflib.crtp.init_drivers()
        self.log("[CF]     Connecting to all drones (Max 3 retries)...")
        self.start_kill_listener()

        try:
            # Connect with retry logic
            for drone in self.drones:
                connected = False
                for attempt in range(3):
                    try:
                        scf = SyncCrazyflie(drone.uri, cf=Crazyflie(rw_cache=drone.cache))
                        scf.open_link()
                        self.scf_list.append(scf)
                        connected = True
                        break
                    except Exception as e:
                        self.log(f"[CF]     {drone.name} connection failed (Attempt {attempt+1}): {e}")
                        time.sleep(1.0)
                if not connected:
                    raise Exception(f"Failed to connect to {drone.name} after 3 attempts.")

            self.log(f"[CF]     All {n} drones connected!\n")

            for drone, scf in zip(self.drones, self.scf_list):
                t = threading.Thread(target=drone.flight_loop, args=(scf,), daemon=True)
                self.flight_threads.append(t)
                t.start()
                time.sleep(3.0)

            return True

        except Exception as e:
            self.log(f"[CF]     Fatal Connection error: {e}")
            self.stop_all()
            return False

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

    def goto_point(self, drone_number, x, y, z):
        """Pure coordinate targeting."""
        if drone_number not in self.drone_by_num: return False
        if z > MAX_ALTITUDE or z < 0.1: return False
        
        self.drone_by_num[drone_number].set_target(x, y, z)
        return True

    def is_arrived(self, drone_number, radius=0.08):
        if drone_number not in self.drone_by_num: return False
        return self.drone_by_num[drone_number].is_arrived(radius)

    def get_state(self):
        """Returns a list of dicts with current state for external planners/UIs."""
        state = []
        for drone in self.drones:
            cx, cy, cz, valid = drone.get_pose()
            with drone.nav_lock:
                tx, ty, tz = drone.target_x, drone.target_y, drone.target_z
            state.append({
                'number': drone.number,
                'name': drone.name,
                'pos': (cx, cy, cz) if valid else None,
                'target': (tx, ty, tz),
                'battery': drone.battery,
                'state': drone.state,
                'arrived': drone.is_arrived()
            })
        return state

    def all_landed(self):
        return len(self.flight_threads) > 0 and all(not t.is_alive() for t in self.flight_threads)

    def wait_for_landing(self):
        self.log("\n[CTRL] Waiting for drones to land...")
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
