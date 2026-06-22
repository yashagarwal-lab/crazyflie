#!/usr/bin/env python3
"""
Interactive Planner Frontend

Commands:
  <n> <grid> <z>              — drone N to grid position at height z
  <n> <x> <y> <z>             — drone N to coordinate
  formation <shape> <param>   — line/triangle/circle with radius/spacing
  formation move <dx> <dy> <dz> — translate formation
  formation rotate <deg>      — rotate formation
  trajectory <n> <shape> <r> <z> — smooth trajectory (circle/square/figure8)
  experiment                  — run scripted waypoint sequence
  land / land <n>             — land all or specific drone
"""

import time
import curses
import threading
from swarm_controller import SwarmController
from core.grid import GRID, print_grid
from core.collision import CollisionAvoider
from core.formation import FormationManager
from core.trajectory import circle_trajectory, square_trajectory, figure8_trajectory

# ── Logging ───────────────────────────────────────────────────────────────────
messages = []
msg_lock = threading.Lock()

def logging_callback(msg):
    with msg_lock:
        messages.append(msg)
        if len(messages) > 10: messages.pop(0)

def event_callback(event_name, drone_number, data):
    if event_name == "arrived":
        logging_callback(f"[EVENT] Drone{drone_number} reached target!")
    elif event_name == "low_battery":
        logging_callback(f"[ALARM] Drone{drone_number} low battery ({data.get('voltage',0):.2f}V)")
    elif event_name == "mocap_lost":
        logging_callback(f"[ALARM] Drone{drone_number} lost MoCap!")


# ── Curses UI ─────────────────────────────────────────────────────────────────
def curses_ui(stdscr, controller, avoider, formation_mgr):
    curses.curs_set(1); stdscr.nodelay(True); stdscr.timeout(100)
    curses.start_color(); curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)
    input_str = ""

    while True:
        if controller.kill_event.is_set() or controller.all_landed():
            time.sleep(1.0); break

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        stdscr.addstr(0, 0, "=== Swarm Planner ===", curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(1, 0, "Cmds: <n> <grid> <z> | <n> <x> <y> <z> | formation | trajectory | land")

        row = 3
        for d in controller.get_state():
            valid = d['pos'] is not None
            cx, cy, cz, cyaw = d['pos'] if valid else (0, 0, 0, 0)
            tx, ty, tz = d['target']
            batt, state, arrived = d['battery'], d['state'], d['arrived']
            ctrl = d.get('controller', '?')
            dist = ((tx-cx)**2+(ty-cy)**2+(tz-cz)**2)**0.5 if valid else 0

            color = curses.color_pair(2) if not valid or state == "KILLED" else \
                    curses.color_pair(3) if batt < 3.4 else curses.color_pair(1)
            st = "[ARRIVED]" if arrived and state == "FLYING" else f"[{state}]"
            pos = f"({cx:+.2f},{cy:+.2f},{cz:+.2f}) y={cyaw:+.0f}°" if valid else "(NO MOCAP)"
            info = f"[{d['name']}|{ctrl}] {batt:.2f}V {st:10} {pos} -> ({tx:+.2f},{ty:+.2f},{tz:+.2f}) d={dist:.2f}"
            if row < h - 4: stdscr.addstr(row, 0, info[:w-1], color)
            row += 1

        row += 1
        if row < h - 4:
            stdscr.addstr(row, 0, "--- Log ---", curses.color_pair(4)); row += 1
            with msg_lock:
                for msg in messages[-(h-row-3):]:
                    if row < h - 2: stdscr.addstr(row, 0, msg[:w-1]); row += 1

        stdscr.addstr(h-1, 0, f"> {input_str}")
        stdscr.move(h-1, 2+len(input_str)); stdscr.refresh()

        try:
            key = stdscr.getch()
            if key != -1:
                if key in (curses.KEY_ENTER, 10, 13):
                    if input_str.strip():
                        process_command(input_str.strip(), controller, avoider, formation_mgr)
                    input_str = ""
                elif key in (curses.KEY_BACKSPACE, 127, 8): input_str = input_str[:-1]
                elif 32 <= key <= 126: input_str += chr(key)
        except KeyboardInterrupt: break
        except Exception: pass


# ── Command Processor ─────────────────────────────────────────────────────────
def process_command(raw, controller, avoider, formation_mgr):
    parts = raw.split()
    if not parts: return

    # land
    if parts[0].lower() == "land":
        if len(parts) == 1: controller.land_all()
        elif len(parts) == 2: controller.land(int(parts[1]))
        return

    # experiment
    if parts[0].lower() == "experiment":
        logging_callback("[EXP] Starting experiment...")
        threading.Thread(target=run_experiment, args=(controller, avoider), daemon=True).start()
        return

    # formation <shape> <param>
    if parts[0].lower() == "formation":
        if len(parts) < 2:
            logging_callback("[ERR] Usage: formation <line|triangle|circle> <param>"); return
        sub = parts[1].lower()
        if sub == "move" and len(parts) == 5:
            dx, dy, dz = float(parts[2]), float(parts[3]), float(parts[4])
            formation_mgr.translate(dx, dy, dz)
            logging_callback(f"[FMT] Translated by ({dx},{dy},{dz})")
        elif sub == "rotate" and len(parts) == 3:
            deg = float(parts[2])
            formation_mgr.rotate(deg)
            logging_callback(f"[FMT] Rotated {deg}°")
        elif sub in ("line", "triangle", "circle") and len(parts) >= 3:
            param = float(parts[2])
            z = float(parts[3]) if len(parts) > 3 else 0.5
            formation_mgr.apply(sub, **({'spacing': param} if sub == 'line' else {'radius': param}), z=z)
            logging_callback(f"[FMT] Applied {sub} (param={param}, z={z})")
        else:
            logging_callback("[ERR] Unknown formation command")
        return

    # trajectory <n> <shape> <param> <z>
    if parts[0].lower() == "trajectory":
        if len(parts) < 4:
            logging_callback("[ERR] Usage: trajectory <n> <circle|square|figure8> <radius> [z]"); return
        dn = int(parts[1])
        shape = parts[2].lower()
        param = float(parts[3])
        z = float(parts[4]) if len(parts) > 4 else 0.8
        threading.Thread(target=run_trajectory, args=(controller, dn, shape, param, z), daemon=True).start()
        logging_callback(f"[TRJ] Drone{dn} {shape} r={param} z={z}")
        return

    # <n> <grid> <z>
    if len(parts) == 3:
        try:
            dn, g, z = int(parts[0]), int(parts[1]), float(parts[2])
            if g in GRID:
                tx, ty = GRID[g]
                avoider.safe_goto(dn, tx, ty, z)
                logging_callback(f"[CMD] Drone{dn} -> Grid{g} ({tx},{ty},{z})")
            else: logging_callback("[ERR] Invalid grid")
        except ValueError: logging_callback("[ERR] Invalid format")
        return

    # <n> <x> <y> <z>
    if len(parts) == 4:
        try:
            dn = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            avoider.safe_goto(dn, x, y, z)
            logging_callback(f"[CMD] Drone{dn} -> ({x},{y},{z})")
        except ValueError: logging_callback("[ERR] Invalid format")
        return

    logging_callback(f"[ERR] Unknown: {raw}")


# ── Trajectory Runner ─────────────────────────────────────────────────────────
def run_trajectory(controller, drone_id, shape, radius, z):
    traj_map = {'circle': circle_trajectory, 'square': square_trajectory, 'figure8': figure8_trajectory}
    if shape not in traj_map:
        logging_callback(f"[ERR] Unknown shape: {shape}"); return

    traj = traj_map[shape](radius=radius, z=z)
    logging_callback(f"[TRJ] Running {shape} ({traj.total_time:.1f}s)")
    t0 = time.time()
    while time.time() - t0 < traj.total_time:
        if controller.kill_event.is_set(): break
        x, y, z = traj.evaluate(time.time() - t0)
        controller.goto_point(drone_id, x, y, z)
        time.sleep(0.02)
    logging_callback("[TRJ] Complete!")


# ── Experiment Runner ─────────────────────────────────────────────────────────
def run_experiment(controller, avoider):
    drone_id = 1
    waypoints = [(0.5,0.5,0.8), (0.5,-0.5,0.8), (-0.5,-0.5,0.8), (-0.5,0.5,0.8), (0.0,0.0,0.5)]
    for (x, y, z) in waypoints:
        if controller.kill_event.is_set(): break
        logging_callback(f"[EXP] -> ({x},{y},{z})")
        avoider.safe_goto(drone_id, x, y, z)
        while not controller.is_arrived(drone_id, radius=0.1):
            if controller.kill_event.is_set(): break
            time.sleep(0.1)
        logging_callback("[EXP] Arrived. Waiting 2s...")
        time.sleep(2.0)
    logging_callback("[EXP] Complete!")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    drone_configs = [
        {'number': 1, 'marker_id': 351, 'default_z': 0.5, 'controller': 'mellinger', 'use_cbf': True},
        # {'number': 2, 'marker_id': 352, 'default_z': 0.5, 'controller': 'mellinger', 'use_cbf': True},
    ]

    controller = SwarmController(drone_configs, logging_callback=logging_callback, event_callback=event_callback)
    avoider = CollisionAvoider(controller)
    formation_mgr = FormationManager(controller, avoider=avoider)

    if controller.start(interactive=True):
        try:
            curses.wrapper(curses_ui, controller, avoider, formation_mgr)
        except Exception as e:
            print(f"UI Error: {e}")
        controller.wait_for_landing()
