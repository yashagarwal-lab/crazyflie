#!/usr/bin/env python3
"""
Interactive Planner Frontend

This file serves as a frontend application that utilizes the `SwarmController` backend.
It contains the user interface, grid definitions, and acts as a template for where
you will build your high-level path planning, trajectory generation, and obstacle
avoidance logic in the future.
"""

import time
import curses
import threading
from swarm_controller import SwarmController
from core.grid import GRID, print_grid

# ── Logging System for UI ─────────────────────────────────────────────────────
messages = []
msg_lock = threading.Lock()

def logging_callback(msg):
    """Callback function passed to the backend to receive its print logs."""
    with msg_lock:
        messages.append(msg)
        if len(messages) > 10:
            messages.pop(0)

def event_callback(event_name, drone_number, data):
    """Callback function to handle structured events from the hardware."""
    if event_name == "arrived":
        logging_callback(f"[EVENT] Drone{drone_number} reached target!")
    elif event_name == "low_battery":
        logging_callback(f"[ALARM] Drone{drone_number} low battery ({data.get('voltage', 0):.2f}V) - Auto Landing!")
    elif event_name == "mocap_lost":
        logging_callback(f"[ALARM] Drone{drone_number} lost MoCap tracking - Auto Landing!")


# ── Interactive Terminal ──────────────────────────────────────────────────────
def curses_ui(stdscr, controller):
    curses.curs_set(1)
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
        # Check if backend triggered an emergency stop or finished landing
        if controller.kill_event.is_set() or controller.all_landed():
            time.sleep(1.0)
            break

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # Header
        stdscr.addstr(0, 0, "=== Planner Algorithm Sandbox ===", curses.color_pair(1) | curses.A_BOLD)
        stdscr.addstr(1, 0, "Commands: <n> <grid> <z> | <n> <x> <y> <z> | experiment | land | Ctrl+C to quit")
        
        # Drone Status
        row = 3
        for d in controller.get_state():
            valid = d['pos'] is not None
            cx, cy, cz = d['pos'] if valid else (0, 0, 0)
            tx, ty, tz = d['target']
            state = d['state']
            batt = d['battery']
            arrived = d['arrived']
            
            dist = ((tx-cx)**2 + (ty-cy)**2 + (tz-cz)**2) ** 0.5 if valid else 0
            
            # Determine color based on health
            if not valid or state == "KILLED":
                status_color = curses.color_pair(2)
            elif batt < 3.4:  # Yellow warning before 3.2V auto-land
                status_color = curses.color_pair(3)
            else:
                status_color = curses.color_pair(1)

            pos_str = f"({cx:+.2f}, {cy:+.2f}, {cz:+.2f})" if valid else "( NO MOCAP )"
            tgt_str = f"({tx:+.2f}, {ty:+.2f}, {tz:+.2f})"
            state_str = f"[{state}]" if not arrived or state != "FLYING" else "[ARRIVED]"
            
            info = f"[{d['name']}] {batt:.2f}V {state_str:10} Pos: {pos_str:20} Tgt: {tgt_str:20} Dist: {dist:.2f}m"
            if row < h - 4:
                stdscr.addstr(row, 0, info, status_color)
            row += 1

        # Event Log
        row += 1
        if row < h - 4:
            stdscr.addstr(row, 0, "--- Event Log ---", curses.color_pair(4))
            row += 1
            with msg_lock:
                msgs = messages[-(h - row - 3):] if h - row - 3 > 0 else []
                for msg in msgs:
                    stdscr.addstr(row, 0, msg[:w-1])
                    row += 1

        # Input Prompt
        prompt_row = h - 1
        stdscr.addstr(prompt_row, 0, f"> {input_str}")
        stdscr.move(prompt_row, 2 + len(input_str))
        stdscr.refresh()

        # Keyboard Input
        try:
            key = stdscr.getch()
            if key != -1:
                if key in (curses.KEY_ENTER, 10, 13):
                    if input_str.strip():
                        process_command(input_str.strip(), controller)
                    input_str = ""
                elif key in (curses.KEY_BACKSPACE, 127, 8):
                    input_str = input_str[:-1]
                elif 32 <= key <= 126:
                    input_str += chr(key)
        except KeyboardInterrupt:
            break
        except Exception:
            pass


def process_command(raw, controller):
    """
    This is where your path planning algorithms will eventually go!
    Right now it just blindly feeds the command straight to the controller.
    """
    parts = raw.split()
    if not parts: return
    
    if parts[0].lower() == "land":
        if len(parts) == 1: controller.land_all()
        elif len(parts) == 2: controller.land(int(parts[1]))
        return

    # E.g. "experiment" -> runs a predefined sequence of waypoints
    if parts[0].lower() == "experiment":
        logging_callback("[EXP] Starting waypoint experiment...")
        threading.Thread(target=run_experiment, args=(controller,), daemon=True).start()
        return

    # E.g. "1 13 0.5" -> drone 1 to grid 13 at z=0.5
    if len(parts) == 3:
        try:
            dn = int(parts[0])
            g = int(parts[1])
            z = float(parts[2])
            if g in GRID:
                tx, ty = GRID[g]
                controller.goto_point(dn, tx, ty, z)
                logging_callback(f"[CMD] Drone{dn} -> Grid{g} ({tx},{ty},{z})")
            else:
                logging_callback("[ERR] Invalid grid number.")
        except ValueError:
            logging_callback("[ERR] Invalid format.")
        return

    # E.g. "1 0.5 -0.5 1.0" -> drone 1 to x=0.5, y=-0.5, z=1.0
    if len(parts) == 4:
        try:
            dn = int(parts[0])
            x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            controller.goto_point(dn, x, y, z)
            logging_callback(f"[CMD] Drone{dn} -> Point ({x},{y},{z})")
        except ValueError:
            logging_callback("[ERR] Invalid coordinate format.")
        return

    logging_callback(f"[ERR] Unknown command: {raw}")


def run_experiment(controller):
    """
    SKELETON EXPERIMENT SCRIPT
    This is where you can script a sequence of events.
    """
    drone_id = 1
    
    # Define a square trajectory
    waypoints = [
        (0.5, 0.5, 0.8),
        (0.5, -0.5, 0.8),
        (-0.5, -0.5, 0.8),
        (-0.5, 0.5, 0.8),
        (0.0, 0.0, 0.5) # Return home
    ]
    
    for (x, y, z) in waypoints:
        if controller.kill_event.is_set():
            break
            
        logging_callback(f"[EXP] Sending drone {drone_id} to ({x}, {y}, {z})")
        controller.goto_point(drone_id, x, y, z)
        
        # Wait until the drone physically arrives
        while not controller.is_arrived(drone_id, radius=0.1):
            if controller.kill_event.is_set():
                break
            time.sleep(0.1)
            
        logging_callback(f"[EXP] Waypoint reached. Waiting 2 seconds...")
        time.sleep(2.0)
        
    logging_callback("[EXP] Experiment complete!")


if __name__ == '__main__':
    # You can now specify the 'uri' explicitly here if it doesn't follow the default pattern
    drone_configs = [
        {'number': 1, 'marker_id': 351, 'default_z': 0.5},
        # {'number': 2, 'marker_id': 352, 'default_z': 0.5, 'uri': 'radio://0/80/2M/E7E7E7E702'},
    ]

    # Initialize the backend
    controller = SwarmController(
        drone_configs, 
        logging_callback=logging_callback,
        event_callback=event_callback
    )
    
    # We must start without printing anything to mess up curses later, 
    # but the initial sanity check needs standard print capabilities.
    if controller.start(interactive=True):
        try:
            # Start the UI frontend
            curses.wrapper(curses_ui, controller)
        except Exception as e:
            print(f"UI Error: {e}")
            
        # Clean up backend
        controller.wait_for_landing()
