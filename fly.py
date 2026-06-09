#!/usr/bin/env python3
"""
Simple Crazyflie keyboard controller using curses + SyncCrazyflie.

API reference (from cflib/crazyflie/commander.py):
  send_setpoint(roll, pitch, yawrate, thrust)
    - roll, pitch  : degrees
    - yawrate      : degrees/s
    - thrust       : uint16, 10001 (min power) to 60000 (full power)
  Watchdog: firmware cuts motors if no setpoint received for 500 ms.
  Safety lock: must send thrust=0 once before non-zero thrust is accepted.
"""
import curses
import time
import threading
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/80/2M/E7E7E7E7E7'

# ── Parameters (from cflib docs) ──
THRUST_FLOOR = 10001    # minimum thrust that actually spins motors
THRUST_MAX = 60000      # full power (per commander.py line 86)
THRUST_STEP = 2000      # increment per key press
PITCH_ANGLE = 10.0      # degrees
ROLL_ANGLE = 10.0       # degrees
YAW_RATE = 50           # degrees/s
SEND_RATE_HZ = 20       # setpoint send rate (must be >2 Hz per watchdog)

# ── Shared state dict (mutable — visible to all threads) ──
state = {
    'thrust': 0,
    'roll': 0.0,
    'pitch': 0.0,
    'yawrate': 0,
    'flying': False,
    'status': 'Initializing...',
}


def sender_loop(cf):
    """Background thread: sends setpoints at 20 Hz."""
    try:
        while state['flying']:
            cf.commander.send_setpoint(
                state['roll'],
                state['pitch'],
                state['yawrate'],
                int(state['thrust']),
            )
            time.sleep(1.0 / SEND_RATE_HZ)
        cf.commander.send_stop_setpoint()
    except Exception as exc:
        state['status'] = f"Sender crash: {exc}"


def draw(stdscr):
    """Render the flight HUD."""
    stdscr.erase()
    h, w = stdscr.getmaxyx()

    def safe(row, col, text, *args):
        if 0 <= row < h and col < w:
            try:
                stdscr.addnstr(row, col, text, w - col, *args)
            except curses.error:
                pass

    G = curses.color_pair(1)
    R = curses.color_pair(2)
    Y = curses.color_pair(3)
    C = curses.color_pair(4)
    B = curses.A_BOLD

    safe(0, 0, "=== CRAZYFLIE FLIGHT CONTROL ===", G | B)
    safe(2, 1, f"Status: {state['status']}", Y)

    thrust = state['thrust']
    pct = thrust / THRUST_MAX if THRUST_MAX else 0
    bar_w = 25
    filled = int(pct * bar_w)
    bar = "#" * filled + "-" * (bar_w - filled)
    color = R | B if pct > 0.7 else (Y if pct > 0.4 else G)
    safe(4, 1, f"Thrust : {int(thrust):>5}  [{bar}]  {pct*100:4.1f}%", color)
    if 0 < thrust < THRUST_FLOOR:
        safe(5, 1, f"  (below {THRUST_FLOOR} — motors won't spin)", Y)

    safe(7, 1, f"Pitch  : {state['pitch']:>+7.1f} deg", C)
    safe(8, 1, f"Roll   : {state['roll']:>+7.1f} deg", C)
    safe(9, 1, f"Yaw    : {state['yawrate']:>+4d}   deg/s", C)

    safe(11, 1, "--- Controls ---", G)
    safe(12, 1, " UP / DOWN   Thrust  +/- 2000")
    safe(13, 1, " W / S       Pitch   fwd / back")
    safe(14, 1, " A / D       Roll    left / right")
    safe(15, 1, " Q / E       Yaw     left / right")
    safe(16, 1, " SPACE       Kill motors (thrust=0)")
    safe(17, 1, " ESC         Land & quit")

    stdscr.refresh()


def main(stdscr):
    # ── Curses setup ──
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(1000 // SEND_RATE_HZ)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    # ── Cflib init ──
    logging.basicConfig(level=logging.ERROR)
    cflib.crtp.init_drivers()

    state['status'] = f"Connecting to {URI} ..."
    draw(stdscr)

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            cf = scf.cf

            state['status'] = "Connected! Arming..."
            draw(stdscr)

            # --- ARMING SEQUENCE ---
            try:
                # 1. system.forceArm (works on some older firmwares)
                cf.param.set_value('system.forceArm', 1)
            except Exception:
                pass
                
            try:
                # 2. Legacy CRTP PLATFORM arming (works on protocol v5)
                from cflib.crtp.crtpstack import CRTPPacket, CRTPPort
                import struct
                pk = CRTPPacket()
                pk.set_header(CRTPPort.PLATFORM, 0)
                pk.data = struct.pack('<BB', 1, 1)
                cf.send_packet(pk)
            except Exception:
                pass
            
            time.sleep(0.5)

            # 3. Safety unlock: firmware requires thrust=0 before accepting thrust.
            # We send it 10 times to ensure no packets are dropped over the radio.
            for _ in range(10):
                cf.commander.send_setpoint(0, 0, 0, 0)
                time.sleep(0.01)

            state['status'] = "Ready! Press UP to increase thrust."
            state['flying'] = True
            draw(stdscr)

            # Start background sender
            threading.Thread(target=sender_loop, args=(cf,), daemon=True).start()

            # ── Main input loop ──
            try:
                while state['flying']:
                    # Auto-center attitude each frame
                    state['roll'] = 0.0
                    state['pitch'] = 0.0
                    state['yawrate'] = 0

                    key = stdscr.getch()

                    if key == curses.KEY_UP:
                        if state['thrust'] < THRUST_FLOOR:
                            state['thrust'] = THRUST_FLOOR
                        else:
                            state['thrust'] = min(state['thrust'] + THRUST_STEP, THRUST_MAX)
                    elif key == curses.KEY_DOWN:
                        state['thrust'] = max(state['thrust'] - THRUST_STEP, 0)
                    elif key in (ord('w'), ord('W')):
                        state['pitch'] = -PITCH_ANGLE
                    elif key in (ord('s'), ord('S')):
                        state['pitch'] = PITCH_ANGLE
                    elif key in (ord('a'), ord('A')):
                        state['roll'] = -ROLL_ANGLE
                    elif key in (ord('d'), ord('D')):
                        state['roll'] = ROLL_ANGLE
                    elif key in (ord('q'), ord('Q')):
                        state['yawrate'] = -YAW_RATE
                    elif key in (ord('e'), ord('E')):
                        state['yawrate'] = YAW_RATE
                    elif key == ord(' '):
                        state['thrust'] = 0
                        state['status'] = "KILL SWITCH — motors stopped!"
                    elif key == 27:  # ESC
                        state['status'] = "Landing..."
                        draw(stdscr)
                        while state['thrust'] > 0:
                            state['thrust'] = max(state['thrust'] - THRUST_STEP, 0)
                            draw(stdscr)
                            time.sleep(0.05)
                        break

                    draw(stdscr)

            except KeyboardInterrupt:
                pass
            finally:
                state['flying'] = False
                state['thrust'] = 0
                time.sleep(0.8)
                cf.param.set_value('system.forceArm', 0)

    except Exception as exc:
        state['status'] = f"Error: {exc}"
        draw(stdscr)
        time.sleep(3)


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
