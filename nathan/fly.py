#!/usr/bin/env python3
"""
Crazyflie keyboard flight controller using MotionCommander (Flow v2).

MotionCommander handles arming, estimator setup, and stabilization.
All movement is velocity-based with automatic altitude hold.

Controls:
  T            Take off (0.3 m)
  L            Land gracefully
  W / S        Forward / Backward
  A / D        Left / Right
  Q / E        Yaw left / right
  UP / DOWN    Ascend / Descend
  SPACE        Emergency stop (kill motors)
  ESC          Land & quit
"""
import curses
import time
import logging
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
logging.basicConfig(level=logging.ERROR)

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M/E7E7E7E701'

# ── Flight parameters ──
DEFAULT_HEIGHT = 0.3    # metres
VELOCITY = 0.3          # m/s lateral
VELOCITY_Z = 0.2        # m/s vertical
YAW_RATE = 60           # deg/s


def draw(stdscr, status, airborne, vx, vy, vz, yaw):
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

    safe(0, 0, "=== CRAZYFLIE FLIGHT CONTROL (Flow v2) ===", G | B)
    safe(2, 1, f"Status: {status}", Y)

    if airborne:
        safe(4, 1, "● AIRBORNE", G | B)
    else:
        safe(4, 1, "○ GROUNDED", Y)

    safe(6, 1, f"Vx (fwd/back) : {vx:+.2f} m/s", C)
    safe(7, 1, f"Vy (left/right): {vy:+.2f} m/s", C)
    safe(8, 1, f"Vz (up/down)  : {vz:+.2f} m/s", C)
    safe(9, 1, f"Yaw rate      : {yaw:+.0f}   deg/s", C)

    safe(11, 1, "--- Controls ---", G)
    safe(12, 1, " T           Take off")
    safe(13, 1, " L           Land")
    safe(14, 1, " W / S       Forward / Backward")
    safe(15, 1, " A / D       Left / Right")
    safe(16, 1, " Q / E       Yaw left / right")
    safe(17, 1, " UP / DOWN   Ascend / Descend")
    safe(18, 1, " SPACE       EMERGENCY STOP")
    safe(19, 1, " ESC         Land & quit")

    stdscr.refresh()


def main(stdscr):
    # ── Curses setup ──
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)  # 20 Hz refresh
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)
    curses.init_pair(3, curses.COLOR_YELLOW, -1)
    curses.init_pair(4, curses.COLOR_CYAN, -1)

    status = "Initializing..."
    airborne = False
    vx = vy = vz = yaw = 0.0
    draw(stdscr, status, airborne, vx, vy, vz, yaw)

    # ── Connect ──
    cflib.crtp.init_drivers()
    status = f"Connecting to {URI}..."
    draw(stdscr, status, airborne, vx, vy, vz, yaw)

    mc = None

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            status = "Connected! Press T to take off."
            draw(stdscr, status, airborne, vx, vy, vz, yaw)

            mc = MotionCommander(scf, default_height=DEFAULT_HEIGHT)

            while True:
                # Reset velocities each frame (hold-to-move)
                vx = vy = vz = yaw = 0.0

                key = stdscr.getch()

                if key in (ord('t'), ord('T')):
                    if not airborne:
                        try:
                            status = "Taking off..."
                            draw(stdscr, status, airborne, vx, vy, vz, yaw)
                            mc.take_off(DEFAULT_HEIGHT, 0.3)
                            airborne = True
                            status = f"Airborne at {DEFAULT_HEIGHT}m! Use WASD to fly."
                        except Exception as e:
                            status = f"Takeoff failed: {e}"

                elif key in (ord('l'), ord('L')):
                    if airborne:
                        try:
                            status = "Landing..."
                            draw(stdscr, status, airborne, vx, vy, vz, yaw)
                            mc.land(0.2)
                            airborne = False
                            status = "Landed. Press T to take off again."
                        except Exception as e:
                            status = f"Land failed: {e}"

                elif key in (ord('w'), ord('W')):
                    vx = VELOCITY
                elif key in (ord('s'), ord('S')):
                    vx = -VELOCITY
                elif key in (ord('a'), ord('A')):
                    vy = VELOCITY
                elif key in (ord('d'), ord('D')):
                    vy = -VELOCITY
                elif key in (ord('q'), ord('Q')):
                    yaw = -YAW_RATE
                elif key in (ord('e'), ord('E')):
                    yaw = YAW_RATE
                elif key == curses.KEY_UP:
                    vz = VELOCITY_Z
                elif key == curses.KEY_DOWN:
                    vz = -VELOCITY_Z

                elif key == ord(' '):
                    # Emergency stop
                    try:
                        scf.cf.commander.send_stop_setpoint()
                    except Exception:
                        pass
                    airborne = False
                    status = "!! EMERGENCY STOP !! Press T to restart."

                elif key == 27:  # ESC
                    if airborne:
                        status = "Landing before exit..."
                        draw(stdscr, status, airborne, vx, vy, vz, yaw)
                        try:
                            mc.land(0.2)
                        except Exception:
                            scf.cf.commander.send_stop_setpoint()
                        airborne = False
                    status = "Exiting..."
                    draw(stdscr, status, airborne, vx, vy, vz, yaw)
                    break

                # Send velocity commands while airborne
                if airborne:
                    try:
                        mc.start_linear_motion(vx, vy, vz, yaw)
                    except Exception as e:
                        status = f"Link error: {e}"
                        airborne = False

                draw(stdscr, status, airborne, vx, vy, vz, yaw)

    except Exception as exc:
        status = f"Connection error: {exc}"
        draw(stdscr, status, False, 0, 0, 0, 0)
        stdscr.timeout(-1)  # block for keypress
        stdscr.addstr(21, 1, "Press any key to exit...")
        stdscr.refresh()
        stdscr.getch()


if __name__ == '__main__':
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
