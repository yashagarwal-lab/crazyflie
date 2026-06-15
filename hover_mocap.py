#!/usr/bin/env python3
"""
MoCap-guided hover for Crazyflie using OptiTrack + cflib.

Flow:
  1. Connect to Crazyflie via USB/radio
  2. Stream position from OptiTrack (NatNet labeled markers)
  3. Run a simple PID to hold a target (x, y, z)
  4. Send velocity setpoints via cflib Motion Commander

Requirements:
    pip install cflib
    NatNetClient.py must be in the same directory (already have it)

Usage:
    python hover_mocap.py
"""

import time
import threading
import logging
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander
from NatNetClient import NatNetClient

# ── Config ──────────────────────────────────────────────────────────────────
URI = "radio://0/80/2M/E7E7E7E701"   # ← change to your Crazyflie URI
                                       # USB: "usb://0"

TARGET_X   =  0.0  # metres (Motive frame)
TARGET_Y   =  0.0   # metres
TARGET_Z   =  0.5    # metres  ← hover height

MAX_SPEED  =  0.3    # m/s cap on any axis
LOOP_HZ    = 50      # controller update rate

# ── PID gains (tune these) ───────────────────────────────────────────────────
# Start conservative; increase Kp if drone is sluggish
KP_XY, KI_XY, KD_XY = 0.6, 0.05, 0.15
KP_Z,  KI_Z,  KD_Z  = 0.8, 0.08, 0.20

# ── Shared MoCap state ───────────────────────────────────────────────────────
class PoseState:
    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.valid = False
        self.frame_count = 0

pose = PoseState()


def labeled_marker_callback(marker_id, pos):
    """NatNet callback — remaps OptiTrack YZX → Crazyflie XYZ."""
    with pose.lock:
        pose.y, pose.z, pose.x = pos   # same remap as your test script
        pose.frame_count += 1
        pose.valid = True


# ── Simple PID ───────────────────────────────────────────────────────────────
class PID:
    def __init__(self, kp, ki, kd, integral_limit=1.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.integral_limit = integral_limit
        self._integral = 0.0
        self._prev_error = 0.0
        self._prev_time  = None

    def update(self, error, now):
        if self._prev_time is None:
            dt = 0.0
        else:
            dt = now - self._prev_time
        self._prev_time = now

        self._integral += error * dt
        # Anti-windup clamp
        self._integral = max(-self.integral_limit,
                             min(self.integral_limit, self._integral))

        derivative = (error - self._prev_error) / dt if dt > 0 else 0.0
        self._prev_error = error

        return self.kp * error + self.ki * self._integral + self.kd * derivative

    def reset(self):
        self._integral   = 0.0
        self._prev_error = 0.0
        self._prev_time  = None


def clamp(val, limit):
    return max(-limit, min(limit, val))


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    logging.basicConfig(level=logging.ERROR)   # suppress cflib noise

    # 1. Start MoCap
    print("[MoCap]  Connecting to Motive NatNet...")
    client = NatNetClient()
    client.labeledMarkerListener = labeled_marker_callback
    client.run()
    print("[MoCap]  Streaming started. Waiting for first frame...")

    timeout = time.time() + 5.0
    while not pose.valid:
        if time.time() > timeout:
            print("[MoCap]  ERROR: No data received. Is Motive streaming?")
            client.stop()
            return
        time.sleep(0.05)

    with pose.lock:
        print(f"[MoCap]  First frame: x={pose.x:.3f}  y={pose.y:.3f}  z={pose.z:.3f}")

    # 2. Connect to Crazyflie
    cflib.crtp.init_drivers()
    print(f"[CF]     Connecting to {URI} ...")

    pid_x = PID(KP_XY, KI_XY, KD_XY)
    pid_y = PID(KP_XY, KI_XY, KD_XY)
    pid_z = PID(KP_Z,  KI_Z,  KD_Z)

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        print("[CF]     Connected! Arming and taking off...")

        # MotionCommander handles take-off/land and accepts velocity setpoints
        with MotionCommander(scf, default_height=TARGET_Z) as mc:
            print(f"[CTRL]   Hovering at z={TARGET_Z}m. Target: "
                  f"x={TARGET_X}, y={TARGET_Y}, z={TARGET_Z}")
            print("         Press Ctrl+C to land.\n")

            dt = 1.0 / LOOP_HZ
            pid_x.reset(); pid_y.reset(); pid_z.reset()

            try:
                while True:
                    loop_start = time.time()

                    with pose.lock:
                        cx, cy, cz = pose.x, pose.y, pose.z
                        got_data   = pose.valid

                    if not got_data:
                        mc.stop()   # hold in place if MoCap drops out
                        time.sleep(dt)
                        continue

                    now = time.time()
                    ex = TARGET_X - cx
                    ey = TARGET_Y - cy
                    ez = TARGET_Z - cz

                    vx = clamp(pid_x.update(ex, now), MAX_SPEED)
                    vy = clamp(pid_y.update(ey, now), MAX_SPEED)
                    vz = clamp(pid_z.update(ez, now), MAX_SPEED)

                    # MotionCommander: start_linear_motion(vx, vy, vz_rate)
                    # vx = forward, vy = left, vz = up  (body or world frame
                    # depending on your firmware — adjust signs if needed)
                    mc.start_linear_motion(vx, vy, vz)

                    # Status line
                    print(f"\r  pos=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})  "
                          f"err=({ex:+.3f}, {ey:+.3f}, {ez:+.3f})  "
                          f"vel=({vx:+.3f}, {vy:+.3f}, {vz:+.3f})  ",
                          end="", flush=True)

                    # Sleep for remainder of loop period
                    elapsed = time.time() - loop_start
                    time.sleep(max(0, dt - elapsed))

            except KeyboardInterrupt:
                print("\n[CTRL]   Ctrl+C — landing...")
                # MotionCommander.__exit__ handles landing automatically

    print("[CF]     Landed and disconnected.")
    client.stop()
    print("[MoCap]  Done.")


if __name__ == '__main__':
    main()