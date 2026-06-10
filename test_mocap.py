#!/usr/bin/env python3
"""
OptiTrack data validation script.

Connects to Motive via NatNetClient and prints rigid body pose data
to verify data reception, axis alignment, and stream quality.

Usage:
    uv run python test_mocap.py

Move the Crazyflie by hand and verify:
  - +X in Motive → +X in terminal
  - +Y in Motive → +Y in terminal
  - +Z in Motive → +Z in terminal
  - Quaternion norm ≈ 1.0
  - Receive rate ≥ 100 Hz
"""
import math
import time
import threading
from NatNetClient import NatNetClient


# ── Shared state ──
class PoseState:
    """Thread-safe container for the latest rigid body pose."""
    def __init__(self):
        self.lock = threading.Lock()
        self.rb_id = -1
        self.x = self.y = self.z = 0.0
        self.qx = self.qy = self.qz = 0.0
        self.qw = 1.0
        self.frame_count = 0
        self.valid = False


pose = PoseState()


# ── Callback ──
def rigid_body_callback(rb_id, pos, rot):
    """Called by NatNetClient on every rigid body in every frame."""
    with pose.lock:
        pose.rb_id = rb_id
        pose.x, pose.y, pose.z = pos
        pose.qx, pose.qy, pose.qz, pose.qw = rot
        pose.frame_count += 1
        pose.valid = True


# ── Terminal colours ──
GREEN = '\033[92m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
RED = '\033[91m'
BOLD = '\033[1m'
RESET = '\033[0m'


def main():
    print(f"\n{BOLD}=== OPTITRACK DATA VALIDATION ==={RESET}\n")

    client = NatNetClient()
    # IPs already default to 127.0.0.1 (same PC as Motive)
    client.rigidBodyListener = rigid_body_callback

    print(f"  {CYAN}Server IP   :{RESET} {client.serverIPAddress}")
    print(f"  {CYAN}Local IP    :{RESET} {client.localIPAddress}")
    print(f"  {CYAN}Multicast   :{RESET} {client.multicastAddress}")
    print(f"  {CYAN}Data port   :{RESET} {client.dataPort}")
    print(f"  {CYAN}Command port:{RESET} {client.commandPort}")
    print()

    print(f"  Starting NatNet listener... ", end="", flush=True)
    client.run()
    print(f"{GREEN}OK{RESET}")
    print(f"  {YELLOW}Waiting for rigid body data (is Motive streaming?){RESET}")
    print(f"  {YELLOW}Press Ctrl+C to stop.{RESET}\n")

    # Print header
    print(f"  {'ID':>3}  {'X':>8}  {'Y':>8}  {'Z':>8}  "
          f"{'qx':>7}  {'qy':>7}  {'qz':>7}  {'qw':>7}  "
          f"{'|q|':>5}  {'Hz':>6}")
    print(f"  {'─' * 85}")

    last_print = time.time()
    last_count = 0
    last_rate_time = time.time()
    hz = 0.0

    try:
        while True:
            now = time.time()

            # Print at ~10 Hz
            if now - last_print < 0.1:
                time.sleep(0.01)
                continue

            last_print = now

            with pose.lock:
                if not pose.valid:
                    print(f"\r  {YELLOW}Waiting for data...{RESET}", end="", flush=True)
                    continue

                rb_id = pose.rb_id
                x, y, z = pose.x, pose.y, pose.z
                qx, qy, qz, qw = pose.qx, pose.qy, pose.qz, pose.qw
                count = pose.frame_count

            # Compute quaternion norm
            qnorm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)

            # Compute receive rate (Hz)
            dt = now - last_rate_time
            if dt >= 1.0:
                hz = (count - last_count) / dt
                last_count = count
                last_rate_time = now

            # Colour the quaternion norm
            if 0.99 <= qnorm <= 1.01:
                qn_str = f"{GREEN}{qnorm:.3f}{RESET}"
            else:
                qn_str = f"{RED}{qnorm:.3f}{RESET}"

            # Colour the Hz
            if hz >= 100:
                hz_str = f"{GREEN}{hz:6.1f}{RESET}"
            elif hz > 0:
                hz_str = f"{YELLOW}{hz:6.1f}{RESET}"
            else:
                hz_str = f"   ---"

            print(f"\r  {rb_id:>3}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}  "
                  f"{qx:>7.4f}  {qy:>7.4f}  {qz:>7.4f}  {qw:>7.4f}  "
                  f"{qn_str}  {hz_str}   ", end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Shutting down...{RESET}")

    client.stop()
    print(f"  {GREEN}Done.{RESET}\n")


if __name__ == '__main__':
    main()
