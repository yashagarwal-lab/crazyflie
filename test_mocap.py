#!/usr/bin/env python3
"""
OptiTrack data validation script (marker set mode).

Connects to Motive via NatNetClient and prints position data from
marker sets. Designed for a single-marker Crazyflie setup.

Usage:
    python test_mocap.py

Move the Crazyflie by hand and verify:
  - +X in Motive → +X in terminal
  - +Y in Motive → +Y in terminal
  - +Z in Motive → +Z in terminal
  - Receive rate ≥ 100 Hz

NOTE: With a single marker, only position (x, y, z) is tracked.
      Orientation (yaw) is NOT available.
"""
import time
import threading
from NatNetClient import NatNetClient


# ── Shared state ──
class PoseState:
    """Thread-safe container for the latest marker position."""
    def __init__(self):
        self.lock = threading.Lock()
        self.x = self.y = self.z = 0.0
        self.set_name = ""
        self.frame_count = 0
        self.valid = False


pose = PoseState()


# ── Callback ──
def marker_set_callback(set_name, marker_index, pos):
    """Called by NatNetClient for each marker in each marker set per frame.

    With a single marker on the drone, we take marker_index 0
    from the first non-'all' marker set we see.
    """
    # Motive sends an 'all' set containing every marker — skip it
    # and use the named set instead for clarity.
    if set_name.lower() == 'all' and pose.set_name != '':
        return

    with pose.lock:
        if marker_index == 0:
            pose.x, pose.y, pose.z = pos
            pose.set_name = set_name
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
    print(f"\n{BOLD}=== OPTITRACK DATA VALIDATION (Marker Set) ==={RESET}\n")

    client = NatNetClient()
    # IPs already default to 127.0.0.1 (same PC as Motive)
    client.markerSetListener = marker_set_callback

    print(f"  {CYAN}Server IP   :{RESET} {client.serverIPAddress}")
    print(f"  {CYAN}Local IP    :{RESET} {client.localIPAddress}")
    print(f"  {CYAN}Multicast   :{RESET} {client.multicastAddress}")
    print(f"  {CYAN}Data port   :{RESET} {client.dataPort}")
    print(f"  {CYAN}Command port:{RESET} {client.commandPort}")
    print()

    print(f"  Starting NatNet listener... ", end="", flush=True)
    client.run()
    print(f"{GREEN}OK{RESET}")
    print(f"  {YELLOW}Waiting for marker set data (is Motive streaming?){RESET}")
    print(f"  {YELLOW}Press Ctrl+C to stop.{RESET}\n")

    # Print header
    print(f"  {'Set Name':>16}  {'X':>8}  {'Y':>8}  {'Z':>8}  {'Hz':>6}")
    print(f"  {'─' * 52}")

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

                x, y, z = pose.x, pose.y, pose.z
                count = pose.frame_count
                set_name = pose.set_name

            # Compute receive rate (Hz)
            dt = now - last_rate_time
            if dt >= 1.0:
                hz = (count - last_count) / dt
                last_count = count
                last_rate_time = now

            # Colour the Hz
            if hz >= 100:
                hz_str = f"{GREEN}{hz:6.1f}{RESET}"
            elif hz > 0:
                hz_str = f"{YELLOW}{hz:6.1f}{RESET}"
            else:
                hz_str = f"   ---"

            # Truncate set name for display
            name_display = set_name[:16]

            print(f"\r  {name_display:>16}  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}  {hz_str}   ",
                  end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Shutting down...{RESET}")

    client.stop()
    print(f"  {GREEN}Done.{RESET}\n")


if __name__ == '__main__':
    main()
