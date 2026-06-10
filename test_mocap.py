#!/usr/bin/env python3
"""
OptiTrack data validation script (unlabeled marker mode).

Connects to Motive via NatNetClient and prints position data from
unlabeled markers. Designed for a single-marker Crazyflie setup.

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
        self.marker_count = 0  # how many unlabeled markers Motive sees
        self.frame_count = 0
        self.valid = False


pose = PoseState()


# ── Callback ──
def unlabeled_marker_callback(marker_index, pos):
    """Called by NatNetClient for each unlabeled marker in every frame.

    With a single marker on the drone and nothing else in the arena,
    marker_index 0 is the Crazyflie. If multiple markers appear,
    we take index 0 and print a warning.
    """
    with pose.lock:
        # Track how many markers are visible this frame
        pose.marker_count = max(pose.marker_count, marker_index + 1)

        if marker_index == 0:
            pose.x, pose.y, pose.z = pos
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
    print(f"\n{BOLD}=== OPTITRACK DATA VALIDATION (Unlabeled Marker) ==={RESET}\n")

    client = NatNetClient()
    # IPs already default to 127.0.0.1 (same PC as Motive)
    client.unlabeledMarkerListener = unlabeled_marker_callback

    print(f"  {CYAN}Server IP   :{RESET} {client.serverIPAddress}")
    print(f"  {CYAN}Local IP    :{RESET} {client.localIPAddress}")
    print(f"  {CYAN}Multicast   :{RESET} {client.multicastAddress}")
    print(f"  {CYAN}Data port   :{RESET} {client.dataPort}")
    print(f"  {CYAN}Command port:{RESET} {client.commandPort}")
    print()

    print(f"  Starting NatNet listener... ", end="", flush=True)
    client.run()
    print(f"{GREEN}OK{RESET}")
    print(f"  {YELLOW}Waiting for unlabeled marker data (is Motive streaming?){RESET}")
    print(f"  {YELLOW}Ensure the Crazyflie marker is the ONLY marker in the arena.{RESET}")
    print(f"  {YELLOW}Press Ctrl+C to stop.{RESET}\n")

    # Print header
    print(f"  {'X':>8}  {'Y':>8}  {'Z':>8}  {'Markers':>7}  {'Hz':>6}")
    print(f"  {'─' * 45}")

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
                n_markers = pose.marker_count
                # Reset for next frame's count
                pose.marker_count = 0

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

            # Colour the marker count
            if n_markers == 1:
                mk_str = f"{GREEN}{n_markers:>7}{RESET}"
            elif n_markers > 1:
                mk_str = f"{RED}{n_markers:>7}{RESET}"
            else:
                mk_str = f"{YELLOW}{'?':>7}{RESET}"

            print(f"\r  {x:>8.3f}  {y:>8.3f}  {z:>8.3f}  {mk_str}  {hz_str}   ",
                  end="", flush=True)

    except KeyboardInterrupt:
        print(f"\n\n  {YELLOW}Shutting down...{RESET}")

    client.stop()
    print(f"  {GREEN}Done.{RESET}\n")


if __name__ == '__main__':
    main()
