#!/usr/bin/env python3
"""
Crazyflie connection & readiness checker.

Checks radio link, deck detection, estimator config, and attempts
a brief MotionCommander hover to confirm flight readiness.
"""
import time
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M/E7E7E7E7E7'

GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
RESET = '\033[0m'

ok   = lambda msg: print(f"  {GREEN}✓{RESET} {msg}")
fail = lambda msg: print(f"  {RED}✗{RESET} {msg}")
info = lambda msg: print(f"  {CYAN}ℹ{RESET} {msg}")


def run():
    cflib.crtp.init_drivers()

    print(f"\n{BOLD}=== CRAZYFLIE PRE-FLIGHT CHECK ==={RESET}\n")

    # 1. Connection
    print(f"{BOLD}[1] Radio Link{RESET}")
    try:
        scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
        scf.open_link()
        cf = scf.cf
        ok(f"Connected to {URI}")
    except Exception as e:
        fail(f"Cannot connect: {e}")
        print(f"\n  {YELLOW}Fix: Unplug/replug Crazyradio, power-cycle drone, "
              f"or check if another program is using the radio.{RESET}")
        return

    try:
        time.sleep(2)

        proto = cf.platform.get_protocol_version()
        info(f"Protocol version: {proto}")

        # 2. Decks
        print(f"\n{BOLD}[2] Deck Detection{RESET}")
        has_flow = False
        for deck in ['bcFlow2', 'bcFlow', 'bcZRanger2', 'bcZRanger']:
            try:
                v = int(cf.param.get_value(f'deck.{deck}'))
                if v:
                    ok(f"{deck} detected")
                    if 'Flow' in deck:
                        has_flow = True
            except Exception:
                pass

        if not has_flow:
            fail("No Flow deck — hover mode won't work!")

        # 3. Estimator
        print(f"\n{BOLD}[3] Estimator{RESET}")
        est = int(cf.param.get_value('stabilizer.estimator'))
        est_names = {0: 'auto', 1: 'complementary', 2: 'kalman'}
        if est == 2:
            ok(f"Kalman estimator active")
        else:
            fail(f"Estimator = {est} ({est_names.get(est, '?')}) — need Kalman (2)")

        selftest = int(cf.param.get_value('system.selftestPassed'))
        if selftest:
            ok("Self-test passed")
        else:
            fail("Self-test FAILED")

        # 4. Battery
        print(f"\n{BOLD}[4] Battery{RESET}")
        try:
            vbat = float(cf.param.get_value('pm.vbat'))
            if vbat > 3.7:
                ok(f"Battery: {vbat:.2f}V (good)")
            elif vbat > 3.3:
                info(f"Battery: {vbat:.2f}V (low)")
            else:
                fail(f"Battery: {vbat:.2f}V (critical!)")
        except Exception:
            info("Could not read battery voltage")

        # 5. Quick hover test
        if has_flow and est == 2 and selftest:
            print(f"\n{BOLD}[5] Hover Test (2 seconds){RESET}")
            answer = input(f"  Attempt hover? (y/n): ").strip().lower()
            if answer == 'y':
                try:
                    with MotionCommander(scf, default_height=0.2) as mc:
                        ok("Takeoff successful!")
                        time.sleep(2)
                        ok("Hovering stable")
                    ok("Landed safely")
                except Exception as e:
                    fail(f"Hover failed: {e}")
            else:
                info("Skipped hover test")

        # 6. Connection still alive?
        print(f"\n{BOLD}[6] Post-check Connection{RESET}")
        try:
            _ = cf.param.get_value('system.selftestPassed')
            ok("Connection still alive")
        except Exception:
            fail("Connection lost!")

    except Exception as e:
        fail(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            scf.close_link()
        except Exception:
            pass

    print(f"\n{BOLD}=== DONE ==={RESET}\n")


if __name__ == '__main__':
    try:
        run()
    except KeyboardInterrupt:
        print("\nAborted.")
