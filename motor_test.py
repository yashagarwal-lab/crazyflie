#!/usr/bin/env python3
"""
Motor test: spins each motor individually to diagnose hardware issues.
Uses motorPowerSet parameters to directly control each motor's PWM.

Motor layout (top view, USB connector = back):
     Back
  M4     M1
      X
  M3     M2
     Front

SAFETY: Remove propellers or secure the drone before running!
"""
import time
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'usb://0'
TEST_POWER = 15000    # low PWM for testing (0-65535)
SPIN_TIME = 2.0       # seconds per motor

logging.basicConfig(level=logging.ERROR)


def motor_test():
    cflib.crtp.init_drivers()

    print(f"Connecting to {URI}...")
    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf
        print("Connected!")
        print(f"Protocol version: {cf.platform.get_protocol_version()}\n")

        # Wait for param TOC to be fully loaded
        print("Waiting for parameters to load...")
        time.sleep(2.0)

        # Force arm for older firmware (protocol < 12)
        if cf.platform.get_protocol_version() < 12:
            print("Old firmware detected, using system.forceArm...")
            cf.param.set_value('system.forceArm', 1)
            time.sleep(0.5)

        print(f"\nWill test each motor at PWM={TEST_POWER} for {SPIN_TIME}s each.")
        print("Motor layout (top view):")
        print("     Back")
        print("  M4     M1")
        print("  M3     M2")
        print("     Front\n")

        input("Press ENTER to start (remove props or secure drone!)...")

        try:
            # Enable direct motor power override (mode 1 = individual PWM)
            cf.param.set_value('motorPowerSet.enable', 1)
            time.sleep(0.2)

            for motor_num in range(1, 5):
                # Zero all motors first
                for m in range(1, 5):
                    cf.param.set_value(f'motorPowerSet.m{m}', 0)
                time.sleep(0.1)

                print(f"  Spinning M{motor_num}...", end=" ", flush=True)
                cf.param.set_value(f'motorPowerSet.m{motor_num}', TEST_POWER)
                time.sleep(SPIN_TIME)
                cf.param.set_value(f'motorPowerSet.m{motor_num}', 0)
                print("done.")
                time.sleep(0.5)

            print("\nAll motors tested.")

        finally:
            # Always disable override and disarm
            for m in range(1, 5):
                cf.param.set_value(f'motorPowerSet.m{m}', 0)
            cf.param.set_value('motorPowerSet.enable', 0)
            if cf.platform.get_protocol_version() < 12:
                cf.param.set_value('system.forceArm', 0)
            print("Motor override disabled. Disarmed.")


if __name__ == '__main__':
    try:
        motor_test()
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        print(f"\nError: {e}")
