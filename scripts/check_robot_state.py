"""
scripts/check_robot_state.py
============================
Print live robot state from the G1 via SDK2.  Run this BEFORE deploying
to confirm communication, joint readings, and IMU are healthy.

Usage:
    python scripts/check_robot_state.py --network enp2s0

Expected output (robot standing):
    Joint angles:  [ 0.00  0.00  0.00  0.35  -0.23  0.00  ... ]
    IMU rpy (deg): [ 0.1   -0.2   90.0 ]
    Base pos:      [ 0.00   0.00   0.74 ]
    State age:     0.003 s

If State age > 0.1 s, the network connection is failing.
"""

from __future__ import annotations

import argparse
import sys
import pathlib
import time

_HERE = pathlib.Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
from robot_interface import G1RobotInterface, ARM_INDICES


def main():
    p = argparse.ArgumentParser(description="Print live G1 robot state")
    p.add_argument("--network", type=str, required=True, help="Network interface (e.g. enp2s0)")
    p.add_argument("--hz",      type=float, default=2.0,  help="Print rate in Hz")
    args = p.parse_args()

    robot = G1RobotInterface(network_interface=args.network, dry_run=True)
    robot.connect()

    print("\nLive robot state (Ctrl+C to stop):")
    print("-" * 60)

    try:
        while True:
            state = robot.get_state()
            age   = time.time() - state.low_state_ts

            rpy_deg = np.degrees(state.imu_rpy)

            print(f"\r"
                  f"q_arm: [{' '.join(f'{v:6.3f}' for v in state.q[15:29])}]  "
                  f"IMU rpy: [{rpy_deg[0]:5.1f} {rpy_deg[1]:5.1f} {rpy_deg[2]:6.1f}]°  "
                  f"base_pos: [{state.base_pos[0]:5.2f} {state.base_pos[1]:5.2f} {state.base_pos[2]:5.2f}]  "
                  f"age: {age*1000:.1f}ms",
                  end="", flush=True)

            time.sleep(1.0 / args.hz)

    except KeyboardInterrupt:
        print("\nDone.")


if __name__ == "__main__":
    main()
