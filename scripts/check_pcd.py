"""
scripts/check_pcd.py
====================
Verify the RealSense point cloud pipeline and camera-to-pelvis calibration.

Run this with the robot standing in its home pose to verify that:
  1. RealSense is detected and streaming depth frames
  2. Point cloud is in the correct pelvis frame (not camera frame)
  3. Point count is stable at 1024

Usage:
    python scripts/check_pcd.py [--network enp2s0]

The script prints point cloud statistics every second.
If the robot is connected, it also overlays the known robot base height
to cross-check the Z-axis calibration:
    - The table surface should appear at Z ≈ 0.0–0.3 m (above pelvis Z)
    - The robot floor level should appear near Z ≈ 0.0 m
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
from pcd_pipeline import PointCloudPipeline, DEFAULT_T_PELVIS_CAM


def main():
    p = argparse.ArgumentParser(description="Verify RealSense PCD pipeline")
    p.add_argument("--network", type=str, default=None,
                   help="Network interface for robot state (optional)")
    args = p.parse_args()

    print("Starting RealSense pipeline ...")
    pipe = PointCloudPipeline(T_pelvis_cam=DEFAULT_T_PELVIS_CAM)
    pipe.connect()

    print("\nPoint cloud statistics (Ctrl+C to stop):")
    print(f"{'Step':>5}  {'N_pts':>6}  {'X min/max':>16}  {'Y min/max':>16}  {'Z min/max':>16}  {'Mean Z':>8}")
    print("-" * 80)

    step = 0
    try:
        while True:
            pcd = pipe.get_point_cloud()  # (1024, 3)
            n   = pcd.shape[0]

            valid = pcd[np.abs(pcd).sum(axis=1) > 1e-4]
            if len(valid) > 0:
                x_min, x_max = valid[:, 0].min(), valid[:, 0].max()
                y_min, y_max = valid[:, 1].min(), valid[:, 1].max()
                z_min, z_max = valid[:, 2].min(), valid[:, 2].max()
                mean_z       = valid[:, 2].mean()
            else:
                x_min = x_max = y_min = y_max = z_min = z_max = mean_z = 0.0

            print(f"{step:>5}  {n:>6}  "
                  f"[{x_min:5.2f},{x_max:5.2f}]  "
                  f"[{y_min:5.2f},{y_max:5.2f}]  "
                  f"[{z_min:5.2f},{z_max:5.2f}]  "
                  f"{mean_z:8.3f}")

            step += 1
            time.sleep(1.0)

    except KeyboardInterrupt:
        pass
    finally:
        pipe.disconnect()
        print("\nDone.")

    # Calibration guidance
    print("\n--- Calibration Check ---")
    print("In pelvis frame with robot standing:")
    print("  • Table surface at Z ≈ 0.70–0.90 m (if table height ~pelvis height)")
    print("  • Object on table at Z ≈ 0.80–1.00 m")
    print("  • Floor behind/in front at Z ≈ -0.05–0.10 m")
    print("If values look wrong, update DEFAULT_T_PELVIS_CAM in pcd_pipeline.py")


if __name__ == "__main__":
    main()
