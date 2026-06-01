"""
pcd_pipeline.py
===============
RealSense depth camera → egocentric point cloud in robot pelvis frame.

Matches the training data convention exactly:
  - 1024 points, XYZ only (no colour)
  - Expressed in the robot pelvis frame (egocentric)
  - Same as Isaac Sim egocentric_pcd observation

Hardware assumptions:
  - Intel RealSense D435i (or compatible D4xx) mounted on robot head/chest
  - Camera extrinsics T_pelvis_cam (4×4 SE3) calibrated offline and stored in
    the config YAML or passed at construction time.

If a RealSense is not available (offline testing), use get_dummy_pcd() which
returns a zero point cloud of the correct shape.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np

# RealSense SDK — only imported when connect() is called so offline tests work
_rs_available = True
try:
    import pyrealsense2 as rs
except ImportError:
    _rs_available = False


class PointCloudPipeline:
    """
    Captures depth frames from a RealSense camera and produces a pelvis-frame
    point cloud of exactly N_PTS points.

    Parameters
    ----------
    T_pelvis_cam : np.ndarray, shape (4, 4)
        Rigid transform: camera frame → pelvis frame.
        Calibrate with e.g. hand-eye calibration or a checkerboard.
    n_pts : int
        Number of points to return (must match training: 1024).
    depth_min_m : float
        Minimum depth to include (metres). Removes too-close points.
    depth_max_m : float
        Maximum depth to include (metres). Removes floor/background.
    width, height : int
        RealSense depth resolution.
    fps : int
        RealSense depth framerate.
    """

    N_PTS = 1024

    def __init__(
        self,
        T_pelvis_cam: np.ndarray,
        n_pts: int = 1024,
        depth_min_m: float = 0.15,
        depth_max_m: float = 2.0,
        width: int = 848,
        height: int = 480,
        fps: int = 30,
    ):
        if T_pelvis_cam.shape != (4, 4):
            raise ValueError("T_pelvis_cam must be a 4×4 SE3 matrix.")

        self.T_pelvis_cam = T_pelvis_cam.astype(np.float32)
        self.n_pts        = n_pts
        self.depth_min_m  = depth_min_m
        self.depth_max_m  = depth_max_m
        self.width        = width
        self.height       = height
        self.fps          = fps

        self._pipeline: Optional[object] = None
        self._intrinsics: Optional[object] = None
        self._latest_pcd  = np.zeros((n_pts, 3), dtype=np.float32)
        self._pcd_lock    = threading.Lock()
        self._capture_thread: Optional[threading.Thread] = None
        self._running     = False

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Start RealSense pipeline and background capture thread."""
        if not _rs_available:
            raise ImportError(
                "pyrealsense2 not installed. "
                "Run: pip install pyrealsense2\n"
                "Or use get_dummy_pcd() for offline testing."
            )

        self._pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)

        profile  = self._pipeline.start(cfg)
        depth_sp = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self._intrinsics = depth_sp.get_intrinsics()

        # Warm up — discard first few frames (auto-exposure settling)
        for _ in range(10):
            self._pipeline.wait_for_frames()

        self._running = True
        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._capture_thread.start()
        print(f"[PCDPipeline] RealSense connected ({self.width}×{self.height}@{self.fps}fps)")

    def disconnect(self) -> None:
        self._running = False
        if self._pipeline is not None:
            self._pipeline.stop()

    # ── Background capture ─────────────────────────────────────────────────────

    def _capture_loop(self) -> None:
        while self._running:
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=200)
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    continue

                pcd = self._depth_to_pelvis_pcd(depth_frame)

                with self._pcd_lock:
                    self._latest_pcd = pcd

            except Exception as e:
                print(f"[PCDPipeline] Capture error: {e}")
                time.sleep(0.01)

    def _depth_to_pelvis_pcd(self, depth_frame) -> np.ndarray:
        """Convert a RealSense depth frame → (n_pts, 3) pelvis-frame point cloud."""
        intr = self._intrinsics
        depth_scale = self._pipeline.get_active_profile() \
            .get_device().first_depth_sensor().get_depth_scale()

        depth_image = np.asanyarray(depth_frame.get_data()).astype(np.float32) * depth_scale

        # Build pixel grid
        v, u = np.meshgrid(
            np.arange(self.height, dtype=np.float32),
            np.arange(self.width,  dtype=np.float32),
            indexing="ij",
        )

        z = depth_image  # (H, W)
        x = (u - intr.ppx) * z / intr.fx
        y = (v - intr.ppy) * z / intr.fy

        pts_cam = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (H*W, 3)

        # Filter by depth range and remove zero-depth pixels
        depth_vals = pts_cam[:, 2]
        mask = (depth_vals > self.depth_min_m) & (depth_vals < self.depth_max_m)
        pts_cam = pts_cam[mask]

        if len(pts_cam) == 0:
            return np.zeros((self.n_pts, 3), dtype=np.float32)

        # Transform to pelvis frame: pts_pelvis = R @ pts_cam + t
        R = self.T_pelvis_cam[:3, :3]
        t = self.T_pelvis_cam[:3,  3]
        pts_pelvis = (pts_cam @ R.T) + t  # (N, 3)

        # Subsample to n_pts
        pts_pelvis = self._subsample(pts_pelvis)
        return pts_pelvis.astype(np.float32)

    def _subsample(self, pts: np.ndarray) -> np.ndarray:
        """Random subsample to exactly n_pts points."""
        N = len(pts)
        if N == 0:
            return np.zeros((self.n_pts, 3), dtype=np.float32)
        if N >= self.n_pts:
            idx = np.random.choice(N, self.n_pts, replace=False)
            return pts[idx]
        # Tile if too few points
        reps = (self.n_pts + N - 1) // N
        return np.tile(pts, (reps, 1))[: self.n_pts]

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_point_cloud(self) -> np.ndarray:
        """
        Return the latest point cloud in pelvis frame.

        Returns
        -------
        np.ndarray, shape (1024, 3), dtype float32
            XYZ coordinates in metres, in robot pelvis frame.
        """
        with self._pcd_lock:
            return self._latest_pcd.copy()

    def get_dummy_pcd(self) -> np.ndarray:
        """Return a zero point cloud — for offline testing without hardware."""
        return np.zeros((self.n_pts, 3), dtype=np.float32)


# ── Camera extrinsics helpers ──────────────────────────────────────────────────

def make_T_pelvis_cam(
    translation_xyz: list[float],
    rotation_wxyz: list[float],
) -> np.ndarray:
    """
    Build the 4×4 camera→pelvis transform from a translation and quaternion.

    Parameters
    ----------
    translation_xyz : [x, y, z]  metres
    rotation_wxyz   : [w, x, y, z]  unit quaternion

    Returns
    -------
    np.ndarray, shape (4, 4)
    """
    w, x, y, z = rotation_wxyz
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),         1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3,  3] = translation_xyz
    return T.astype(np.float32)


# ── Default extrinsics placeholder ─────────────────────────────────────────────
# REPLACE with your actual calibrated values before deployment.
# Run: python scripts/check_pcd.py to visually verify alignment.

DEFAULT_T_PELVIS_CAM = make_T_pelvis_cam(
    translation_xyz=[0.07, 0.0, 0.55],   # camera ~55cm above pelvis, 7cm forward
    rotation_wxyz=[0.0, 0.707, 0.0, 0.707],  # camera pointing forward-down (PLACEHOLDER)
)
