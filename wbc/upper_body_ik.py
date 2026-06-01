"""
wbc/upper_body_ik.py — Standalone PINK IK for G1 upper body (arms + waist).

No Isaac Lab dependency. Mirrors G1WBCUpperbodyController from
isaaclab_arena_g1 but is self-contained for real robot deployment.

Input:  target SE3 poses for left/right wrist in pelvis frame (from EquivDP3 action)
Output: 29D SDK2 joint position targets (arm + waist joints set; leg indices zeroed)
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
import pink
from pink.tasks import FrameTask, PostureTask
from scipy.spatial.transform import Rotation as R

from .robot_model_real import (
    G1RobotModel,
    ARM_JOINT_NAMES,
    WAIST_JOINT_NAMES,
    SDK2_JOINT_INDEX,
    SDK2_N_JOINTS,
    LEFT_WRIST_LINK,
    RIGHT_WRIST_LINK,
)


class G1UpperBodyIK:
    """
    PINK IK controller for G1 arms and waist.

    Solves for arm + waist joint targets given left/right wrist SE3 targets
    expressed in the pelvis (base) frame.
    """

    def __init__(self, robot_model: G1RobotModel, cfg: dict):
        """
        Args:
            robot_model: G1RobotModel instance
            cfg: dict with keys:
                ik_dt               — IK integration timestep (s)
                ik_steps_per_frame  — solver iterations per call
                arm_kp, arm_kd      — PD gains for arm joints (for output metadata)
                waist_kp, waist_kd  — PD gains for waist
        """
        self.robot_model = robot_model
        self.dt          = cfg.get("ik_dt", 0.02)
        self.n_steps     = cfg.get("ik_steps_per_frame", 1)
        self.arm_kp      = cfg.get("arm_kp", 80.0)
        self.arm_kd      = cfg.get("arm_kd", 2.0)
        self.waist_kp    = cfg.get("waist_kp", 200.0)
        self.waist_kd    = cfg.get("waist_kd", 5.0)

        model = robot_model.model
        data  = robot_model.data

        # ── Pink configuration ─────────────────────────────────────────────
        self.configuration = pink.Configuration(model, data, robot_model.q_default.copy())
        self.configuration.model.lowerPositionLimit = model.lowerPositionLimit
        self.configuration.model.upperPositionLimit = model.upperPositionLimit

        # ── EEF frame tasks ───────────────────────────────────────────────
        self.left_task  = FrameTask(LEFT_WRIST_LINK,  position_cost=1.0, orientation_cost=0.5)
        self.right_task = FrameTask(RIGHT_WRIST_LINK, position_cost=1.0, orientation_cost=0.5)
        self.posture_task = PostureTask(cost=0.01, lm_damping=1.0)

        for task in [self.left_task, self.right_task, self.posture_task]:
            task.set_target_from_configuration(self.configuration)

        self._tasks = [self.left_task, self.right_task, self.posture_task]
        print("[G1UpperBodyIK] PINK IK initialized")

    def set_current_config(self, sdk2_q: np.ndarray) -> None:
        """Update IK warm-start from current SDK2 joint readings (29,)."""
        pin_q = self.robot_model.sdk2_to_pin_config(sdk2_q)
        self.configuration.q = pin_q
        self.configuration.update()

    def solve(
        self,
        left_wrist_pos: np.ndarray,    # (3,)  in pelvis frame
        left_wrist_quat_wxyz: np.ndarray,  # (4,) [w,x,y,z]
        right_wrist_pos: np.ndarray,   # (3,)
        right_wrist_quat_wxyz: np.ndarray, # (4,)
        left_hand_state: int = 0,
        right_hand_state: int = 0,
    ) -> np.ndarray:
        """
        Solve IK and return full SDK2 joint position target array (29,).

        Arm and waist joints are set; leg joints are returned as 0.0
        (caller should blend with current leg joint positions or ignore).

        Args:
            left_wrist_pos/quat_wxyz:  left wrist target pose in pelvis frame
            right_wrist_pos/quat_wxyz: right wrist target pose in pelvis frame
            left_hand_state:  0 = open, 1 = close (hands not in URDF; for metadata)
            right_hand_state: 0 = open, 1 = close
        Returns:
            sdk2_q_target: (29,) joint positions; leg indices = 0 (not commanded)
        """
        # Build SE3 targets
        def make_se3(pos, quat_wxyz):
            # quat_wxyz [w,x,y,z] → scipy [x,y,z,w]
            q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            rot = R.from_quat(q_xyzw).as_matrix()
            return pin.SE3(rot, pos)

        self.left_task.set_target(make_se3(left_wrist_pos, left_wrist_quat_wxyz))
        self.right_task.set_target(make_se3(right_wrist_pos, right_wrist_quat_wxyz))

        for _ in range(self.n_steps):
            velocity = pink.solve_ik(
                self.configuration,
                self._tasks,
                dt=self.dt,
                solver="osqp",
            )
            self.configuration.q = pin.integrate(
                self.configuration.model, self.configuration.q, velocity * self.dt
            )
            self.configuration.update()

        # Extract result as SDK2 array
        sdk2_q = self.robot_model.pin_config_to_sdk2(self.configuration.q)

        # Zero out leg indices (0-11) — only arm/waist are IK-controlled
        sdk2_q[:12] = 0.0

        return sdk2_q

    def get_arm_sdk2_indices(self) -> list[int]:
        """Return SDK2 motor indices for arm joints (14 joints, indices 15-28)."""
        return [SDK2_JOINT_INDEX[n] for n in ARM_JOINT_NAMES]

    def get_waist_sdk2_indices(self) -> list[int]:
        """Return SDK2 motor indices for waist joints (3 joints, indices 12-14)."""
        return [SDK2_JOINT_INDEX[n] for n in WAIST_JOINT_NAMES]


def pos_quat_to_se3(pos: np.ndarray, quat_wxyz: np.ndarray) -> np.ndarray:
    """Build a 4x4 homogeneous transform from position + quaternion [w,x,y,z]."""
    q_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    rot = R.from_quat(q_xyzw).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rot
    T[:3,  3] = pos
    return T
