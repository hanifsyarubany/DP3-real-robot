"""
wbc/robot_model_real.py — Standalone pinocchio G1 model for real robot.

No Isaac Lab dependency. Used for:
  - FK to build agent_pos (28D) from SDK2 LowState joint readings
  - IK target frame management for PINK upper-body controller
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


# ── G1 joint name → SDK2 motor_cmd index mapping ────────────────────────────
# Verified against unitree_sdk2py G1 examples and g1_supplemental_info.py.
# motor_cmd[i] controls the joint at index i.
SDK2_JOINT_INDEX: dict[str, int] = {
    "left_hip_pitch_joint":      0,
    "left_hip_roll_joint":       1,
    "left_hip_yaw_joint":        2,
    "left_knee_joint":           3,
    "left_ankle_pitch_joint":    4,
    "left_ankle_roll_joint":     5,
    "right_hip_pitch_joint":     6,
    "right_hip_roll_joint":      7,
    "right_hip_yaw_joint":       8,
    "right_knee_joint":          9,
    "right_ankle_pitch_joint":  10,
    "right_ankle_roll_joint":   11,
    "waist_yaw_joint":          12,
    "waist_roll_joint":         13,
    "waist_pitch_joint":        14,
    "left_shoulder_pitch_joint": 15,
    "left_shoulder_roll_joint":  16,
    "left_shoulder_yaw_joint":   17,
    "left_elbow_joint":          18,
    "left_wrist_roll_joint":     19,
    "left_wrist_pitch_joint":    20,
    "left_wrist_yaw_joint":      21,
    "right_shoulder_pitch_joint": 22,
    "right_shoulder_roll_joint":  23,
    "right_shoulder_yaw_joint":   24,
    "right_elbow_joint":          25,
    "right_wrist_roll_joint":     26,
    "right_wrist_pitch_joint":    27,
    "right_wrist_yaw_joint":      28,
}
SDK2_N_JOINTS = 29

# Arm indices only (for PINK IK and arm LowCmd)
ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]
ARM_SDK2_INDICES = [SDK2_JOINT_INDEX[n] for n in ARM_JOINT_NAMES]  # [15..28]

WAIST_JOINT_NAMES = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
WAIST_SDK2_INDICES = [SDK2_JOINT_INDEX[n] for n in WAIST_JOINT_NAMES]  # [12,13,14]

# EEF link names (must match URDF)
LEFT_WRIST_LINK  = "left_wrist_yaw_link"
RIGHT_WRIST_LINK = "right_wrist_yaw_link"
PELVIS_LINK      = "pelvis"


class G1RobotModel:
    """
    Pinocchio-based G1 robot model for FK and IK.

    Floating base is treated as FIXED (identity) for FK so that
    computed EEF poses are naturally in the pelvis frame — matching
    how training data was generated in simulation.
    """

    def __init__(self, urdf_path: str, mesh_dir: str):
        self.robot = pin.RobotWrapper.BuildFromURDF(
            filename=urdf_path,
            package_dirs=[mesh_dir],
            root_joint=None,   # fixed base — pelvis is the root
        )
        self.model = self.robot.model
        self.data  = self.robot.data

        # Build pinocchio joint name → configuration index map
        self._joint_name_to_cfg_idx: dict[str, int] = {}
        for jname, jidx in SDK2_JOINT_INDEX.items():
            try:
                pin_id = self.model.getJointId(jname)
                cfg_idx = self.model.joints[pin_id].idx_q
                self._joint_name_to_cfg_idx[jname] = cfg_idx
            except Exception:
                print(f"[G1RobotModel] Warning: joint '{jname}' not found in URDF")

        self.n_cfg = self.model.nq

        # Frame IDs
        self._left_wrist_fid  = self.model.getFrameId(LEFT_WRIST_LINK)
        self._right_wrist_fid = self.model.getFrameId(RIGHT_WRIST_LINK)

        # Default joint configuration (neutral pose)
        self.q_default = pin.neutral(self.model)

        print(f"[G1RobotModel] Loaded from {urdf_path}")
        print(f"[G1RobotModel]   nq={self.model.nq}, nv={self.model.nv}")

    # ── Configuration from SDK2 LowState ─────────────────────────────────────

    def sdk2_to_pin_config(self, sdk2_q: np.ndarray) -> np.ndarray:
        """
        Convert SDK2 joint array (29,) → pinocchio configuration vector (nq,).

        Args:
            sdk2_q: joint positions from SDK2 motor_state[i].q, shape (29,)
        Returns:
            pin_q: pinocchio config vector, shape (nq,)
        """
        pin_q = self.q_default.copy()
        for jname, sdk2_idx in SDK2_JOINT_INDEX.items():
            cfg_idx = self._joint_name_to_cfg_idx.get(jname)
            if cfg_idx is not None:
                pin_q[cfg_idx] = sdk2_q[sdk2_idx]
        return pin_q

    def pin_config_to_sdk2(self, pin_q: np.ndarray) -> np.ndarray:
        """Inverse of sdk2_to_pin_config: pinocchio config → SDK2 array (29,)."""
        sdk2_q = np.zeros(SDK2_N_JOINTS)
        for jname, sdk2_idx in SDK2_JOINT_INDEX.items():
            cfg_idx = self._joint_name_to_cfg_idx.get(jname)
            if cfg_idx is not None:
                sdk2_q[sdk2_idx] = pin_q[cfg_idx]
        return sdk2_q

    # ── Forward Kinematics ────────────────────────────────────────────────────

    def compute_fk(self, sdk2_q: np.ndarray) -> None:
        """Run pinocchio FK from SDK2 joint readings (29,). Cache in self.data."""
        pin_q = self.sdk2_to_pin_config(sdk2_q)
        pin.forwardKinematics(self.model, self.data, pin_q)
        pin.updateFramePlacements(self.model, self.data)

    def get_wrist_poses_in_pelvis(self, sdk2_q: np.ndarray) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """
        Compute left and right wrist EEF poses in pelvis frame.

        Since the floating base is fixed at identity, FK results are already
        in the pelvis frame.

        Returns:
            {
              "left":  (pos (3,), quat_wxyz (4,)),
              "right": (pos (3,), quat_wxyz (4,)),
            }
        """
        self.compute_fk(sdk2_q)

        def frame_pose(fid) -> tuple[np.ndarray, np.ndarray]:
            T = self.data.oMf[fid]
            pos  = T.translation.copy()                            # (3,)
            rot  = R.from_matrix(T.rotation)
            quat = rot.as_quat()                                   # [x, y, z, w]
            quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # [w, x, y, z]
            return pos, quat_wxyz

        left_pos,  left_quat  = frame_pose(self._left_wrist_fid)
        right_pos, right_quat = frame_pose(self._right_wrist_fid)

        return {"left": (left_pos, left_quat), "right": (right_pos, right_quat)}

    def build_agent_pos(
        self,
        sdk2_q: np.ndarray,
        robot_pos_world: np.ndarray,
        robot_quat_wxyz_world: np.ndarray,
    ) -> np.ndarray:
        """
        Build the 28D agent_pos vector matching simulation layout:
          right_eef_pos  (3)  — right wrist position in pelvis frame
          right_eef_quat (4)  — right wrist quat [w,x,y,z] in pelvis frame
          left_eef_pos   (3)  — left wrist position in pelvis frame
          left_eef_quat  (4)  — left wrist quat [w,x,y,z] in pelvis frame
          body_eef_pos   (3)  — same as right_eef_pos (matches sim behaviour)
          body_eef_quat  (4)  — same as right_eef_quat
          robot_pos      (3)  — robot base position in world frame (SDK2 estimator)
          robot_quat     (4)  — robot base quat [w,x,y,z] in world frame (IMU)

        Args:
            sdk2_q:              joint positions (29,) from SDK2 LowState
            robot_pos_world:     base position (3,) from SDK2 HighState
            robot_quat_wxyz_world: base quat (4,) [w,x,y,z] from SDK2 IMU
        """
        wrist_poses = self.get_wrist_poses_in_pelvis(sdk2_q)
        left_pos,  left_quat  = wrist_poses["left"]
        right_pos, right_quat = wrist_poses["right"]

        agent_pos = np.concatenate([
            right_pos,             # (3)
            right_quat,            # (4)  [w,x,y,z]
            left_pos,              # (3)
            left_quat,             # (4)
            right_pos,             # (3)  body_eef = right_eef (sim convention)
            right_quat,            # (4)
            robot_pos_world,       # (3)
            robot_quat_wxyz_world, # (4)
        ])   # total: 28
        assert agent_pos.shape == (28,), f"agent_pos shape {agent_pos.shape} != (28,)"
        return agent_pos.astype(np.float32)

    # ── Joint limit utilities ─────────────────────────────────────────────────

    def get_arm_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (lower, upper) joint limits for ARM joints only, shape (14,)."""
        lower = np.array([self.model.lowerPositionLimit[self._joint_name_to_cfg_idx[n]]
                          for n in ARM_JOINT_NAMES])
        upper = np.array([self.model.upperPositionLimit[self._joint_name_to_cfg_idx[n]]
                          for n in ARM_JOINT_NAMES])
        return lower, upper

    def clip_arm_joints(self, arm_q_sdk2: np.ndarray) -> np.ndarray:
        """Clip 14D arm joint array to model limits."""
        lower, upper = self.get_arm_joint_limits()
        return np.clip(arm_q_sdk2, lower, upper)
