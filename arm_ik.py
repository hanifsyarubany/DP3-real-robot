"""
arm_ik.py
=========
PINK inverse kinematics for the Unitree G1 arms.

The diffusion policy outputs wrist EEF targets in the robot pelvis frame:
    c_t[2:9]  = left  wrist [pos(3) + quat(4)]  in pelvis frame
    c_t[9:16] = right wrist [pos(3) + quat(4)]  in pelvis frame

This module solves IK and returns the 14 arm joint angles (indices 15-28)
that achieve those targets.

Joint layout (matches SDK2 indices):
    Left arm  (7 joints, SDK2 idx 15-21):
        shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
        wrist_roll, wrist_pitch, wrist_yaw
    Right arm (7 joints, SDK2 idx 22-28): same order

URDF used: g1_custom_collision_29dof.urdf (pelvis as root, no world joint)
EEF links: left_wrist_yaw_link, right_wrist_yaw_link

Quaternion convention: [w, x, y, z] (matching Isaac Sim and the dataset).
"""

from __future__ import annotations

import pathlib
from typing import Optional

import numpy as np
import pinocchio as pin

# Default URDF path — configure in config/deploy_config.yaml → wbc.urdf_path
_DEFAULT_URDF_PATH = pathlib.Path(
    "/workspaces/isaaclab_arena/submodules/IsaacLab/source/TWIST2/assets/g1/g1_custom_collision_29dof.urdf"
)

# Joint names in pinocchio order — must match the URDF joint ordering.
# These are the 14 controllable arm joints (7 per side).
LEFT_ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINT_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]
ARM_JOINT_NAMES = LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES

# EEF frame names — must match sim: action_constants.py LEFT/RIGHT_WRIST_LINK_NAME
# and g1.py get_target_link_position_in_target_frame(target_link_name=...)
LEFT_EEF_LINK  = "left_wrist_yaw_link"
RIGHT_EEF_LINK = "right_wrist_yaw_link"


def _quat_wxyz_to_matrix(q_wxyz: np.ndarray) -> np.ndarray:
    """Convert [w, x, y, z] quaternion to 3×3 rotation matrix."""
    w, x, y, z = q_wxyz.astype(float)
    return np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ])


class ArmIK:
    """
    Differential IK for G1 arms using pinocchio + PINK.

    Parameters
    ----------
    urdf_path : str or Path, optional
        Path to G1 URDF. Configure in config/deploy_config.yaml → wbc.urdf_path.
    dt : float
        IK integration timestep (seconds).
    n_steps : int
        Number of IK solver iterations per call.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        dt: float = 0.002,
        n_steps: int = 10,
    ):
        self.dt      = dt
        self.n_steps = n_steps

        urdf = str(urdf_path or _DEFAULT_URDF_PATH)
        if not pathlib.Path(urdf).exists():
            raise FileNotFoundError(f"G1 URDF not found at: {urdf}")

        # Kinematic model only — no mesh files needed for IK.
        # buildModelsFromUrdf also loads visual meshes (→ FileNotFoundError on STL).
        self.model = pin.buildModelFromUrdf(urdf)
        self.data  = self.model.createData()

        # Current joint configuration (full model, nq dims)
        self.q_current = pin.neutral(self.model)

        # Map arm joint names → pinocchio joint indices
        self._arm_joint_ids = []
        for name in ARM_JOINT_NAMES:
            if self.model.existJointName(name):
                jid = self.model.getJointId(name)
                self._arm_joint_ids.append(jid)
            else:
                raise ValueError(f"Joint '{name}' not found in URDF. Check joint names.")

        # Map EEF link names → pinocchio frame indices
        self._left_eef_frame  = self._get_frame_id(LEFT_EEF_LINK)
        self._right_eef_frame = self._get_frame_id(RIGHT_EEF_LINK)

        # PINK task objects (lazy init in solve())
        self._pink_tasks_ready = False
        self._init_pink()

        print(f"[ArmIK] Loaded G1 URDF ({self.model.nq} DoF)")
        print(f"[ArmIK]   Left  EEF: '{LEFT_EEF_LINK}' (frame {self._left_eef_frame})")
        print(f"[ArmIK]   Right EEF: '{RIGHT_EEF_LINK}' (frame {self._right_eef_frame})")

    def _get_frame_id(self, link_name: str) -> int:
        if not self.model.existFrame(link_name):
            raise ValueError(f"Frame '{link_name}' not in URDF model.")
        return self.model.getFrameId(link_name)

    def _init_pink(self) -> None:
        """Initialise PINK tasks for left and right EEF."""
        try:
            import pink
            from pink.tasks import FrameTask

            self._left_task = FrameTask(
                LEFT_EEF_LINK,
                position_cost=1.0,
                orientation_cost=0.5,
            )
            self._right_task = FrameTask(
                RIGHT_EEF_LINK,
                position_cost=1.0,
                orientation_cost=0.5,
            )
            self._pink = pink
            self._pink_tasks_ready = True

        except ImportError:
            print("[ArmIK] Warning: PINK not installed. Falling back to Jacobian pseudo-inverse IK.")
            print("        Install: pip install pin-pink")
            self._pink_tasks_ready = False

    def update_joint_angles(self, q_arm: np.ndarray) -> None:
        """
        Update the current arm joint configuration from real robot state.

        Call this every control step with the latest joint readings so IK
        warm-starts from the current pose.

        Parameters
        ----------
        q_arm : np.ndarray, shape (14,)
            Current arm joint angles [left(7), right(7)] in radians.
        """
        assert q_arm.shape == (14,)
        for local_i, jid in enumerate(self._arm_joint_ids):
            idx = self.model.joints[jid].idx_q
            self.q_current[idx] = q_arm[local_i]

    def solve(
        self,
        left_pos_pelvis:  np.ndarray,
        left_quat_pelvis: np.ndarray,
        right_pos_pelvis:  np.ndarray,
        right_quat_pelvis: np.ndarray,
    ) -> np.ndarray:
        """
        Solve IK for both arms simultaneously.

        Targets are expressed in the pelvis frame (matching the WBC action space).

        Parameters
        ----------
        left_pos_pelvis  : (3,)  left wrist target position in pelvis frame
        left_quat_pelvis : (4,)  left wrist target orientation [w,x,y,z] in pelvis frame
        right_pos_pelvis : (3,)  right wrist target position in pelvis frame
        right_quat_pelvis: (4,)  right wrist target orientation [w,x,y,z] in pelvis frame

        Returns
        -------
        q_arm : np.ndarray, shape (14,)
            Joint angle targets [left(7), right(7)] in radians, clamped to limits.
        """
        R_left  = _quat_wxyz_to_matrix(left_quat_pelvis)
        R_right = _quat_wxyz_to_matrix(right_quat_pelvis)

        T_left  = pin.SE3(R_left,  left_pos_pelvis.astype(float))
        T_right = pin.SE3(R_right, right_pos_pelvis.astype(float))

        if self._pink_tasks_ready:
            return self._solve_pink(T_left, T_right)
        else:
            return self._solve_jacobian(T_left, T_right)

    def _solve_pink(self, T_left: pin.SE3, T_right: pin.SE3) -> np.ndarray:
        """PINK differential IK solver (preferred)."""
        from pink import solve_ik
        from pink.tasks import FrameTask

        q = self.q_current.copy()

        self._left_task.set_target(T_left)
        self._right_task.set_target(T_right)
        tasks = [self._left_task, self._right_task]

        for _ in range(self.n_steps):
            pin.framesForwardKinematics(self.model, self.data, q)

            velocity = solve_ik(
                self._pink.Configuration(self.model, self.data, q),
                tasks,
                self.dt,
                solver="quadprog",
            )
            q = pin.integrate(self.model, q, velocity * self.dt)
            q = pin.normalize(self.model, q)

        # Extract arm joints
        self.q_current = q
        return self._extract_arm_joints(q)

    def _solve_jacobian(self, T_left: pin.SE3, T_right: pin.SE3) -> np.ndarray:
        """Fallback: Jacobian pseudo-inverse IK (no PINK dependency)."""
        q = self.q_current.copy()
        damp = 1e-6

        for _ in range(self.n_steps):
            pin.framesForwardKinematics(self.model, self.data, q)
            pin.computeJointJacobians(self.model, self.data, q)

            err_left  = pin.log6(
                self.data.oMf[self._left_eef_frame].actInv(T_left)
            ).vector
            err_right = pin.log6(
                self.data.oMf[self._right_eef_frame].actInv(T_right)
            ).vector

            J_left  = pin.getFrameJacobian(
                self.model, self.data, self._left_eef_frame, pin.LOCAL
            )
            J_right = pin.getFrameJacobian(
                self.model, self.data, self._right_eef_frame, pin.LOCAL
            )

            J  = np.vstack([J_left, J_right])
            e  = np.concatenate([err_left, err_right])
            dq = J.T @ np.linalg.solve(J @ J.T + damp * np.eye(12), e)
            q  = pin.integrate(self.model, q, dq * self.dt)
            q  = pin.normalize(self.model, q)

        self.q_current = q
        return self._extract_arm_joints(q)

    def _extract_arm_joints(self, q: np.ndarray) -> np.ndarray:
        """Extract the 14 arm joint values from the full configuration vector."""
        arm_q = np.zeros(14, dtype=np.float32)
        for local_i, jid in enumerate(self._arm_joint_ids):
            idx = self.model.joints[jid].idx_q
            arm_q[local_i] = q[idx]
        return arm_q

    def forward_kinematics(self, q_arm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute forward kinematics for both EEFs.

        Parameters
        ----------
        q_arm : (14,) current arm joint angles

        Returns
        -------
        left_pos  : (3,)  left EEF position in pelvis frame
        left_quat : (4,)  left EEF orientation [w,x,y,z] in pelvis frame
        right_pos : (3,)  right EEF position in pelvis frame
        right_quat: (4,)  right EEF orientation [w,x,y,z] in pelvis frame
        """
        self.update_joint_angles(q_arm)
        pin.framesForwardKinematics(self.model, self.data, self.q_current)

        T_left  = self.data.oMf[self._left_eef_frame]
        T_right = self.data.oMf[self._right_eef_frame]

        def _mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
            q_xyzw = pin.Quaternion(R).coeffs()  # pinocchio returns [x,y,z,w]
            return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float32)

        return (
            T_left.translation.astype(np.float32),
            _mat_to_quat_wxyz(T_left.rotation),
            T_right.translation.astype(np.float32),
            _mat_to_quat_wxyz(T_right.rotation),
        )
