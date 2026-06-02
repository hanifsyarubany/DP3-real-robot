"""
obs_builder.py
==============
Build the observation dict that EquivDP3.predict_action() expects, from
real robot sensor data.  The format must match the training data exactly.

Observation space (from checkpoint shape_meta):
    point_cloud : (1, n_obs_steps, 1024, 3)   pelvis-frame XYZ
    agent_pos   : (1, n_obs_steps, 28)         world-frame proprioception

agent_pos layout (28D) — EEF in PELVIS frame, robot pose in WORLD frame:
    [0:3]   right_eef_pos   right wrist_yaw_link position   in pelvis frame
    [3:7]   right_eef_quat  right wrist_yaw_link orientation [w,x,y,z] in pelvis frame
    [7:10]  left_eef_pos    left  wrist_yaw_link position   in pelvis frame
    [10:14] left_eef_quat   left  wrist_yaw_link orientation [w,x,y,z] in pelvis frame
    [14:17] body_eef_pos    = copy of right_eef_pos  (sim convention, g1.py line 482-488)
    [17:21] body_eef_quat   = copy of right_eef_quat
    [21:24] robot_pos       pelvis/base position  in world frame  (SDK2 state estimator)
    [24:28] robot_quat      pelvis/base orientation [w,x,y,z] in world frame (IMU)

EEF positions come from pinocchio FK with a FIXED (non-floating) base, so the
pelvis is the root and all frame positions are naturally in the pelvis frame.
No world-frame transformation is needed or correct.
"""

from __future__ import annotations

import collections
import pathlib
from typing import Optional

import numpy as np
import pinocchio as pin
import torch

from robot_interface import RobotState

# ── URDF & FK setup ────────────────────────────────────────────────────────────
# Default path — override via urdf_path argument to ObservationBuilder().
# Configure in config/deploy_config.yaml  →  wbc.urdf_path
_DEFAULT_URDF_PATH = pathlib.Path(
    "/workspaces/isaaclab_arena/submodules/IsaacLab/source/TWIST2/assets/g1/g1_custom_collision_29dof.urdf"
)

_LEFT_EEF_LINK  = "left_wrist_yaw_link"   # matches sim: get_target_link_position_in_target_frame
_RIGHT_EEF_LINK = "right_wrist_yaw_link"  # matches action_constants.py LEFT/RIGHT_WRIST_LINK_NAME
_BASE_LINK      = "pelvis"

N_OBS_STEPS = 2
N_PTS       = 1024
AGENT_POS_DIM = 28


def _quat_wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    """[w,x,y,z] → 3×3 rotation matrix."""
    w, x, y, z = q.astype(float)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)],
    ])


def _pin_quat_to_wxyz(q: pin.Quaternion) -> np.ndarray:
    """Pinocchio Quaternion [x,y,z,w] → [w,x,y,z]."""
    c = q.coeffs()  # [x,y,z,w]
    return np.array([c[3], c[0], c[1], c[2]], dtype=np.float32)


class ObservationBuilder:
    """
    Converts raw robot state + point cloud into the EquivDP3 observation dict.

    Parameters
    ----------
    n_obs_steps : int
        Number of observation frames to stack (must be 2, matching training).
    n_pts : int
        Points per cloud frame (must be 1024, matching training).
    device : str
        PyTorch device for the returned tensors.
    urdf_path : optional path override for the G1 URDF.
    """

    def __init__(
        self,
        n_obs_steps: int = N_OBS_STEPS,
        n_pts: int = N_PTS,
        device: str = "cuda",
        urdf_path: Optional[str] = None,
    ):
        self.n_obs_steps = n_obs_steps
        self.n_pts       = n_pts
        self.device      = device

        # Rolling history deque
        self._history: collections.deque = collections.deque(maxlen=n_obs_steps)

        # Pinocchio model for FK
        urdf = str(urdf_path or _DEFAULT_URDF_PATH)
        if not pathlib.Path(urdf).exists():
            raise FileNotFoundError(f"G1 URDF not found: {urdf}")

        # Kinematic model only — no mesh files needed for FK/IK.
        # buildModelsFromUrdf also loads visual meshes (→ FileNotFoundError on STL).
        self._model = pin.buildModelFromUrdf(urdf)
        self._data  = self._model.createData()

        self._left_frame  = self._model.getFrameId(_LEFT_EEF_LINK)
        self._right_frame = self._model.getFrameId(_RIGHT_EEF_LINK)
        # body_eef is NOT a separate frame — it's a copy of right_eef (sim convention)

        print(f"[ObsBuilder] Loaded G1 URDF for FK ({self._model.nq} DoF)")

    # ── FK helper ──────────────────────────────────────────────────────────────

    def _compute_fk(self, q_full: np.ndarray) -> dict[str, np.ndarray]:
        """
        Run pinocchio FK and return EEF poses in the PELVIS frame.

        The URDF has pelvis as root with no floating base joint, so all
        frame translations are natively expressed in the pelvis frame —
        exactly matching get_target_link_position_in_target_frame(target_frame="pelvis")
        used during training.
        """
        q_pin = pin.neutral(self._model)
        for sdk_idx in range(min(29, self._model.nq)):
            q_pin[sdk_idx] = q_full[sdk_idx]

        pin.framesForwardKinematics(self._model, self._data, q_pin)

        def _frame_pose_wxyz(frame_id):
            oMf  = self._data.oMf[frame_id]
            pos  = oMf.translation.astype(np.float32)
            quat = _pin_quat_to_wxyz(pin.Quaternion(oMf.rotation))
            return pos, quat

        left_pos,  left_quat  = _frame_pose_wxyz(self._left_frame)
        right_pos, right_quat = _frame_pose_wxyz(self._right_frame)

        return {
            "left_pos":  left_pos,   "left_quat":  left_quat,
            "right_pos": right_pos,  "right_quat": right_quat,
        }

    # ── Public API ─────────────────────────────────────────────────────────────

    def build_agent_pos(self, state: RobotState) -> np.ndarray:
        """
        Compute 28D agent_pos from RobotState.

        All EEF poses are:
          1. Computed via pinocchio FK (in pelvis frame)
          2. Transformed to world frame using state.base_pos / state.base_quat

        Parameters
        ----------
        state : RobotState  (from G1RobotInterface.get_state())

        Returns
        -------
        np.ndarray, shape (28,), dtype float32
        """
        # FK → wrist poses in pelvis frame (no world-frame transform needed)
        fk = self._compute_fk(state.q)
        r_pos,  r_quat  = fk["right_pos"], fk["right_quat"]
        l_pos,  l_quat  = fk["left_pos"],  fk["left_quat"]

        # body_eef is a duplicate of right_eef — sim convention (see g1.py L482-488):
        #   "Body eefs are not used for transforms so values are not important,
        #    but they must be present for datagen to run since 'body' is considered an eef"
        agent_pos = np.concatenate([
            r_pos,  r_quat,        # [0:7]   right wrist_yaw_link in pelvis frame
            l_pos,  l_quat,        # [7:14]  left  wrist_yaw_link in pelvis frame
            r_pos,  r_quat,        # [14:21] body_eef = copy of right_eef
            state.base_pos,        # [21:24] robot base position  in WORLD frame
            state.base_quat,       # [24:28] robot base quaternion in WORLD frame
        ]).astype(np.float32)

        assert agent_pos.shape == (AGENT_POS_DIM,), f"agent_pos shape: {agent_pos.shape}"
        return agent_pos

    def push(self, pcd: np.ndarray, agent_pos: np.ndarray) -> None:
        """
        Add one timestep of observations to the rolling history.

        Parameters
        ----------
        pcd       : (1024, 3) pelvis-frame point cloud
        agent_pos : (28,)  world-frame proprioception
        """
        assert pcd.shape       == (self.n_pts, 3),          f"PCD shape mismatch: {pcd.shape}"
        assert agent_pos.shape == (AGENT_POS_DIM,),         f"AgentPos shape mismatch: {agent_pos.shape}"
        self._history.append((pcd.astype(np.float32), agent_pos.astype(np.float32)))

    def pad_to_full(self) -> None:
        """Repeat the last frame until history is full (use at episode start)."""
        if len(self._history) == 0:
            raise RuntimeError("Call push() at least once before pad_to_full().")
        while len(self._history) < self.n_obs_steps:
            self._history.append(self._history[-1])

    def get_obs_input(self) -> dict[str, torch.Tensor]:
        """
        Build the batched observation dict for policy.predict_action().

        Returns
        -------
        dict with:
            "point_cloud" : torch.FloatTensor, shape (1, n_obs_steps, 1024, 3)
            "agent_pos"   : torch.FloatTensor, shape (1, n_obs_steps, 28)
        """
        if len(self._history) < self.n_obs_steps:
            raise RuntimeError(
                f"History has {len(self._history)} frames but need {self.n_obs_steps}. "
                "Call push() + pad_to_full() first."
            )

        pcds    = np.stack([f[0] for f in self._history], axis=0)  # (T, 1024, 3)
        ap      = np.stack([f[1] for f in self._history], axis=0)  # (T, 28)

        return {
            "point_cloud": torch.from_numpy(pcds).float().unsqueeze(0).to(self.device),   # (1,T,1024,3)
            "agent_pos":   torch.from_numpy(ap).float().unsqueeze(0).to(self.device),      # (1,T,28)
        }

    def clear(self) -> None:
        """Clear history (call at episode reset)."""
        self._history.clear()
