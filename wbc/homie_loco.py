"""
wbc/homie_loco.py — Standalone HOMIE v2 locomotion policy for real robot.

No Isaac Lab dependency. Mirrors G1HomiePolicyV2 from isaaclab_arena_g1 but
is self-contained for companion PC deployment.

Architecture:
  Input:  516-dim observation vector (86-dim single obs × 6 history frames)
  Output: 15 lower-body joint position TARGETS (SDK2 indices 0–14: legs + waist)

Single obs (86-dim) layout:
  [0:3]   cmd   × cmd_scale       — [vx, vy, vyaw] × [2.0, 2.0, 0.5]
  [3:4]   height_cmd              — base height setpoint (m)
  [4:7]   [roll_cmd, pitch, yaw]  — torso orientation commands (rad)
  [7:10]  omega × ang_vel_scale   — IMU angular velocity × 0.5
  [10:13] gravity_orientation     — gravity vector in body frame (3D)
  [13:42] qj_scaled               — all 29 joint positions (scaled)
  [42:71] dqj_scaled              — all 29 joint velocities (scaled)
  [71:86] prev_action             — previous 15-joint output action

Policy selection:
  |cmd| < 0.05 m/s  → stand.onnx  (stationary balancing)
  |cmd| >= 0.05 m/s → walk.onnx   (walking)
"""

from __future__ import annotations

import collections
import pathlib
import yaml

import numpy as np
import onnxruntime as ort

# ── Constants matching g1_homie_v2.yaml and G1HomiePolicyV2 ──────────────────
N_LOWER_JOINTS = 15   # legs (12) + waist (3) — HOMIE output
N_BODY_JOINTS  = 29   # all body joints used in observation
SINGLE_OBS_DIM = 86   # per-frame observation dim  (verified: 3+1+3+3+3+29+29+15)
OBS_HISTORY    = 6    # frames of history stacked
OBS_DIM        = SINGLE_OBS_DIM * OBS_HISTORY   # 516 — ONNX input dim

# SDK2 joint indices for lower-body joints (HOMIE output order matches SDK2)
LOWER_BODY_SDK2_INDICES = list(range(0, 15))   # 0-14: legs + waist

# Default config path (relative to this file's parent)
_CONFIG_PATH = pathlib.Path(__file__).resolve().parent.parent / "config" / "g1_homie_v2.yaml"


def _get_gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
    """Project gravity vector [0,0,-1] into body frame using IMU quaternion [w,x,y,z]."""
    w, x, y, z = quat_wxyz.astype(float)
    # Rotate [0,0,-1] by inverse of quaternion q
    # = q_conj * [0,0,-1] * q  (simplified to rotation matrix row 3)
    gx = 2.0 * (x * z - w * y)
    gy = 2.0 * (y * z + w * x)
    gz = w*w - x*x - y*y + z*z
    return np.array([-gx, -gy, -gz], dtype=np.float32)


def _load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key in ("default_angles", "cmd_scale", "cmd_init"):
        if key in cfg:
            cfg[key] = np.array(cfg[key], dtype=np.float32)
    return cfg


class HomieLocoPolicy:
    """
    HOMIE v2 locomotion policy — standalone, no Isaac dependency.

    Handles observation history, gait clock, policy selection, and action
    output entirely in numpy/onnxruntime.

    Args:
        stand_onnx: path to stand.onnx
        walk_onnx:  path to walk.onnx
        config_path: path to g1_homie_v2.yaml (defaults to real-robot-deployment/config/)
    """

    def __init__(
        self,
        stand_onnx: str,
        walk_onnx: str,
        config_path: str | None = None,
    ):
        cfg_path = config_path or str(_CONFIG_PATH)
        self.cfg = _load_config(cfg_path)

        self._stand = self._load_onnx(stand_onnx)
        self._walk  = self._load_onnx(walk_onnx)

        # Observation history buffer
        self._obs_history: collections.deque = collections.deque(maxlen=OBS_HISTORY)
        self._obs_buf = np.zeros((1, OBS_DIM), dtype=np.float32)

        # State
        self._action = np.zeros((1, N_LOWER_JOINTS), dtype=np.float32)
        self._gait_phase = np.zeros((1, 1), dtype=np.float32)

        # Goals (set via set_goal)
        self._cmd        = np.zeros((1, 3), dtype=np.float32)  # [vx, vy, vyaw]
        self._height_cmd = float(self.cfg.get("height_cmd", 0.74))
        self._roll_cmd   = np.zeros((1,), dtype=np.float32)
        self._pitch_cmd  = np.zeros((1,), dtype=np.float32)
        self._yaw_cmd    = np.zeros((1,), dtype=np.float32)
        self._freq_cmd   = float(self.cfg.get("freq_cmd", 0.75))

        # Padded default angles for all 29 joints (HOMIE yaml has 15, pad with 0 for arms)
        d = self.cfg["default_angles"]
        self._default_angles = np.zeros(N_BODY_JOINTS, dtype=np.float32)
        self._default_angles[:len(d)] = d

        print(f"[HomieLocoPolicy] Loaded stand={stand_onnx}")
        print(f"[HomieLocoPolicy]        walk={walk_onnx}")

    # ── ONNX loading ──────────────────────────────────────────────────────────

    def _load_onnx(self, path: str):
        sess = ort.InferenceSession(path)
        input_name = sess.get_inputs()[0].name

        def run(obs: np.ndarray) -> np.ndarray:
            return sess.run(None, {input_name: obs})[0]

        return run

    # ── Goal interface (maps EquivDP3 action → locomotion commands) ───────────

    def set_goal(
        self,
        vx: float, vy: float, vyaw: float,
        height_cmd: float | None = None,
        torso_rpy: np.ndarray | None = None,
    ) -> None:
        """
        Set locomotion commands from EquivDP3 action slice c_t[16:23].

        Args:
            vx, vy, vyaw: navigation velocity commands (m/s, rad/s)
            height_cmd:   base height setpoint (m), default 0.74
            torso_rpy:    (3,) torso orientation [roll, pitch, yaw] (rad)
        """
        self._cmd[0, 0] = vx
        self._cmd[0, 1] = vy
        self._cmd[0, 2] = vyaw
        if height_cmd is not None:
            self._height_cmd = float(height_cmd)
        if torso_rpy is not None:
            self._roll_cmd[0]  = torso_rpy[0]
            self._pitch_cmd[0] = torso_rpy[1]
            self._yaw_cmd[0]   = torso_rpy[2]

    # ── Observation build ─────────────────────────────────────────────────────

    def _build_single_obs(
        self,
        q_all: np.ndarray,    # (29,) all joint positions from SDK2
        dq_all: np.ndarray,   # (29,) all joint velocities
        imu_quat_wxyz: np.ndarray,  # (4,) [w,x,y,z]
        imu_gyro: np.ndarray,       # (3,) angular velocity rad/s
    ) -> np.ndarray:
        """Build one 86-dim observation frame."""
        cfg = self.cfg

        # Advance gait clock
        self._gait_phase = np.remainder(
            self._gait_phase + 0.02 * self._freq_cmd, 1.0
        )

        # Scaled observations
        qj_scaled  = (q_all  - self._default_angles) * float(cfg["dof_pos_scale"])
        dqj_scaled = dq_all  * float(cfg["dof_vel_scale"])
        omega_scaled = imu_gyro * float(cfg["ang_vel_scale"])
        gravity     = _get_gravity_orientation(imu_quat_wxyz)     # (3,)
        cmd_scaled  = self._cmd[0] * cfg["cmd_scale"]             # (3,)

        obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        obs[0:3]   = cmd_scaled
        obs[3]     = self._height_cmd
        obs[4:7]   = [self._roll_cmd[0], self._pitch_cmd[0], self._yaw_cmd[0]]
        obs[7:10]  = omega_scaled
        obs[10:13] = gravity
        obs[13:42] = qj_scaled    # all 29 joints
        obs[42:71] = dqj_scaled   # all 29 joints
        obs[71:86] = self._action[0]  # previous 15-joint action
        return obs

    def _update_obs_buffer(self, single_obs: np.ndarray) -> None:
        """Push single obs into history and rebuild flat 516-dim buffer."""
        self._obs_history.append(single_obs.copy())
        # Pad history with zeros if not full yet
        while len(self._obs_history) < OBS_HISTORY:
            self._obs_history.appendleft(np.zeros(SINGLE_OBS_DIM, dtype=np.float32))
        for i, frame in enumerate(self._obs_history):
            self._obs_buf[0, i * SINGLE_OBS_DIM:(i + 1) * SINGLE_OBS_DIM] = frame

    # ── Inference ─────────────────────────────────────────────────────────────

    def get_action(
        self,
        q_all: np.ndarray,
        dq_all: np.ndarray,
        imu_quat_wxyz: np.ndarray,
        imu_gyro: np.ndarray,
    ) -> np.ndarray:
        """
        Run one HOMIE inference step.

        Args:
            q_all:          (29,) all joint positions (SDK2 order)
            dq_all:         (29,) all joint velocities
            imu_quat_wxyz:  (4,)  IMU quaternion [w,x,y,z]
            imu_gyro:       (3,)  IMU angular velocity (rad/s)

        Returns:
            q_lower_target: (15,) lower-body joint position targets
                            (SDK2 indices 0-14, legs + waist)
        """
        single_obs = self._build_single_obs(q_all, dq_all, imu_quat_wxyz, imu_gyro)
        self._update_obs_buffer(single_obs)

        # Policy selection: stand if near-zero command, else walk
        cmd_norm = float(np.linalg.norm(self._cmd[0]))
        policy = self._stand if cmd_norm < 0.05 else self._walk

        raw_action = policy(self._obs_buf)   # (1, 15)
        self._action = raw_action

        # Convert action → joint position targets
        cfg = self.cfg
        d   = self._default_angles[:N_LOWER_JOINTS]
        q_target = raw_action[0] * float(cfg["action_scale"]) + d  # (15,)
        return q_target.astype(np.float32)

    def reset(self) -> None:
        """Reset history and action state (call at episode start)."""
        self._obs_history.clear()
        self._obs_buf[:] = 0.0
        self._action[:]  = 0.0
        self._gait_phase[:] = 0.0
