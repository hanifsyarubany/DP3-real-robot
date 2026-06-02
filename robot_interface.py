"""
robot_interface.py
==================
Unitree G1 robot interface — development mode, full 29-DOF LowCmd.

Architecture (confirmed from real-robot experiments):
  Sport mode  : LocoClient works, but arm LowCmd is blocked. ✗
  Dev mode    : Full LowCmd for all 29 joints, LocoClient disabled. ✓

  → We use DEVELOPMENT MODE exclusively.
  → HOMIE ONNX handles legs (indices 0-14).
  → PINK IK handles arms (indices 15-28).
  → A single LowCmd publishes ALL 29 joint targets at 50 Hz.

Startup sequence (CRITICAL — must follow exactly):
  1. Robot powered on, lying flat.
  2. Send Damp command (zero torque, all motors).
  3. Slowly bring robot to stand using zero-velocity LowCmd with low kp.
  4. Once standing, raise kp to operational values and start policy.

  See DEPLOYMENT_GUIDE.md § "Development Mode Startup" for full procedure.

State reading:
  - LowState  (rt/lowstate):      joint pos, vel, IMU quat, gyro
  - SportModeState (rt/sportmodestate): base XY position estimate
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Unitree SDK2 imports ───────────────────────────────────────────────────────
try:
    from unitree_sdk2py.core.channel import (
        ChannelFactoryInitialize,
        ChannelPublisher,
        ChannelSubscriber,
    )
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
    from unitree_sdk2py.utils.crc import CRC
    SDK2_AVAILABLE = True
except ImportError:
    SDK2_AVAILABLE = False
    print("[robot_interface] WARNING: unitree_sdk2py not found — DRY-RUN mode")

# ── Joint index constants ──────────────────────────────────────────────────────
G1_NUM_MOTORS     = 29
LOWER_BODY_INDICES = list(range(0, 15))   # legs (0-11) + waist (12-14) — HOMIE controls
ARM_INDICES        = list(range(15, 29))  # left arm (15-21) + right arm (22-28) — PINK IK

# PD gains ─────────────────────────────────────────────────────────────────────
# Lower body (HOMIE targets)
LEG_KP   = 150.0   # N·m/rad  — stiff enough for stable standing
LEG_KD   = 4.0     # N·m·s/rad
WAIST_KP = 200.0
WAIST_KD = 5.0

# Arms (PINK IK targets)
ARM_KP_DEFAULT = 80.0   # reduce to 20 for first tests via --low-kp
ARM_KD_DEFAULT = 2.0

# Joint limits (radians) — G1 hardware spec ────────────────────────────────────
# [left_leg(6), right_leg(6), waist(3), left_arm(7), right_arm(7)]
_Q_MIN = np.array([
    -0.6, -0.4, -1.57,  -0.1, -0.87, -0.26,   # left leg
    -0.6, -0.87, -1.57,  -0.1, -0.87, -0.26,   # right leg
    -2.6, -0.52, -0.52,                         # waist
    -3.14, -0.34, -3.14, -0.04, -3.14, -1.57, -3.14,  # left arm
    -3.14, -2.61, -3.14, -0.04, -3.14, -1.57, -3.14,  # right arm
], dtype=np.float32)

_Q_MAX = np.array([
    0.6, 0.87, 1.57,  2.87, 0.52, 0.26,       # left leg
    0.6, 0.4,  1.57,  2.87, 0.52, 0.26,        # right leg
    2.6, 0.52, 0.52,                            # waist
    3.14, 2.61, 3.14, 2.87, 3.14, 1.57, 3.14,  # left arm
    3.14, 0.34, 3.14, 2.87, 3.14, 1.57, 3.14,  # right arm
], dtype=np.float32)


@dataclass
class RobotState:
    """Latest robot state snapshot — updated by SDK2 callbacks."""
    q:    np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))
    dq:   np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))
    tau:  np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))

    imu_quat:  np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.], dtype=np.float32))
    imu_rpy:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_gyro:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    base_pos:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.], dtype=np.float32))

    low_state_ts:  float = 0.0
    high_state_ts: float = 0.0


class G1RobotInterface:
    """
    Thread-safe G1 interface for development mode (full 29-DOF LowCmd).

    Args:
        network_interface: Ethernet interface name (e.g. "enp2s0", "eth0")
        dry_run: If True, commands are computed but NOT sent to robot.
    """

    def __init__(self, network_interface: str, dry_run: bool = False):
        self.network_interface = network_interface
        self.dry_run = dry_run or (not SDK2_AVAILABLE)

        self._state      = RobotState()
        self._state_lock = threading.Lock()
        self._connected  = False

        if SDK2_AVAILABLE:
            self._crc     = CRC()
            self._low_cmd = unitree_hg_msg_dds__LowCmd_()
            self._init_low_cmd_safe()
        else:
            self._crc     = None
            self._low_cmd = None

        self._pub: Optional[object] = None

    def _init_low_cmd_safe(self):
        """Set all motor commands to a safe zero-torque default."""
        for i in range(G1_NUM_MOTORS):
            self._low_cmd.motor_cmd[i].mode = 0x01   # FOC
            self._low_cmd.motor_cmd[i].q    = 0.0
            self._low_cmd.motor_cmd[i].dq   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0
            self._low_cmd.motor_cmd[i].kp   = 0.0    # passive until connect()
            self._low_cmd.motor_cmd[i].kd   = 0.0

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise SDK2 channels and wait for first state message."""
        if self.dry_run:
            print("[RobotInterface] DRY-RUN — no SDK2 connection")
            return

        print(f"[RobotInterface] Connecting on '{self.network_interface}' ...")
        ChannelFactoryInitialize(0, self.network_interface)

        # LowState subscriber
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._lowstate_cb, 10)

        # SportModeState for base XY position
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            self._highstate_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self._highstate_sub.Init(self._highstate_cb, 10)
        except Exception as e:
            print(f"[RobotInterface] SportModeState subscriber skipped: {e}")

        # LowCmd publisher (development mode — controls all 29 joints)
        self._pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._pub.Init()

        # Wait for first LowState
        print("[RobotInterface] Waiting for LowState ...")
        t0 = time.time()
        while self._state.low_state_ts == 0.0:
            time.sleep(0.02)
            if time.time() - t0 > 5.0:
                raise TimeoutError(
                    "No LowState in 5 s — check Ethernet cable and robot power."
                )

        self._connected = True
        print("[RobotInterface] Connected.")

    # ── State callbacks ────────────────────────────────────────────────────────

    def _lowstate_cb(self, msg) -> None:
        with self._state_lock:
            for i in range(G1_NUM_MOTORS):
                self._state.q[i]   = msg.motor_state[i].q
                self._state.dq[i]  = msg.motor_state[i].dq
                self._state.tau[i] = msg.motor_state[i].tau_est
            imu = msg.imu_state
            self._state.imu_quat = np.array(imu.quaternion, dtype=np.float32)
            self._state.imu_rpy  = np.array(imu.rpy,        dtype=np.float32)
            self._state.imu_gyro = np.array(imu.gyroscope,  dtype=np.float32)
            self._state.base_quat = self._state.imu_quat.copy()
            self._state.low_state_ts = time.time()

    def _highstate_cb(self, msg) -> None:
        with self._state_lock:
            self._state.base_pos = np.array(msg.position, dtype=np.float32)[:3]
            self._state.high_state_ts = time.time()

    # ── State reads ────────────────────────────────────────────────────────────

    def get_state(self) -> RobotState:
        with self._state_lock:
            s = RobotState(
                q=self._state.q.copy(), dq=self._state.dq.copy(),
                tau=self._state.tau.copy(),
                imu_quat=self._state.imu_quat.copy(),
                imu_rpy=self._state.imu_rpy.copy(),
                imu_gyro=self._state.imu_gyro.copy(),
                base_pos=self._state.base_pos.copy(),
                base_quat=self._state.base_quat.copy(),
                low_state_ts=self._state.low_state_ts,
                high_state_ts=self._state.high_state_ts,
            )
        return s

    def get_arm_joints(self) -> np.ndarray:
        """Return current arm joint angles (14,) — SDK2 indices 15-28."""
        with self._state_lock:
            return self._state.q[15:29].copy()

    # ── Command sending ────────────────────────────────────────────────────────

    def send_full_body(
        self,
        q_lower: np.ndarray,    # (15,) HOMIE output — legs + waist
        q_arms:  np.ndarray,    # (14,) PINK IK output — left + right arm
        arm_kp:  float = ARM_KP_DEFAULT,
        arm_kd:  float = ARM_KD_DEFAULT,
    ) -> None:
        """
        Publish one full 29-DOF LowCmd (development mode).

        q_lower: HOMIE joint position targets for indices 0-14
        q_arms:  PINK IK joint position targets for indices 15-28
        """
        q_full = np.concatenate([q_lower, q_arms]).astype(np.float32)
        q_full = np.clip(q_full, _Q_MIN, _Q_MAX)

        if self.dry_run:
            return

        for i in LOWER_BODY_INDICES:
            kp = WAIST_KP if i >= 12 else LEG_KP
            kd = WAIST_KD if i >= 12 else LEG_KD
            self._low_cmd.motor_cmd[i].mode = 0x01
            self._low_cmd.motor_cmd[i].q    = float(q_full[i])
            self._low_cmd.motor_cmd[i].dq   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0
            self._low_cmd.motor_cmd[i].kp   = kp
            self._low_cmd.motor_cmd[i].kd   = kd

        for local_i, global_i in enumerate(ARM_INDICES):
            self._low_cmd.motor_cmd[global_i].mode = 0x01
            self._low_cmd.motor_cmd[global_i].q    = float(q_full[global_i])
            self._low_cmd.motor_cmd[global_i].dq   = 0.0
            self._low_cmd.motor_cmd[global_i].tau  = 0.0
            self._low_cmd.motor_cmd[global_i].kp   = arm_kp
            self._low_cmd.motor_cmd[global_i].kd   = arm_kd

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)

    # ── Startup helpers ────────────────────────────────────────────────────────

    def damp(self) -> None:
        """Zero-torque on all joints (safe resting state)."""
        if self.dry_run:
            return
        for i in range(G1_NUM_MOTORS):
            self._low_cmd.motor_cmd[i].mode = 0x01
            self._low_cmd.motor_cmd[i].q    = 0.0
            self._low_cmd.motor_cmd[i].dq   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0
            self._low_cmd.motor_cmd[i].kp   = 0.0
            self._low_cmd.motor_cmd[i].kd   = 2.0   # small damping
        self._low_cmd.crc = self._crc.Crc(self._low_cmd)
        self._pub.Write(self._low_cmd)

    def stand_up(self, duration_s: float = 3.0) -> None:
        """
        Bring robot from damped state to standing using a linear interpolation
        from current joint angles to the HOMIE default standing pose.

        ONLY call after robot is lying flat and in damped state.
        One person must be ready at the emergency stop.
        """
        if self.dry_run:
            print("[RobotInterface][DRY-RUN] stand_up() skipped")
            return

        from wbc.homie_loco import N_LOWER_JOINTS
        import yaml, pathlib
        cfg_path = pathlib.Path(__file__).parent / "config" / "g1_homie_v2.yaml"
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        stand_q = np.array(cfg["default_angles"], dtype=np.float32)  # (15,)
        arm_q   = np.zeros(14, dtype=np.float32)

        # Read current pose
        state = self.get_state()
        q0_lower = state.q[:15].copy()
        q0_arms  = state.q[15:].copy()

        dt    = 0.02
        steps = int(duration_s / dt)
        print(f"[RobotInterface] Standing up over {duration_s:.1f} s ...")

        for step in range(steps):
            alpha    = (step + 1) / steps
            q_lower  = (1 - alpha) * q0_lower + alpha * stand_q
            q_arms_t = (1 - alpha) * q0_arms  + alpha * arm_q
            self.send_full_body(
                q_lower, q_arms_t,
                arm_kp=20.0, arm_kd=1.0,   # low gain during standup
            )
            time.sleep(dt)

        print("[RobotInterface] Standing.")

    # ── Emergency stop ─────────────────────────────────────────────────────────

    def emergency_stop(self) -> None:
        """Zero all torques immediately."""
        print("[RobotInterface] *** EMERGENCY STOP ***")
        if not self.dry_run:
            self.damp()
