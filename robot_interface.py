"""
robot_interface.py
==================
Unitree G1 robot interface using unitree_sdk2py.

Provides two control channels:
  1. LocoClient  – high-level sport mode locomotion (vx, vy, vyaw, height)
  2. LowCmd      – low-level arm joint position control (indices 15-28)

Sport mode handles legs (indices 0-14) internally.  We take over only the
arm joints via LowCmd.  This is the standard "arm teleoperation" architecture
supported by Unitree's G1 platform.

State reading:
  - LowState  : joint angles, velocities, IMU quaternion/rpy (rt/lowstate)
  - HighState : base position from sport mode state estimator (rt/sportmodestate)

Quaternion convention: [w, x, y, z] throughout (matching Isaac Sim / dataset).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ── Unitree SDK2 imports ───────────────────────────────────────────────────────
from unitree_sdk2py.core.channel import (
    ChannelFactoryInitialize,
    ChannelPublisher,
    ChannelSubscriber,
)
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient

# ── G1 joint index constants ──────────────────────────────────────────────────
# Legs + waist (handled by sport mode)
LEG_WAIST_INDICES = list(range(0, 15))   # 0-14

# Arms (we control directly)
LEFT_ARM_INDICES  = list(range(15, 22))  # 15-21: shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw
RIGHT_ARM_INDICES = list(range(22, 29))  # 22-28: same for right
ARM_INDICES       = LEFT_ARM_INDICES + RIGHT_ARM_INDICES  # 15-28

G1_NUM_MOTORS = 29

# PD gains for arm joint position control — tune on real robot before deployment
ARM_KP = 80.0   # position gain  (N·m/rad)
ARM_KD = 2.0    # velocity gain  (N·m·s/rad)

# Arm joint limits (radians) — from Unitree G1 spec
# [shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw]
LEFT_ARM_Q_MIN  = np.array([-3.14, -1.57, -3.14, -1.57, -3.14, -1.57, -3.14], dtype=np.float32)
LEFT_ARM_Q_MAX  = np.array([ 3.14,  2.88,  3.14,  4.53,  3.14,  1.57,  3.14], dtype=np.float32)
RIGHT_ARM_Q_MIN = np.array([-3.14, -2.88, -3.14, -1.57, -3.14, -1.57, -3.14], dtype=np.float32)
RIGHT_ARM_Q_MAX = np.array([ 3.14,  1.57,  3.14,  4.53,  3.14,  1.57,  3.14], dtype=np.float32)

ARM_Q_MIN = np.concatenate([LEFT_ARM_Q_MIN,  RIGHT_ARM_Q_MIN])  # (14,)
ARM_Q_MAX = np.concatenate([LEFT_ARM_Q_MAX,  RIGHT_ARM_Q_MAX])  # (14,)

# Sport mode velocity limits
MAX_VX   = 0.5   # m/s forward
MAX_VY   = 0.3   # m/s lateral
MAX_VYAW = 0.5   # rad/s yaw rate


@dataclass
class RobotState:
    """Snapshot of robot state — updated asynchronously from SDK2 callbacks."""
    # Joint states (29 joints)
    q:    np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))
    dq:   np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))
    tau:  np.ndarray = field(default_factory=lambda: np.zeros(G1_NUM_MOTORS, dtype=np.float32))

    # IMU (from LowState)
    imu_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.], dtype=np.float32))  # [w,x,y,z]
    imu_rpy:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    imu_gyro: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    # Base pose from sport mode state estimator
    base_pos:  np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    base_quat: np.ndarray = field(default_factory=lambda: np.array([1., 0., 0., 0.], dtype=np.float32))  # [w,x,y,z]

    # Timestamps
    low_state_ts:  float = 0.0
    high_state_ts: float = 0.0


class G1RobotInterface:
    """
    Thread-safe interface to the Unitree G1 robot.

    Parameters
    ----------
    network_interface : str
        Network interface name connected to the G1 (e.g. "enp2s0", "eth0").
    dry_run : bool
        If True, commands are computed but NOT sent to the robot.
        Use for offline testing with a live SDK2 connection.
    """

    def __init__(self, network_interface: str, dry_run: bool = False):
        self.network_interface = network_interface
        self.dry_run = dry_run

        self._state      = RobotState()
        self._state_lock = threading.Lock()
        self._crc        = CRC()
        self._connected  = False

        # LocoClient for sport mode locomotion
        self._loco: Optional[LocoClient] = None

        # LowCmd publisher for arm joint control
        self._low_cmd_pub: Optional[ChannelPublisher] = None
        self._low_cmd     = unitree_hg_msg_dds__LowCmd_()

        # Initialise all motor commands to safe defaults (mode=0, zero torque)
        for i in range(G1_NUM_MOTORS):
            self._low_cmd.motor_cmd[i].mode = 0x01  # FOC mode
            self._low_cmd.motor_cmd[i].q    = 0.0
            self._low_cmd.motor_cmd[i].dq   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0
            self._low_cmd.motor_cmd[i].kp   = 0.0
            self._low_cmd.motor_cmd[i].kd   = 0.0

    # ── Connection ─────────────────────────────────────────────────────────────

    def connect(self) -> None:
        """Initialise SDK2, subscribe to state topics, connect LocoClient."""
        print(f"[RobotInterface] Initialising SDK2 on interface '{self.network_interface}' ...")
        ChannelFactoryInitialize(0, self.network_interface)

        # Subscribe to low-level state
        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._lowstate_callback, 10)

        # Subscribe to sport mode state (for base position)
        try:
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
            self._highstate_sub = ChannelSubscriber("rt/sportmodestate", SportModeState_)
            self._highstate_sub.Init(self._highstate_callback, 10)
        except Exception as e:
            print(f"[RobotInterface] Warning: sport mode state subscriber failed: {e}")

        # LowCmd publisher for arm joints
        if not self.dry_run:
            self._low_cmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
            self._low_cmd_pub.Init()

        # LocoClient for locomotion commands
        self._loco = LocoClient()
        self._loco.SetTimeout(10.0)
        self._loco.Init()

        # Wait for first state message
        print("[RobotInterface] Waiting for first robot state ...")
        timeout = 5.0
        t0 = time.time()
        while self._state.low_state_ts == 0.0:
            time.sleep(0.05)
            if time.time() - t0 > timeout:
                raise TimeoutError("No LowState received within 5 s. Check network connection.")

        self._connected = True
        print("[RobotInterface] Connected.")

    # ── State callbacks ────────────────────────────────────────────────────────

    def _lowstate_callback(self, msg: LowState_) -> None:
        with self._state_lock:
            for i in range(G1_NUM_MOTORS):
                self._state.q[i]   = msg.motor_state[i].q
                self._state.dq[i]  = msg.motor_state[i].dq
                self._state.tau[i] = msg.motor_state[i].tau_est

            imu = msg.imu_state
            # SDK2 IMU quaternion order: [w, x, y, z]
            self._state.imu_quat = np.array(imu.quaternion, dtype=np.float32)[[0, 1, 2, 3]]
            self._state.imu_rpy  = np.array(imu.rpy,        dtype=np.float32)
            self._state.imu_gyro = np.array(imu.gyroscope,  dtype=np.float32)

            # Use IMU quat as base orientation (better than sport mode estimate for fast moves)
            self._state.base_quat = self._state.imu_quat.copy()
            self._state.low_state_ts = time.time()

    def _highstate_callback(self, msg) -> None:
        with self._state_lock:
            self._state.base_pos  = np.array(msg.position, dtype=np.float32)[:3]
            self._state.high_state_ts = time.time()

    # ── State reading ──────────────────────────────────────────────────────────

    def get_state(self) -> RobotState:
        """Return a copy of the latest robot state (thread-safe)."""
        with self._state_lock:
            s = RobotState()
            s.q           = self._state.q.copy()
            s.dq          = self._state.dq.copy()
            s.tau         = self._state.tau.copy()
            s.imu_quat    = self._state.imu_quat.copy()
            s.imu_rpy     = self._state.imu_rpy.copy()
            s.imu_gyro    = self._state.imu_gyro.copy()
            s.base_pos    = self._state.base_pos.copy()
            s.base_quat   = self._state.base_quat.copy()
            s.low_state_ts = self._state.low_state_ts
            s.high_state_ts = self._state.high_state_ts
            return s

    def get_arm_joints(self) -> np.ndarray:
        """Return current arm joint angles (14,) indices 15-28."""
        with self._state_lock:
            return self._state.q[15:29].copy()

    # ── Locomotion commands ────────────────────────────────────────────────────

    def send_locomotion(self, vx: float, vy: float, vyaw: float) -> None:
        """
        Send velocity command to sport mode controller.

        Parameters are clamped to safe limits before sending.
        """
        vx   = float(np.clip(vx,   -MAX_VX,   MAX_VX))
        vy   = float(np.clip(vy,   -MAX_VY,   MAX_VY))
        vyaw = float(np.clip(vyaw, -MAX_VYAW, MAX_VYAW))

        if self.dry_run:
            return
        if self._loco is not None:
            self._loco.Move(vx, vy, vyaw)

    def send_stand_height(self, height: float) -> None:
        """Set body stand height (meters). Clamped to [0.65, 0.90]."""
        height = float(np.clip(height, 0.65, 0.90))
        if self.dry_run:
            return
        if self._loco is not None:
            self._loco.SetStandHeight(height)

    def stop_locomotion(self) -> None:
        """Stop locomotion (zero velocity)."""
        if self.dry_run:
            return
        if self._loco is not None:
            self._loco.StopMove()

    # ── Arm joint commands ─────────────────────────────────────────────────────

    def send_arm_joints(self, q_targets: np.ndarray, kp: float = ARM_KP, kd: float = ARM_KD) -> None:
        """
        Send position targets to arm joints (indices 15-28).

        Parameters
        ----------
        q_targets : np.ndarray, shape (14,)
            Target joint angles in radians [left_arm(7), right_arm(7)].
        kp, kd : float
            PD gains. Use lower values (kp=20, kd=1) during first tests.
        """
        assert q_targets.shape == (14,), f"Expected (14,) arm targets, got {q_targets.shape}"

        # Clamp to joint limits
        q_targets = np.clip(q_targets, ARM_Q_MIN, ARM_Q_MAX)

        # Zero out leg/waist motors — we do NOT command them
        for i in LEG_WAIST_INDICES:
            self._low_cmd.motor_cmd[i].mode = 0x00  # disable (sport mode owns these)
            self._low_cmd.motor_cmd[i].kp   = 0.0
            self._low_cmd.motor_cmd[i].kd   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0

        # Set arm joint commands
        for local_i, global_i in enumerate(ARM_INDICES):
            self._low_cmd.motor_cmd[global_i].mode = 0x01  # FOC mode
            self._low_cmd.motor_cmd[global_i].q    = float(q_targets[local_i])
            self._low_cmd.motor_cmd[global_i].dq   = 0.0
            self._low_cmd.motor_cmd[global_i].tau  = 0.0
            self._low_cmd.motor_cmd[global_i].kp   = kp
            self._low_cmd.motor_cmd[global_i].kd   = kd

        self._low_cmd.crc = self._crc.Crc(self._low_cmd)

        if self.dry_run:
            return
        if self._low_cmd_pub is not None:
            self._low_cmd_pub.Write(self._low_cmd)

    # ── Emergency stop ─────────────────────────────────────────────────────────

    def emergency_stop(self) -> None:
        """
        Immediately damp all motors and stop locomotion.
        Should be called on any safety violation.
        """
        print("[RobotInterface] *** EMERGENCY STOP ***")
        try:
            self.stop_locomotion()
            if self._loco is not None:
                self._loco.Damp()
        except Exception as e:
            print(f"[RobotInterface] Emergency stop error (loco): {e}")

        # Zero arm torques
        for i in ARM_INDICES:
            self._low_cmd.motor_cmd[i].mode = 0x00
            self._low_cmd.motor_cmd[i].kp   = 0.0
            self._low_cmd.motor_cmd[i].kd   = 0.0
            self._low_cmd.motor_cmd[i].tau  = 0.0
        try:
            self._low_cmd.crc = self._crc.Crc(self._low_cmd)
            if not self.dry_run and self._low_cmd_pub is not None:
                self._low_cmd_pub.Write(self._low_cmd)
        except Exception as e:
            print(f"[RobotInterface] Emergency stop error (arms): {e}")

    def stand_up(self) -> None:
        """Bring robot from lie/squat position to standing. Call once at startup."""
        if self.dry_run:
            print("[RobotInterface][DRY RUN] stand_up() skipped")
            return
        print("[RobotInterface] Standing up ...")
        self._loco.Damp()
        time.sleep(1.0)
        self._loco.Squat2StandUp()
        time.sleep(3.0)
        print("[RobotInterface] Standing.")

    def damp(self) -> None:
        """Put robot into damping (compliant) mode."""
        if self.dry_run:
            return
        if self._loco is not None:
            self._loco.Damp()
