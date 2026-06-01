"""
sdk2_interface.py — Unitree SDK2 wrapper for G1 real robot deployment.

Architecture (Option 1):
  - LocoClient.Move(vx, vy, vyaw)  →  legs (Unitree internal WBC, sport mode)
  - LowCmd arm joints (15-28)      →  arms via arm_sdk mode
  - LowState subscription          →  read joint positions + IMU

IMPORTANT: The G1 must be in a state where:
  1. Sport mode is active for locomotion (LocoClient)
  2. Arm SDK mode is enabled for independent arm control
  See DEPLOYMENT_GUIDE.md for the startup sequence.

NOTE: unitree_sdk2py is only available on the companion PC connected to the robot
over Ethernet. It is NOT available in the sim training environment.
"""

from __future__ import annotations

import threading
import time
import numpy as np

# ── SDK2 imports (only available on companion PC with unitree_sdk2py) ────────
try:
    from unitree_sdk2py.core.channel import ChannelFactory, ChannelFactoryInitialize
    from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
    from unitree_sdk2py.utils.crc import CRC
    SDK2_AVAILABLE = True
except ImportError:
    SDK2_AVAILABLE = False
    print("[sdk2_interface] WARNING: unitree_sdk2py not found — running in DRY-RUN mode")

from wbc.robot_model_real import (
    SDK2_N_JOINTS,
    ARM_SDK2_INDICES,
    WAIST_SDK2_INDICES,
)


# ── G1 Joint PD Gain Defaults ─────────────────────────────────────────────────
# These are conservative safe values. Tune after verifying tracking performance.
DEFAULT_ARM_KP   = 80.0
DEFAULT_ARM_KD   = 2.0
DEFAULT_WAIST_KP = 200.0
DEFAULT_WAIST_KD = 5.0


class G1SDK2Interface:
    """
    Thin wrapper around unitree_sdk2py for G1 real robot deployment.

    Responsibilities:
      - Subscribe to LowState  → provide joint positions + IMU
      - Subscribe to HighState → provide base position estimate
      - Publish LocoClient commands for locomotion
      - Publish arm joint targets via arm_sdk / LowCmd
    """

    def __init__(self, cfg: dict):
        """
        Args:
            cfg: dict from deploy_config.yaml robot + wbc sections
        """
        self._network_iface = cfg["robot"]["network_interface"]
        self._arm_kp   = cfg["wbc"].get("arm_kp",   DEFAULT_ARM_KP)
        self._arm_kd   = cfg["wbc"].get("arm_kd",   DEFAULT_ARM_KD)
        self._waist_kp = cfg["wbc"].get("waist_kp", DEFAULT_WAIST_KP)
        self._waist_kd = cfg["wbc"].get("waist_kd", DEFAULT_WAIST_KD)

        # Latest state (thread-safe via lock)
        self._lock          = threading.Lock()
        self._joint_pos     = np.zeros(SDK2_N_JOINTS, dtype=np.float32)
        self._joint_vel     = np.zeros(SDK2_N_JOINTS, dtype=np.float32)
        self._imu_quat_wxyz = np.array([1., 0., 0., 0.], dtype=np.float32)
        self._base_pos      = np.zeros(3, dtype=np.float32)
        self._state_ready   = False

        self._dry_run = not SDK2_AVAILABLE

        if not self._dry_run:
            self._init_sdk2()
        else:
            print("[G1SDK2Interface] DRY-RUN: no commands will be sent to robot")

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_sdk2(self):
        ChannelFactoryInitialize(0, self._network_iface)

        # ── LowState subscriber ──────────────────────────────────────────────
        self._low_state_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._low_state_sub.Init(self._low_state_callback, 10)

        # ── LocoClient for locomotion ────────────────────────────────────────
        self._loco_client = LocoClient()
        self._loco_client.SetTimeout(10.0)
        self._loco_client.Init()

        # ── Arm LowCmd publisher ─────────────────────────────────────────────
        # The G1 supports arm joint control via "rt/arm_sdk" alongside sport mode.
        # If your SDK version uses a different topic, update this string.
        # Verified topic name: "rt/arm_sdk" (unitree_sdk2py >= 1.0.0 for G1)
        self._arm_cmd_pub = ChannelPublisher("rt/arm_sdk", LowCmd_)
        self._arm_cmd_pub.Init()

        self._crc = CRC()
        print(f"[G1SDK2Interface] Initialised on {self._network_iface}")

        # Wait for first state
        timeout = 5.0
        t0 = time.time()
        while not self._state_ready and time.time() - t0 < timeout:
            time.sleep(0.05)
        if not self._state_ready:
            raise RuntimeError("[G1SDK2Interface] Timed out waiting for LowState — check connection")

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def _low_state_callback(self, msg: "LowState_"):
        with self._lock:
            for i in range(SDK2_N_JOINTS):
                self._joint_pos[i] = msg.motor_state[i].q
                self._joint_vel[i] = msg.motor_state[i].dq

            # IMU quaternion: SDK2 gives [w, x, y, z]
            q = msg.imu_state.quaternion
            self._imu_quat_wxyz = np.array([q[0], q[1], q[2], q[3]], dtype=np.float32)

            # Base position from state estimator (HighState or SportModeState)
            # SDK2 LowState does not include base position directly.
            # We use IMU integration approximation here; for better accuracy
            # subscribe to "rt/sportmodestate" separately (see comments below).
            self._state_ready = True

    # ── State Reads ───────────────────────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Return latest joint positions (29,) in radians."""
        with self._lock:
            return self._joint_pos.copy()

    def get_joint_velocities(self) -> np.ndarray:
        """Return latest joint velocities (29,) in rad/s."""
        with self._lock:
            return self._joint_vel.copy()

    def get_imu_quat_wxyz(self) -> np.ndarray:
        """Return latest IMU quaternion [w, x, y, z]."""
        with self._lock:
            return self._imu_quat_wxyz.copy()

    def get_base_pos(self) -> np.ndarray:
        """
        Return estimated base position (3,) in world frame.

        NOTE: LowState does not include base XY position. For full positional
        tracking, subscribe to "rt/sportmodestate" separately and read
        SportModeState_.position[3]. This returns zeros if not implemented.
        """
        with self._lock:
            return self._base_pos.copy()

    # ── Locomotion Commands ───────────────────────────────────────────────────

    def send_loco_cmd(
        self,
        vx: float,
        vy: float,
        vyaw: float,
    ) -> None:
        """
        Send locomotion velocity command to Unitree internal WBC (sport mode).

        Args:
            vx:   forward velocity (m/s)
            vy:   lateral velocity (m/s)
            vyaw: yaw rate (rad/s)
        """
        if self._dry_run:
            return
        self._loco_client.Move(vx, vy, vyaw)

    def stop_loco(self) -> None:
        """Stop locomotion immediately."""
        if self._dry_run:
            return
        self._loco_client.StopMove()

    # ── Arm Commands ──────────────────────────────────────────────────────────

    def send_arm_joint_targets(
        self,
        arm_waist_q: np.ndarray,
        prev_arm_q: np.ndarray | None = None,
        max_delta: float = 0.08,
    ) -> None:
        """
        Send arm + waist joint position targets via arm_sdk (rt/arm_sdk).

        Only indices in ARM_SDK2_INDICES (15-28) and WAIST_SDK2_INDICES (12-14)
        are written to the LowCmd. Leg indices (0-11) are NOT touched.

        Args:
            arm_waist_q: (29,) full SDK2 joint array — only arm/waist used
            prev_arm_q:  (29,) previous command for delta clipping (safety)
            max_delta:   max joint change per step (rad)
        """
        if self._dry_run:
            return

        cmd = LowCmd_()
        cmd.mode_pr      = 0
        cmd.mode_machine = 0

        active_indices = ARM_SDK2_INDICES + WAIST_SDK2_INDICES

        for idx in active_indices:
            target_q = float(arm_waist_q[idx])

            # Safety: rate-limit joint delta
            if prev_arm_q is not None and max_delta > 0:
                prev_q   = float(prev_arm_q[idx])
                delta    = np.clip(target_q - prev_q, -max_delta, max_delta)
                target_q = prev_q + delta

            kp = self._arm_kp   if idx in ARM_SDK2_INDICES else self._waist_kp
            kd = self._arm_kd   if idx in ARM_SDK2_INDICES else self._waist_kd

            cmd.motor_cmd[idx].q   = target_q
            cmd.motor_cmd[idx].dq  = 0.0
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].kp  = kp
            cmd.motor_cmd[idx].kd  = kd

        cmd.crc = self._crc.Crc(cmd)
        self._arm_cmd_pub.Write(cmd)

    def zero_arm_torques(self) -> None:
        """Send zero-torque (limp) command to arm joints — use for safe shutdown."""
        if self._dry_run:
            return
        cmd = LowCmd_()
        for idx in ARM_SDK2_INDICES + WAIST_SDK2_INDICES:
            cmd.motor_cmd[idx].q   = 0.0
            cmd.motor_cmd[idx].dq  = 0.0
            cmd.motor_cmd[idx].tau = 0.0
            cmd.motor_cmd[idx].kp  = 0.0
            cmd.motor_cmd[idx].kd  = 2.0   # small damping to settle safely
        cmd.crc = self._crc.Crc(cmd)
        self._arm_cmd_pub.Write(cmd)

    # ── Safety / Shutdown ─────────────────────────────────────────────────────

    def safe_stop(self) -> None:
        """Emergency stop: halt locomotion and limp arms."""
        print("[G1SDK2Interface] SAFE STOP")
        self.stop_loco()
        self.zero_arm_torques()
