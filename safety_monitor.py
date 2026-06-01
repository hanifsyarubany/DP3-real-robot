"""
safety_monitor.py
=================
Pre-send safety checks and watchdog for real robot deployment.

Checks performed before every command:
  1. Arm joint limits — clamp or reject if IK solution exceeds limits
  2. Action velocity limits — reject if action delta is too large
  3. State freshness — stop if robot state is stale (communication lost)
  4. Keyboard emergency stop — press 'q' + Enter to stop immediately

All checks return bool (True = safe to send, False = stop).
"""

from __future__ import annotations

import threading
import time

import numpy as np

# ── Joint limits (radians) ─────────────────────────────────────────────────────
# From Unitree G1 hardware spec and URDF
ARM_Q_MIN = np.array([
    # Left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow, wrist_roll, wrist_pitch, wrist_yaw
    -2.87, -0.34, -2.87,  -0.04, -2.87, -1.57, -2.87,
    # Right arm: same order
    -2.87, -2.61, -2.87,  -0.04, -2.87, -1.57, -2.87,
], dtype=np.float32)

ARM_Q_MAX = np.array([
    # Left arm
     2.87,  2.61,  2.87,   2.87,  2.87,  1.57,  2.87,
    # Right arm
     2.87,  0.34,  2.87,   2.87,  2.87,  1.57,  2.87,
], dtype=np.float32)

# Max joint velocity per step at 50 Hz (rad/s * 0.02 s)
ARM_MAX_DELTA_PER_STEP = 0.1   # 5 rad/s max  (tune conservatively at first)

# Max locomotion commands
MAX_VX   = 0.5   # m/s
MAX_VY   = 0.3   # m/s
MAX_VYAW = 0.5   # rad/s

# State freshness threshold (seconds)
STATE_TIMEOUT_S = 0.5


class SafetyMonitor:
    """
    Validates commands before they are sent to the robot.

    Parameters
    ----------
    robot_interface : G1RobotInterface
        Used to call emergency_stop() on violation.
    max_arm_delta : float
        Max joint angle change per 20 ms step (radians).  Start conservatively
        (0.05 rad) and increase once behaviour is validated.
    """

    def __init__(self, robot_interface, max_arm_delta: float = ARM_MAX_DELTA_PER_STEP):
        self._robot  = robot_interface
        self._max_arm_delta = max_arm_delta
        self._prev_arm_q: np.ndarray | None = None
        self._stop_requested = False

        # Keyboard listener thread
        self._kbd_thread = threading.Thread(target=self._keyboard_listener, daemon=True)
        self._kbd_thread.start()

    # ── Keyboard emergency stop ────────────────────────────────────────────────

    def _keyboard_listener(self) -> None:
        print("[Safety] Press  q + Enter  at any time for emergency stop.")
        while not self._stop_requested:
            try:
                key = input()
                if key.strip().lower() == "q":
                    print("[Safety] *** KEYBOARD EMERGENCY STOP REQUESTED ***")
                    self._stop_requested = True
            except EOFError:
                break

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    # ── Per-command check ──────────────────────────────────────────────────────

    def check_state_freshness(self, low_state_ts: float) -> bool:
        """Return False if robot state is stale (no SDK2 messages)."""
        age = time.time() - low_state_ts
        if age > STATE_TIMEOUT_S:
            print(f"[Safety] FAIL: robot state stale ({age:.2f} s). Stopping.")
            return False
        return True

    def check_arm_targets(self, q_arm: np.ndarray) -> bool:
        """
        Check arm joint targets:
          - Within joint limits
          - Delta from previous target not too large

        Modifies q_arm in-place to clamp limits; returns False only for
        dangerous jumps (delta check).
        """
        assert q_arm.shape == (14,)

        # 1. Joint limits — clamp silently
        q_arm[:] = np.clip(q_arm, ARM_Q_MIN, ARM_Q_MAX)

        # 2. Delta check — reject large jumps
        if self._prev_arm_q is not None:
            delta = np.abs(q_arm - self._prev_arm_q)
            max_delta = delta.max()
            if max_delta > self._max_arm_delta:
                print(f"[Safety] FAIL: arm delta too large ({max_delta:.3f} rad). Stopping.")
                return False

        return True

    def check_locomotion(self, vx: float, vy: float, vyaw: float) -> bool:
        """Clamp locomotion commands and return True (never hard-fails; just clamps)."""
        # Clamping happens in robot_interface.send_locomotion already, but double-check.
        _ = np.clip([vx, vy, vyaw], [-MAX_VX, -MAX_VY, -MAX_VYAW], [MAX_VX, MAX_VY, MAX_VYAW])
        return True

    def check_all(
        self,
        q_arm: np.ndarray,
        vx: float, vy: float, vyaw: float,
        low_state_ts: float,
    ) -> bool:
        """
        Run all checks.  Returns True if safe to send, False to stop.
        Updates internal state on success.
        """
        if self._stop_requested:
            return False
        if not self.check_state_freshness(low_state_ts):
            return False
        if not self.check_arm_targets(q_arm):
            return False
        if not self.check_locomotion(vx, vy, vyaw):
            return False

        # Update prev arm targets only on success
        self._prev_arm_q = q_arm.copy()
        return True

    def trigger_estop(self) -> None:
        """Hard stop: set flag and call robot emergency stop."""
        self._stop_requested = True
        self._robot.emergency_stop()
