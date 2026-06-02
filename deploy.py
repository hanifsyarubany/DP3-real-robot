"""
deploy.py — Deploy EquivDP3 BC policy on real Unitree G1 (development mode).

Control architecture
--------------------
  EquivDP3 @ 6 Hz  →  23D WBC action chunk (horizon=16, execute 8)
      │
      ├── c_t[2:16]  wrist EEF poses   → PINK IK   → arm joints (15-28)  ─┐
      ├── c_t[16:19] nav_cmd [vx,vy,w] → HOMIE loco → leg joints (0-11)  ─┤→ LowCmd @ 50 Hz
      ├── c_t[19]    base_height        → HOMIE loco → waist joints (12-14) ┘
      └── c_t[20:23] torso_rpy         → HOMIE loco

Mode: DEVELOPMENT MODE — full 29-DOF LowCmd. No LocoClient.
Locomotion is handled entirely by the HOMIE ONNX policy (stand.onnx / walk.onnx).

Threading
---------
  policy_thread : EquivDP3 DDIM inference @ ~6 Hz
  control_loop  : 50 Hz — reads state, runs HOMIE + PINK IK, sends LowCmd

Usage
-----
    python deploy.py [--network eth0] [--checkpoint checkpoint/final.ckpt]
                     [--dry-run] [--low-kp] [--max-steps 2000]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np
import torch

_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from policy_loader   import load_policy
from robot_interface import G1RobotInterface
from pcd_pipeline    import PointCloudPipeline, DEFAULT_T_PELVIS_CAM
from arm_ik          import ArmIK
from obs_builder     import ObservationBuilder
from safety_monitor  import SafetyMonitor
from wbc.homie_loco  import HomieLocoPolicy

# ── Policy constants (must match training) ─────────────────────────────────────
N_OBS_STEPS    = 2
N_ACTION_STEPS = 8
HORIZON        = 16
HLC_FREQ       = 6     # Hz
LLC_FREQ       = 50    # Hz
LLC_DT         = 1.0 / LLC_FREQ

_STAND_ONNX = str(_HERE / "checkpoint" / "homie_v2" / "stand.onnx")
_WALK_ONNX  = str(_HERE / "checkpoint" / "homie_v2" / "walk.onnx")


def parse_args():
    p = argparse.ArgumentParser(description="Deploy EquivDP3 on Unitree G1")
    p.add_argument("--network",    default="eth0")
    p.add_argument("--checkpoint", default=str(_HERE / "checkpoint" / "final.ckpt"))
    p.add_argument("--stand-onnx", default=_STAND_ONNX)
    p.add_argument("--walk-onnx",  default=_WALK_ONNX)
    p.add_argument("--device",     default="cuda")
    p.add_argument("--dry-run",    action="store_true")
    p.add_argument("--low-kp",     action="store_true",
                   help="Conservative arm gains (kp=20) for first tests")
    p.add_argument("--max-steps",  type=int, default=2000)
    return p.parse_args()


def main():
    args = parse_args()
    arm_kp = 20.0 if args.low_kp else 80.0
    arm_kd = 1.0  if args.low_kp else 2.0

    print("=" * 60)
    print("  EquivDP3 Real Robot Deployment  (dev mode)")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY-RUN — no commands sent ***")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  HOMIE stand : {args.stand_onnx}")
    print(f"  HOMIE walk  : {args.walk_onnx}")
    print(f"  Arm kp/kd   : {arm_kp} / {arm_kd}")
    print("=" * 60)

    # ── 1. Load EquivDP3 ──────────────────────────────────────────────────────
    print("\n[1/6] Loading EquivDP3 ...")
    device = args.device if torch.cuda.is_available() else "cpu"
    policy, _ = load_policy(args.checkpoint, device=device)

    # ── 2. Robot interface (dev mode, full LowCmd) ────────────────────────────
    print("\n[2/6] Connecting to G1 (development mode) ...")
    robot = G1RobotInterface(network_interface=args.network, dry_run=args.dry_run)
    robot.connect()

    # ── 3. HOMIE locomotion policy ────────────────────────────────────────────
    print("\n[3/6] Loading HOMIE v2 locomotion policy ...")
    homie = HomieLocoPolicy(stand_onnx=args.stand_onnx, walk_onnx=args.walk_onnx)

    # ── 4. PCD + obs builder + arm IK ─────────────────────────────────────────
    print("\n[4/6] Starting RealSense + obs builder + arm IK ...")
    pcd_pipe    = PointCloudPipeline(T_pelvis_cam=DEFAULT_T_PELVIS_CAM)
    obs_builder = ObservationBuilder(device=device)
    arm_ik      = ArmIK()
    if not args.dry_run:
        pcd_pipe.connect()

    # ── 5. Safety ─────────────────────────────────────────────────────────────
    print("\n[5/6] Safety monitor started (press  q + Enter  to E-stop).")
    safety = SafetyMonitor(robot)

    # ── 6. Stand up ───────────────────────────────────────────────────────────
    print("\n[6/6] Standing up via LowCmd interpolation ...")
    if not args.dry_run:
        robot.damp()
        time.sleep(1.0)
        robot.stand_up(duration_s=3.0)
    print("\nReady — starting policy.\n")

    # ── Shared state ──────────────────────────────────────────────────────────
    _chunk_lock  = threading.Lock()
    _action_chunk = np.zeros((HORIZON, 23), dtype=np.float32)
    _chunk_step  = [N_ACTION_STEPS]     # triggers first inference immediately
    _running     = [True]
    _first_done  = threading.Event()

    # Pre-fill obs history with one real frame
    state = robot.get_state()
    pcd   = pcd_pipe.get_point_cloud() if not args.dry_run else pcd_pipe.get_dummy_pcd()
    obs_builder.push(pcd, obs_builder.build_agent_pos(state))
    obs_builder.pad_to_full()

    # ── Policy thread (6 Hz) ──────────────────────────────────────────────────

    def policy_thread():
        while _running[0] and not safety.stop_requested:
            t0 = time.time()
            obs = obs_builder.get_obs_input()
            with torch.no_grad():
                chunk = policy.predict_action(obs)["action_pred"][0].cpu().numpy()
            with _chunk_lock:
                _action_chunk[:] = chunk
                _chunk_step[0]   = 0
            _first_done.set()
            time.sleep(max(0.0, 1.0 / HLC_FREQ - (time.time() - t0)))

    threading.Thread(target=policy_thread, daemon=True).start()
    print("Waiting for first EquivDP3 inference ...")
    _first_done.wait(timeout=15.0)
    print("Inference ready — 50 Hz control loop starting.\n")

    # ── 50 Hz control loop ────────────────────────────────────────────────────
    step   = 0
    t_loop = time.time()

    try:
        while step < args.max_steps and not safety.stop_requested:
            t_start = time.time()

            # Sense
            state = robot.get_state()
            pcd   = pcd_pipe.get_point_cloud() if not args.dry_run else pcd_pipe.get_dummy_pcd()
            obs_builder.push(pcd, obs_builder.build_agent_pos(state))

            # Get current action from chunk
            with _chunk_lock:
                cs  = min(_chunk_step[0], N_ACTION_STEPS - 1)
                c_t = _action_chunk[cs].copy()
                _chunk_step[0] += 1

            # Decode 23D action
            left_wrist_pos   = c_t[2:5]
            left_wrist_quat  = c_t[5:9]
            right_wrist_pos  = c_t[9:12]
            right_wrist_quat = c_t[12:16]
            nav_vx    = float(np.clip(c_t[16], -0.3, 0.3))
            nav_vy    = float(np.clip(c_t[17], -0.2, 0.2))
            nav_vyaw  = float(np.clip(c_t[18], -0.3, 0.3))
            base_h    = float(np.clip(c_t[19],  0.65, 0.90))
            torso_rpy = c_t[20:23]

            # HOMIE: lower body (legs + waist)
            homie.set_goal(vx=nav_vx, vy=nav_vy, vyaw=nav_vyaw,
                           height_cmd=base_h, torso_rpy=torso_rpy)
            q_lower = homie.get_action(
                q_all=state.q, dq_all=state.dq,
                imu_quat_wxyz=state.imu_quat, imu_gyro=state.imu_gyro,
            )   # (15,)

            # PINK IK: arms
            arm_ik.update_joint_angles(robot.get_arm_joints())
            q_arms = arm_ik.solve(
                left_wrist_pos, left_wrist_quat,
                right_wrist_pos, right_wrist_quat,
            )   # (14,)

            # Safety
            if not safety.check_all(q_arm=q_arms, vx=nav_vx, vy=nav_vy,
                                    vyaw=nav_vyaw, low_state_ts=state.low_state_ts):
                print(f"[Deploy] Safety stop at step {step}")
                break

            # Send full 29-DOF LowCmd
            robot.send_full_body(q_lower, q_arms, arm_kp=arm_kp, arm_kd=arm_kd)
            step += 1

            time.sleep(max(0.0, LLC_DT - (time.time() - t_start)))

    except KeyboardInterrupt:
        print("\n[Deploy] Keyboard interrupt.")

    finally:
        _running[0] = False
        robot.emergency_stop()
        if not args.dry_run:
            pcd_pipe.disconnect()
        t = time.time() - t_loop
        print(f"[Deploy] {step} steps in {t:.1f} s ({step/max(t,1):.1f} Hz avg)")


if __name__ == "__main__":
    main()
