"""
deploy.py
=========
Main deployment script: run EquivDP3 BC policy on a real Unitree G1 robot.

Control hierarchy
-----------------
  EquivDP3 @ 6 Hz  →  23D WBC action chunk (16 steps, execute 8)
      │
      ├── c_t[0:2]   hand states        → ignored (always 0 = open)
      ├── c_t[2:9]   left  wrist pose   → PINK IK → left  arm joints (SDK2 LowCmd)
      ├── c_t[9:16]  right wrist pose   → PINK IK → right arm joints (SDK2 LowCmd)
      ├── c_t[16:19] nav_cmd [vx,vy,vw] → LocoClient.Move()
      ├── c_t[19]    base_height        → LocoClient.SetStandHeight()
      └── c_t[20:23] torso_rpy          → (logged only — not directly mapped)

Threading model
---------------
  policy_thread : EquivDP3 inference @ ~6 Hz (takes ~70-100 ms)
  control_loop  : sends commands @ 50 Hz, consuming action chunk produced by policy

Usage
-----
    python deploy.py --network eth0 --checkpoint checkpoint/final.ckpt [--dry-run]

Arguments
---------
    --network     Network interface to the G1 (e.g. enp2s0, eth0)
    --checkpoint  Path to .ckpt file (default: checkpoint/final.ckpt)
    --device      PyTorch device (default: cuda)
    --dry-run     Run policy + IK but do NOT send commands to robot
    --low-kp      Reduced arm PD gain for first tests (default: 20)
    --max-steps   Episode step limit before auto-stop (default: 2000)
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import threading
import time

import numpy as np
import torch

# ── Make local modules importable regardless of cwd ───────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from policy_loader  import load_policy
from robot_interface import G1RobotInterface
from pcd_pipeline   import PointCloudPipeline, DEFAULT_T_PELVIS_CAM
from arm_ik         import ArmIK
from obs_builder    import ObservationBuilder
from safety_monitor import SafetyMonitor

# ── Constants (must match training) ──────────────────────────────────────────
N_OBS_STEPS    = 2
N_ACTION_STEPS = 8     # steps to execute per DP3 inference
HORIZON        = 16    # DP3 prediction horizon
HLC_FREQ       = 6     # Hz — EquivDP3 inference frequency
LLC_FREQ       = 50    # Hz — arm joint + locomotion commands
LLC_DT         = 1.0 / LLC_FREQ  # 0.02 s per control step


def parse_args():
    p = argparse.ArgumentParser(description="Deploy EquivDP3 on Unitree G1")
    p.add_argument("--network",    type=str, default="eth0",
                   help="Network interface to G1 (e.g. enp2s0)")
    p.add_argument("--checkpoint", type=str,
                   default=str(_HERE / "checkpoint" / "final.ckpt"),
                   help="Path to .ckpt file")
    p.add_argument("--device",     type=str, default="cuda",
                   help="PyTorch device for policy inference")
    p.add_argument("--dry-run",    action="store_true",
                   help="Run policy + IK but do not send commands")
    p.add_argument("--low-kp",     action="store_true",
                   help="Use conservative arm PD gains (kp=20) for first tests")
    p.add_argument("--max-steps",  type=int, default=2000,
                   help="Max control steps before auto-stop")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  EquivDP3 Real Robot Deployment")
    print("=" * 60)
    if args.dry_run:
        print("  *** DRY RUN MODE — no commands sent to robot ***")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Network    : {args.network}")
    print(f"  Device     : {args.device}")
    print("=" * 60)

    # ── 1. Load policy ────────────────────────────────────────────────────────
    print("\n[1/6] Loading policy ...")
    policy, cfg = load_policy(args.checkpoint, device=args.device)

    # ── 2. Initialise robot interface ─────────────────────────────────────────
    print("\n[2/6] Connecting to G1 robot ...")
    robot = G1RobotInterface(network_interface=args.network, dry_run=args.dry_run)
    robot.connect()

    # ── 3. Point cloud pipeline ───────────────────────────────────────────────
    print("\n[3/6] Starting RealSense pipeline ...")
    pcd_pipe = PointCloudPipeline(T_pelvis_cam=DEFAULT_T_PELVIS_CAM)
    if args.dry_run:
        print("  [DRY RUN] Using dummy zero point cloud.")
    else:
        pcd_pipe.connect()

    # ── 4. Observation builder + IK ───────────────────────────────────────────
    print("\n[4/6] Initialising observation builder and arm IK ...")
    obs_builder = ObservationBuilder(device=args.device)
    arm_ik      = ArmIK()

    # ── 5. Safety monitor ─────────────────────────────────────────────────────
    print("\n[5/6] Setting up safety monitor ...")
    arm_kp = 20.0 if args.low_kp else 80.0
    arm_kd = 1.0  if args.low_kp else 2.0
    safety = SafetyMonitor(robot)
    print(f"  Arm PD gains: kp={arm_kp}, kd={arm_kd}")

    # ── 6. Stand up ───────────────────────────────────────────────────────────
    print("\n[6/6] Standing robot up ...")
    if not args.dry_run:
        robot.stand_up()
    print("\nReady. Starting control loop. (Press  q + Enter  to stop)\n")

    # ── Shared state between policy_thread and control_loop ───────────────────
    _action_chunk_lock   = threading.Lock()
    _action_chunk        = np.zeros((HORIZON, 23), dtype=np.float32)
    _chunk_step          = [N_ACTION_STEPS]   # force first inference immediately
    _policy_running      = [True]
    _inference_complete  = threading.Event()

    # ── Policy inference thread ───────────────────────────────────────────────

    def policy_thread():
        """Runs EquivDP3 inference and updates _action_chunk."""
        while _policy_running[0] and not safety.stop_requested:
            t0 = time.time()

            # Build obs from current history
            obs = obs_builder.get_obs_input()

            with torch.no_grad():
                result = policy.predict_action(obs)

            new_chunk = result["action_pred"][0].cpu().numpy()  # (16, 23)

            with _action_chunk_lock:
                _action_chunk[:] = new_chunk
                _chunk_step[0]   = 0

            _inference_complete.set()

            elapsed = time.time() - t0
            # Sleep for remainder of HLC period (nominal 1/6 s ≈ 166 ms)
            sleep_t = max(0.0, 1.0 / HLC_FREQ - elapsed)
            time.sleep(sleep_t)

    # ── Pre-fill observation history ──────────────────────────────────────────
    state = robot.get_state()
    pcd   = pcd_pipe.get_point_cloud() if not args.dry_run else pcd_pipe.get_dummy_pcd()
    ap    = obs_builder.build_agent_pos(state)
    obs_builder.push(pcd, ap)
    obs_builder.pad_to_full()

    # Wait for first policy inference
    inf_thread = threading.Thread(target=policy_thread, daemon=True)
    inf_thread.start()
    print("Waiting for first policy inference ...")
    _inference_complete.wait(timeout=10.0)
    print("First inference complete. Executing policy.\n")

    # ── Main 50 Hz control loop ────────────────────────────────────────────────
    step   = 0
    t_loop = time.time()

    try:
        while step < args.max_steps and not safety.stop_requested:
            t_start = time.time()

            # ── Sense ────────────────────────────────────────────────────────
            state = robot.get_state()
            pcd   = pcd_pipe.get_point_cloud() if not args.dry_run else pcd_pipe.get_dummy_pcd()
            ap    = obs_builder.build_agent_pos(state)
            obs_builder.push(pcd, ap)

            # ── Get current action from chunk ─────────────────────────────
            with _action_chunk_lock:
                cs   = min(_chunk_step[0], N_ACTION_STEPS - 1)
                c_t  = _action_chunk[cs].copy()
                _chunk_step[0] += 1

            # ── Decode 23D action ─────────────────────────────────────────
            # c_t[0:2]   hand states  (always 0)
            left_wrist_pos  = c_t[2:5]    # left  wrist pos in pelvis frame
            left_wrist_quat = c_t[5:9]    # left  wrist quat [w,x,y,z] in pelvis frame
            right_wrist_pos  = c_t[9:12]  # right wrist pos in pelvis frame
            right_wrist_quat = c_t[12:16] # right wrist quat [w,x,y,z] in pelvis frame
            nav_vx   = c_t[16]
            nav_vy   = c_t[17]
            nav_vyaw = c_t[18]
            base_height = float(np.clip(c_t[19], 0.65, 0.90))

            # ── Arm IK ────────────────────────────────────────────────────
            arm_ik.update_joint_angles(robot.get_arm_joints())
            q_arm = arm_ik.solve(
                left_wrist_pos,  left_wrist_quat,
                right_wrist_pos, right_wrist_quat,
            )

            # ── Safety checks ─────────────────────────────────────────────
            safe = safety.check_all(
                q_arm      = q_arm,
                vx         = nav_vx,
                vy         = nav_vy,
                vyaw       = nav_vyaw,
                low_state_ts = state.low_state_ts,
            )
            if not safe:
                print(f"[Deploy] Safety check failed at step {step}. Stopping.")
                break

            # ── Send commands ─────────────────────────────────────────────
            robot.send_arm_joints(q_arm, kp=arm_kp, kd=arm_kd)
            robot.send_locomotion(nav_vx, nav_vy, nav_vyaw)
            robot.send_stand_height(base_height)

            step += 1

            # ── Maintain 50 Hz ────────────────────────────────────────────
            elapsed = time.time() - t_start
            sleep_t = max(0.0, LLC_DT - elapsed)
            time.sleep(sleep_t)

    except KeyboardInterrupt:
        print("\n[Deploy] Keyboard interrupt received.")

    finally:
        # ── Graceful shutdown ─────────────────────────────────────────────
        print("\n[Deploy] Shutting down ...")
        _policy_running[0] = False
        robot.stop_locomotion()
        robot.damp()
        if not args.dry_run:
            pcd_pipe.disconnect()
        print(f"[Deploy] Completed {step} control steps.")
        total_time = time.time() - t_loop
        print(f"[Deploy] Total run time: {total_time:.1f} s  ({step/total_time:.1f} Hz avg)")


if __name__ == "__main__":
    main()
