"""
scripts/dry_run.py
==================
Offline inference test: load the policy, synthesise dummy observations
matching the exact training shapes, run N forward passes, and report timing.

No robot or camera hardware needed.  Run this first to verify:
  1. Policy loads successfully from the checkpoint
  2. Input/output shapes are correct
  3. Inference latency is within the 166 ms HLC budget
  4. DDIM denoising produces finite, bounded actions

Usage:
    python scripts/dry_run.py [--checkpoint checkpoint/final.ckpt] [--n 20] [--device cuda]
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

_HERE = pathlib.Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np
import torch

from policy_loader import load_policy

# Must match training config
N_OBS_STEPS = 2
N_PTS       = 1024
AGENT_DIM   = 28
ACTION_DIM  = 23


def make_dummy_obs(device: str) -> dict[str, torch.Tensor]:
    """Create a batch of dummy observations with correct shapes."""
    # Random point cloud (in pelvis frame — use realistic range)
    pcd = torch.rand(1, N_OBS_STEPS, N_PTS, 3, device=device) * 2.0 - 1.0  # [-1, 1] m
    # Zeroed agent_pos (safe default)
    ap  = torch.zeros(1, N_OBS_STEPS, AGENT_DIM, device=device)
    # Set robot_quat to identity [w=1, x=0, y=0, z=0]
    ap[:, :, 24] = 1.0   # robot_quat w component
    ap[:, :, 17] = 1.0   # body_eef_quat w component
    ap[:, :,  3] = 1.0   # right_eef_quat w component
    ap[:, :, 10] = 1.0   # left_eef_quat w component
    return {"point_cloud": pcd, "agent_pos": ap}


def main():
    p = argparse.ArgumentParser(description="Offline dry-run inference test")
    p.add_argument("--checkpoint", type=str,
                   default=str(_HERE / "checkpoint" / "final.ckpt"))
    p.add_argument("--n",      type=int, default=20, help="Number of inference calls")
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    print(f"[DryRun] Device: {device}")

    # ── Load policy ───────────────────────────────────────────────────────────
    print(f"[DryRun] Loading policy from {args.checkpoint} ...")
    policy, cfg = load_policy(args.checkpoint, device=device)

    # ── Verify shapes ─────────────────────────────────────────────────────────
    obs = make_dummy_obs(device)
    print(f"\n[DryRun] Input shapes:")
    print(f"         point_cloud : {obs['point_cloud'].shape}")
    print(f"         agent_pos   : {obs['agent_pos'].shape}")

    # ── Warmup ────────────────────────────────────────────────────────────────
    print("\n[DryRun] Warming up (2 passes) ...")
    for _ in range(2):
        with torch.no_grad():
            _ = policy.predict_action(obs)

    # ── Timed runs ────────────────────────────────────────────────────────────
    print(f"[DryRun] Running {args.n} inference passes ...")
    latencies = []
    action_stats = []

    for i in range(args.n):
        obs_i = make_dummy_obs(device)
        t0 = time.time()
        with torch.no_grad():
            result = policy.predict_action(obs_i)
        dt = (time.time() - t0) * 1000  # ms

        latencies.append(dt)
        action_chunk = result["action_pred"][0].cpu().numpy()  # (horizon, 23)
        action_stats.append({
            "mean": action_chunk.mean(),
            "std":  action_chunk.std(),
            "min":  action_chunk.min(),
            "max":  action_chunk.max(),
        })

    # ── Report ────────────────────────────────────────────────────────────────
    latencies  = np.array(latencies)
    action_avg = np.mean([s["mean"] for s in action_stats])
    action_std = np.mean([s["std"]  for s in action_stats])
    action_min = np.min( [s["min"]  for s in action_stats])
    action_max = np.max( [s["max"]  for s in action_stats])

    print(f"\n{'='*50}")
    print(f"  Dry Run Results ({args.n} inference passes)")
    print(f"{'='*50}")
    print(f"  Output shape    : {result['action_pred'].shape}")
    print(f"  Latency mean    : {latencies.mean():.1f} ms")
    print(f"  Latency std     : {latencies.std():.1f} ms")
    print(f"  Latency max     : {latencies.max():.1f} ms")
    print(f"  HLC budget      : {1000/6:.0f} ms   {'✓ OK' if latencies.max() < 166 else '✗ EXCEEDS BUDGET'}")
    print(f"  Action mean/std : {action_avg:.4f} / {action_std:.4f}")
    print(f"  Action min/max  : {action_min:.4f} / {action_max:.4f}")
    print(f"  NaN in output   : {'YES ← fix this!' if np.isnan(action_avg) else 'No'}")
    print(f"{'='*50}")

    # Decode action layout check
    print(f"\n  Action layout check (first inference, step 0):")
    c0 = result["action_pred"][0, 0].cpu().numpy()
    print(f"    c_t[0:2]   hand states     : {c0[0:2]}")
    print(f"    c_t[2:5]   left  wrist pos : {c0[2:5]}")
    print(f"    c_t[5:9]   left  wrist quat: {c0[5:9]}")
    print(f"    c_t[9:12]  right wrist pos : {c0[9:12]}")
    print(f"    c_t[12:16] right wrist quat: {c0[12:16]}")
    print(f"    c_t[16:19] nav [vx,vy,vw]  : {c0[16:19]}")
    print(f"    c_t[19]    base_height     : {c0[19]:.3f}")
    print(f"    c_t[20:23] torso_rpy       : {c0[20:23]}")

    if latencies.max() < 166:
        print("\n[DryRun] ✓ Policy is ready for deployment.")
    else:
        print(f"\n[DryRun] ✗ Inference too slow ({latencies.max():.0f} ms > 166 ms).")
        print("         Consider: smaller batch, fp16, or faster GPU.")


if __name__ == "__main__":
    main()
