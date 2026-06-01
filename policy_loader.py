"""
policy_loader.py
================
Load EquivDP3 from a training checkpoint with zero Isaac Sim / Arena dependency.

The checkpoint stores:
    payload["cfg"]       – OmegaConf config used during training
    payload["ema_model"] – EMA-averaged model state dict (preferred)
    payload["model"]     – raw model state dict (fallback)

Usage
-----
    from policy_loader import load_policy
    policy, cfg = load_policy("checkpoint/final.ckpt", device="cuda")
    # policy is EquivDP3, ready for .predict_action()
"""

from __future__ import annotations

import pathlib
import sys

import torch
from omegaconf import OmegaConf

# ── sys.path: DP3 source and equivariant training root ────────────────────────
_DEPLOY_DIR = pathlib.Path(__file__).resolve().parent
_EQUIV_DIR  = _DEPLOY_DIR.parent
_DP3_DIR    = _EQUIV_DIR.parent / "3D-Diffusion-Policy" / "3D-Diffusion-Policy"

for _p in [_DP3_DIR, _EQUIV_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Register OmegaConf resolver used in configs (safe to call multiple times)
try:
    OmegaConf.register_new_resolver("eval", eval)
except Exception:
    pass


# ── Public API ─────────────────────────────────────────────────────────────────

def load_policy(ckpt_path: str, device: str = "cuda"):
    """
    Load EquivDP3 from checkpoint.

    Parameters
    ----------
    ckpt_path : str
        Path to .ckpt file (Stage 1 BC or Stage 2 DPPO).
    device : str
        PyTorch device string, e.g. "cuda", "cuda:0", "cpu".

    Returns
    -------
    policy : EquivDP3
        Policy in eval mode on the specified device.
    cfg : OmegaConf DictConfig
        Training config stored in the checkpoint (contains n_obs_steps,
        horizon, n_action_steps, shape_meta, etc.)
    """
    import hydra

    ckpt_path = pathlib.Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    payload = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg     = payload["cfg"]

    policy = hydra.utils.instantiate(cfg.policy)

    state = payload.get("ema_model") or payload.get("model")
    if state is None:
        raise KeyError("Checkpoint has neither 'ema_model' nor 'model' key.")

    policy.load_state_dict(state, strict=False)
    policy = policy.to(device).eval()

    print(f"[PolicyLoader] EquivDP3 loaded from {ckpt_path}")
    print(f"[PolicyLoader]   n_obs_steps   = {cfg.n_obs_steps}")
    print(f"[PolicyLoader]   horizon       = {cfg.horizon}")
    print(f"[PolicyLoader]   n_action_steps= {cfg.n_action_steps}")
    print(f"[PolicyLoader]   device        = {device}")

    return policy, cfg
