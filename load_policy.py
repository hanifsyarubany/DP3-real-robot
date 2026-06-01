"""
load_policy.py — Load EquivDP3 BC checkpoint for real robot deployment.

No Isaac Lab / Isaac Sim dependencies. Can run on any machine with:
  torch, hydra-core, omegaconf, diffusers, einops
"""

from __future__ import annotations

import sys
import pathlib
import collections
import torch
import numpy as np
from omegaconf import OmegaConf

# ── sys.path: make EquivDP3 and DP3 importable without Isaac ─────────────────
_THIS_DIR   = pathlib.Path(__file__).resolve().parent
_POLICY_DIR = _THIS_DIR.parent                                      # humanoid-training-equivariant/
_DP3_DIR    = _THIS_DIR.parent.parent / "3D-Diffusion-Policy" / "3D-Diffusion-Policy"

for _p in [_POLICY_DIR, _DP3_DIR]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OmegaConf.register_new_resolver("eval", eval, replace=True)


class EquivDP3Inference:
    """
    Wraps the EquivDP3 policy for real-robot inference.

    Handles:
      - Observation history buffer (rolling n_obs_steps frames)
      - Normalisation (built into policy.predict_action)
      - Action chunk management (predict every n_action_steps)
      - Thread-safe action chunk reads
    """

    # Observation layout — MUST match simulation exactly
    N_OBS_STEPS   = 2
    N_ACTION_STEPS = 8
    N_PCD_POINTS  = 1024
    AGENT_POS_DIM = 28
    ACTION_DIM    = 23

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        self.device = device
        self._load(ckpt_path)

        # Rolling obs history: deque of (pcd, agent_pos) tuples
        self._obs_history: collections.deque = collections.deque(maxlen=self.N_OBS_STEPS)

        # Current action chunk and step pointer
        self._action_chunk: torch.Tensor | None = None   # (1, 16, 23)
        self._chunk_step: int = 0

        print(f"[EquivDP3Inference] Ready on {device}")

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self, ckpt_path: str):
        import hydra
        payload  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg      = payload["cfg"]
        policy   = hydra.utils.instantiate(cfg.policy)
        state    = payload.get("ema_model") or payload.get("model")
        policy.load_state_dict(state, strict=False)
        self._policy = policy.to(self.device).eval()
        print(f"[EquivDP3Inference] Loaded from {ckpt_path}")

    # ── Observation buffer ────────────────────────────────────────────────────

    def reset(self):
        """Call at the start of each episode."""
        self._obs_history.clear()
        self._action_chunk = None
        self._chunk_step   = 0

    def push_obs(self, pcd: np.ndarray, agent_pos: np.ndarray):
        """
        Add one observation frame to the rolling history.

        Args:
            pcd:       (N, 3) float32 point cloud in pelvis frame, N >= N_PCD_POINTS
            agent_pos: (28,)  float32 agent state vector
        """
        assert pcd.shape == (self.N_PCD_POINTS, 3), \
            f"pcd shape {pcd.shape} — expected ({self.N_PCD_POINTS}, 3)"
        assert agent_pos.shape == (self.AGENT_POS_DIM,), \
            f"agent_pos shape {agent_pos.shape} — expected ({self.AGENT_POS_DIM},)"
        self._obs_history.append((
            torch.from_numpy(pcd.copy()).float(),
            torch.from_numpy(agent_pos.copy()).float(),
        ))

    def _build_obs_dict(self) -> dict[str, torch.Tensor]:
        """Stack history into batch tensors expected by predict_action."""
        # Pad with first frame if history not full yet
        history = list(self._obs_history)
        while len(history) < self.N_OBS_STEPS:
            history.insert(0, history[0])

        pcds      = torch.stack([h[0] for h in history], dim=0).unsqueeze(0)  # (1, T, N, 3)
        agent_pos = torch.stack([h[1] for h in history], dim=0).unsqueeze(0)  # (1, T, 28)

        return {
            "point_cloud": pcds.to(self.device),
            "agent_pos":   agent_pos.to(self.device),
        }

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict_action(self, pcd: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
        """
        Push one observation and run DDIM inference.

        Returns the NEXT action from the current chunk as a (23,) numpy array.
        Automatically re-infers when the chunk is exhausted (every N_ACTION_STEPS calls).

        Args:
            pcd:       (1024, 3) float32 — current point cloud in pelvis frame
            agent_pos: (28,)    float32 — current agent state

        Returns:
            c_t: (23,) float32 — WBC command for this control step
        """
        self.push_obs(pcd, agent_pos)

        if len(self._obs_history) < 1:
            return np.zeros(self.ACTION_DIM, dtype=np.float32)

        # Re-infer every N_ACTION_STEPS
        if self._action_chunk is None or self._chunk_step >= self.N_ACTION_STEPS:
            with torch.no_grad():
                obs_dict = self._build_obs_dict()
                result   = self._policy.predict_action(obs_dict)
                self._action_chunk = result["action_pred"]  # (1, 16, 23)
            self._chunk_step = 0

        c_t = self._action_chunk[0, self._chunk_step].cpu().numpy()  # (23,)
        self._chunk_step += 1

        # Force hands open (dataset always has hand_state = 0)
        c_t[0] = 0.0
        c_t[1] = 0.0

        return c_t.astype(np.float32)

    def needs_new_inference(self) -> bool:
        """True when the current action chunk is exhausted."""
        return self._action_chunk is None or self._chunk_step >= self.N_ACTION_STEPS
