"""
scripts/test_homie_sim.py
=========================
Test / teleoperate HOMIE v2 locomotion in Galileo G1 Isaac Sim environment.

Two modes
---------
  --teleop    Keyboard control (default when running GUI). Hold keys to move.
  --auto      Predefined sequence: stand → walk → turn → stop (good for headless).

Keyboard bindings (Se2Keyboard — active in Isaac Sim viewport):
  Arrow Up   / Numpad 8  : walk forward
  Arrow Down / Numpad 2  : walk backward
  Arrow Left / Numpad 4  : strafe right
  Arrow Right/ Numpad 6  : strafe left
  Z          / Numpad 7  : turn left  (yaw +)
  X          / Numpad 9  : turn right (yaw -)
  L                      : stop / reset velocity to zero
  ESC                    : quit

Usage
-----
    # GUI + keyboard teleoperation (recommended)
    ./isaaclab.sh -p .../test_homie_sim.py --teleop

    # GUI + automated sequence
    ./isaaclab.sh -p .../test_homie_sim.py --auto

    # Headless + automated sequence
    ./isaaclab.sh -p .../test_homie_sim.py --auto --headless
"""

from __future__ import annotations

import sys, os, argparse, tempfile

# ── pinocchio before AppLauncher (libhpp-fcl / assimp ABI fix) ────────────────
import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Test / teleop HOMIE v2 in Galileo G1 sim")
parser.add_argument("--teleop", action="store_true",
                    help="Keyboard teleoperation mode (requires GUI)")
parser.add_argument("--auto",   action="store_true",
                    help="Run automated stand/walk/turn/stop sequence")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Default: teleop if neither flag given
if not args_cli.teleop and not args_cli.auto:
    args_cli.teleop = True

app_launcher   = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ── Post-launch imports ───────────────────────────────────────────────────────
import numpy as np
import torch
import tqdm
import gymnasium as gym

import isaaclab.sim as sim_utils
from isaaclab.devices.keyboard.se2_keyboard import Se2Keyboard, Se2KeyboardCfg

# ── Cache-first retrieve_file_path (avoids re-downloading Nucleus assets) ─────
import isaaclab.utils.assets as _assets_mod
_orig_retrieve = _assets_mod.retrieve_file_path

def _retrieve_cached(path: str, download_dir=None, force_download: bool = True) -> str:
    _tmp = tempfile.gettempdir()
    fname = path.split("/")[-1]
    for cand in [os.path.join(_tmp, fname), os.path.join(_tmp, "urdf", fname)]:
        if os.path.isfile(cand):
            return cand
    return _orig_retrieve(path, download_dir=download_dir, force_download=False)

_assets_mod.retrieve_file_path = _retrieve_cached
for _mn in list(sys.modules):
    _m = sys.modules[_mn]
    if hasattr(_m, "retrieve_file_path") and getattr(_m, "retrieve_file_path") is _orig_retrieve:
        setattr(_m, "retrieve_file_path", _retrieve_cached)

import isaaclab_tasks       # noqa: F401
import isaaclab_mimic.envs  # noqa: F401

_ARENA_ROOT = "/workspaces/isaaclab_arena"
if _ARENA_ROOT not in sys.path:
    sys.path.insert(0, _ARENA_ROOT)

# ── WBC action helpers ────────────────────────────────────────────────────────
# Arms at natural hang position (matches training idle pose)
_WRIST_L = np.array([0.201,  0.145, 0.101, 1.0,  0.010, -0.008, -0.011], dtype=np.float32)
_WRIST_R = np.array([0.201, -0.145, 0.101, 1.0, -0.010, -0.008, -0.011], dtype=np.float32)

def make_action(vx=0.0, vy=0.0, vyaw=0.0, height=0.74) -> np.ndarray:
    """Build a 23D WBC action with given locomotion commands."""
    return np.array([
        0.0, 0.0,          # hand states (always open)
        *_WRIST_L,         # left  wrist pose [pos+quat] in pelvis frame
        *_WRIST_R,         # right wrist pose
        vx, vy, vyaw,      # nav_cmd
        height,            # base_height
        0.0, 0.0, 0.0,     # torso_rpy
    ], dtype=np.float32)


# ── Environment builder ───────────────────────────────────────────────────────

def build_env(device_str: str):
    """Build Galileo G1 locomanip env — same scene as training/eval."""
    from isaaclab_arena.assets.asset_registry import AssetRegistry
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.tasks.dummy_task import DummyTask
    from isaaclab_arena.utils.pose import Pose

    reg = AssetRegistry()
    background  = reg.get_asset_by_name("galileo_locomanip")()
    pick_object = reg.get_asset_by_name("brown_box")()
    blue_bin    = reg.get_asset_by_name("blue_sorting_bin")()
    embodiment  = reg.get_asset_by_name("g1_wbc_pink")(enable_cameras=False)

    # Exact initial poses from GalileoG1LocomanipPickAndPlaceEnvironment
    pick_object.set_initial_pose(
        Pose(position_xyz=(0.5785, 0.18, 0.0707), rotation_wxyz=(0., 0., 1., 0.)))
    blue_bin.set_initial_pose(
        Pose(position_xyz=(-0.2450, -1.6272, -0.2641), rotation_wxyz=(0., 0., 0., 1.)))
    embodiment.set_initial_pose(
        Pose(position_xyz=(0., 0.18, 0.), rotation_wxyz=(1., 0., 0., 0.)))

    arena_env = IsaacLabArenaEnvironment(
        name="homie_test",
        embodiment=embodiment,
        scene=Scene(assets=[background, pick_object, blue_bin]),
        task=DummyTask(),
    )
    arena_args = argparse.Namespace(
        device=device_str, num_envs=1,
        disable_fabric=False, mimic=False,
        object="brown_box", embodiment="g1_wbc_pink",
        teleop_device=None, enable_cameras=False,
    )
    builder = ArenaEnvBuilder(arena_env, arena_args)
    task_name, env_cfg = builder.build_registered()
    env_cfg.scene.num_envs = 1
    env_cfg.observations.policy.concatenate_terms = False

    env = gym.make(task_name, cfg=env_cfg)
    print(f"\n[HomieTest] Environment ready: {task_name}")
    return env.unwrapped


# ── Teleoperation mode ────────────────────────────────────────────────────────

def run_teleop(env, device_str: str):
    """Interactive keyboard teleoperation loop."""
    kb = Se2Keyboard(Se2KeyboardCfg(
        v_x_sensitivity=0.30,      # max forward speed (m/s)
        v_y_sensitivity=0.20,      # max strafe speed  (m/s)
        omega_z_sensitivity=0.30,  # max yaw rate      (rad/s)
        sim_device=device_str,
    ))

    _quit = [False]
    kb.add_callback("ESCAPE", lambda: _quit.__setitem__(0, True))

    print("\n" + "=" * 55)
    print("  HOMIE Keyboard Teleoperation")
    print("=" * 55)
    print(kb)           # prints key bindings
    print("\n  ESC  : quit")
    print("  L    : stop (zero velocity)")
    print("=" * 55 + "\n")

    env.reset()
    kb.reset()

    step = 0
    while simulation_app.is_running() and not _quit[0]:
        cmd = kb.advance()   # (3,) tensor: [vx, vy, vyaw]
        vx, vy, vyaw = float(cmd[0]), float(cmd[1]), float(cmd[2])

        action = torch.from_numpy(make_action(vx, vy, vyaw)).unsqueeze(0).to(device_str)
        with torch.inference_mode():
            env.step(action)

        if step % 50 == 0:   # print every ~1 s
            pos = env.scene["robot"].data.root_link_pos_w[0, :3].cpu().numpy()
            print(f"\r  cmd: vx={vx:+.2f} vy={vy:+.2f} vyaw={vyaw:+.2f}  "
                  f"pos: [{pos[0]:+.2f} {pos[1]:+.2f} {pos[2]:+.2f}]", end="", flush=True)
        step += 1

    print("\n[HomieTest] Teleoperation ended.")


# ── Automated sequence mode ───────────────────────────────────────────────────

def run_auto(env, device_str: str):
    """Predefined stand → walk → turn → stop sequence with pass/fail check."""
    env.reset()

    def get_pos():
        return env.scene["robot"].data.root_link_pos_w[0, :3].cpu().numpy()

    def step_n(n, vx=0.0, vy=0.0, vyaw=0.0, label=""):
        act = torch.from_numpy(make_action(vx, vy, vyaw)).unsqueeze(0).to(device_str)
        for _ in tqdm.tqdm(range(n), desc=label, ncols=70):
            with torch.inference_mode():
                env.step(act)

    pos0 = get_pos()
    print(f"\n  Start pos : {pos0.round(3)}")

    step_n(100,            label="Stand still")
    p1    = get_pos()
    drift = float(np.linalg.norm(p1[:2] - pos0[:2]))
    print(f"  After stand : {p1.round(3)}  drift={drift:.3f} m")

    step_n(200, vx=0.3,   label="Walk forward (vx=0.3)")
    p2    = get_pos()
    fwd   = float(p2[0] - p1[0])
    print(f"  After walk  : {p2.round(3)}  X_fwd={fwd:+.3f} m")

    step_n(150, vyaw=0.3, label="Turn left   (vyaw=0.3)")
    p3    = get_pos()
    print(f"  After turn  : {p3.round(3)}")

    step_n(100,            label="Stop")
    p4    = get_pos()
    print(f"  Final pos   : {p4.round(3)}")

    stand_ok  = drift < 0.20
    walk_ok   = fwd   > 0.15
    z_drop    = float(pos0[2] - p2[2])
    height_ok = z_drop < 0.30

    print("\n" + "=" * 55)
    print("  HOMIE Sim Validation")
    print("=" * 55)
    print(f"  Stand stability : {'OK' if stand_ok  else 'FAIL'}  drift={drift:.3f} m  (< 0.20)")
    print(f"  Walk forward    : {'OK' if walk_ok   else 'FAIL'}  X={fwd:+.3f} m  (> 0.15)")
    print(f"  Did not fall    : {'OK' if height_ok else 'FAIL'}  Z drop={z_drop:.3f} m  (< 0.30)")
    print("=" * 55)
    if stand_ok and walk_ok and height_ok:
        print("  HOMIE validated — safe to proceed to real robot.")
    else:
        print("  Check failed — review before real robot deployment.")
    print("=" * 55 + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Dome light for GUI visibility
    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(1.0, 1.0, 1.0))
    light_cfg.func("/World/DomeLight", light_cfg)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    env = build_env(device_str)

    try:
        if args_cli.teleop:
            run_teleop(env, device_str)
        else:
            run_auto(env, device_str)
    finally:
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
