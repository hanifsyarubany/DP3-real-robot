# EquivDP3 Real Robot Deployment Guide
## Unitree G1 — Pick-and-Place Loco-Manipulation

---

## Overview

This guide walks through deploying the trained EquivDP3 BC policy to the real Unitree G1 robot from a cold start to a live run.  Follow every section in order — each step is a safety gate before the next.

### System Architecture

```
Companion PC (Ubuntu 22.04, GPU)
  │
  ├── RealSense D435i  ──────→  pelvis-frame point cloud (1024 pts)
  │
  ├── Unitree SDK2  ──────────→  read: joint angles, IMU, base pose
  │                              write: arm joints (LowCmd), loco (LocoClient)
  │
  └── EquivDP3 (6 Hz)
        │
        ├── VNN encoder + DDIM → 23D WBC action chunk
        ├── PINK IK            → arm joint targets (indices 15–28)
        └── LocoClient         → sport-mode velocity commands (legs handled internally)

Unitree G1 (192.168.123.x)
  ├── Legs: Unitree internal WBC (sport mode)
  └── Arms: PD control from companion PC LowCmd
```

---

## Prerequisites

### Hardware
- [ ] Unitree G1 robot (29 DoF, firmware ≥ 1.0)
- [ ] Companion PC: Ubuntu 22.04, NVIDIA GPU (RTX 3060+), 16 GB RAM
- [ ] Intel RealSense D435i (USB 3.0)
- [ ] Ethernet cable: companion PC ↔ G1 network switch
- [ ] Emergency stop method: wired remote or keyboard `q + Enter`

### Software (Companion PC)
```bash
pip install unitree-sdk2py pyrealsense2 pin-pink pinocchio torch torchvision
pip install hydra-core omegaconf diffusers termcolor
```

Check SDK2 is installed:
```bash
python -c "from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('SDK2 OK')"
```

### Network Setup
```bash
# Set companion PC IP on the G1 subnet
sudo ip addr add 192.168.123.100/24 dev enp2s0
sudo ip link set enp2s0 up

# Verify connectivity
ping 192.168.123.1    # G1 router
```

---

## Step 1 — Cold Start: Power On Robot

1. Place robot on a flat, clear surface (≥ 2 m clearance in all directions).
2. Lie robot flat on its back.
3. Power on: press battery button until LEDs turn solid green.
4. Wait ~30 s for internal boot sequence.
5. Robot will beep twice when ready.

> **Do not attempt to stand the robot yet.**

---

## Step 2 — Check Robot State (No Commands Sent)

Verify SDK2 communication and sensor readings before any motion.

```bash
cd real-robot-deployment
python scripts/check_robot_state.py --network enp2s0
```

**Expected output** (robot lying flat):
```
q_arm: [ 0.000  0.000  0.000  0.000 ... ]   ← all near zero
IMU rpy: [  0.1   0.0  90.0]°               ← lying on back: roll ≈ 0, pitch ≈ 0
base_pos: [  0.00   0.00   0.10]             ← base ~10 cm off floor
age: 3ms                                     ← MUST be < 20 ms
```

**If `age > 100 ms`:** Network connection failing.  Check:
- Ethernet cable is plugged in
- IP address is set correctly (`ip addr show enp2s0`)
- G1 is powered and booted

---

## Step 3 — Verify Policy Loads (No Robot Needed)

Run the offline dry-run to confirm the checkpoint loads and inference is fast enough.

```bash
python scripts/dry_run.py --checkpoint checkpoint/final.ckpt --n 20 --device cuda
```

**Expected output:**
```
  Latency mean    : ~70 ms
  Latency max     : < 166 ms   ✓ OK
  NaN in output   : No
  Action layout check:
    c_t[16:19] nav [vx,vy,vw]  : [-0.0xx  0.0xx  0.0xx]
    c_t[19]    base_height     : 0.7xx
```

**If latency > 166 ms:** Use `--device cuda:0`, check GPU is available with `nvidia-smi`.

---

## Step 4 — Calibrate Camera Extrinsics

> **This step is critical.** Wrong extrinsics = wrong point cloud frame = policy failure.

The policy was trained on point clouds in the **pelvis frame**.  You must calibrate the transform from camera to pelvis frame.

### Measurement (manual)
With robot standing upright (height 0.74 m):
1. Measure camera mount position relative to pelvis origin (in metres).
2. Measure camera orientation (rotation from camera axes to pelvis axes).

Update `DEFAULT_T_PELVIS_CAM` in [pcd_pipeline.py](pcd_pipeline.py):
```python
DEFAULT_T_PELVIS_CAM = make_T_pelvis_cam(
    translation_xyz=[x, y, z],      # camera position in pelvis frame (metres)
    rotation_wxyz=[w, x, y, z],     # camera orientation as quaternion
)
```

### Verify calibration
```bash
python scripts/check_pcd.py
```

**Expected point cloud statistics** (robot standing, table at ~0.8 m height):
```
Z min/max: [-0.05,  1.20]   ← floor below pelvis, ceiling above
Mean Z:    ~0.60             ← most points around table height
```
If Z values are inverted or wildly off, the rotation is wrong.

---

## Step 5 — Stand Robot Up (No Policy)

> **Have one person at the emergency stop.** Stand back ≥ 1 m.

```bash
python deploy.py --network enp2s0 --dry-run
```

Wait for the "Standing robot up ..." message, then **Ctrl+C** immediately after the robot stands.

Verify:
- Robot stands at correct height (~0.74 m)
- Arms hang naturally at sides
- No unusual motor noise

> If the robot falls or acts erratically: press `q + Enter` or use the wired remote.

---

## Step 6 — Dry Run With Robot Connected

Run policy inference + IK + safety checks **without sending commands** to the robot (arms are NOT moved).

```bash
python deploy.py --network enp2s0 --dry-run --max-steps 100
```

Watch the terminal output:
- No safety violations triggered
- Arm IK solving without errors
- `chunk_step` incrementing smoothly
- No NaN warnings

---

## Step 7 — First Live Run With Low Gains

> **Clear a 3 m radius around the robot. Have emergency stop ready.**

Use `--low-kp` to start with conservative arm PD gains (kp=20 instead of 80).  Arms will move slowly and compliantly.

```bash
python deploy.py --network enp2s0 --checkpoint checkpoint/final.ckpt --low-kp --max-steps 200
```

**Observe for the first 10 seconds:**
- [ ] Arms move smoothly (no jerking)
- [ ] Robot stays balanced while arms move
- [ ] Navigation commands are small (robot standing still or slow walk)
- [ ] No safety stop triggered

**Press `q + Enter` to stop cleanly after observing.**

---

## Step 8 — Full Deployment

Once low-gain run looks safe, proceed with nominal gains:

```bash
python deploy.py --network enp2s0 --checkpoint checkpoint/final.ckpt --max-steps 2000
```

Set up the task scenario:
- Place pick object at nominal training position (~0.5 m in front)
- Place bin target at ~1.0 m distance

---

## Emergency Procedures

### Keyboard E-Stop (primary)
In the deploy terminal:
```
q + Enter
```
Robot will damp all motors and stop locomotion within ~0.5 s.

### Hardware E-Stop (backup)
Press the physical emergency stop on the Unitree wired remote.

### Power Cut (last resort)
Pull the battery connector.  Robot will collapse — keep clear.

### If robot falls
1. Cut power or hit E-stop
2. Do not attempt to catch the robot
3. Check for damage before powering back on

---

## Tuning Guide

### Arm gains (`--low-kp` vs default)
| Stage | kp | kd | Behavior |
|---|---|---|---|
| First tests | 20 | 1 | Slow, compliant, safe |
| Validated runs | 80 | 2 | Normal tracking |
| Fast tasks | 120 | 3 | Aggressive tracking |

Set in [robot_interface.py](robot_interface.py): `ARM_KP`, `ARM_KD`.

### Safety delta limit
In [safety_monitor.py](safety_monitor.py): `ARM_MAX_DELTA_PER_STEP` (default 0.1 rad).
Reduce to 0.05 rad for extra caution on first runs.

### Locomotion speed limits
In [robot_interface.py](robot_interface.py): `MAX_VX`, `MAX_VY`, `MAX_VYAW`.
Default: 0.5 m/s, 0.3 m/s, 0.5 rad/s.

---

## Observation Space Reference

The policy requires these **exact** input shapes:

| Key | Shape | Frame | Source |
|---|---|---|---|
| `point_cloud` | `(1, 2, 1024, 3)` | Pelvis frame | RealSense → `pcd_pipeline.py` |
| `agent_pos` | `(1, 2, 28)` | Mixed (see below) | FK + IMU → `obs_builder.py` |

### agent_pos (28D) layout:
```
[0:3]   right_eef_pos   right wrist_yaw_link XYZ     in PELVIS frame  (FK, fixed base)
[3:7]   right_eef_quat  right wrist_yaw_link [w,x,y,z] in PELVIS frame
[7:10]  left_eef_pos    left  wrist_yaw_link XYZ     in PELVIS frame
[10:14] left_eef_quat   left  wrist_yaw_link [w,x,y,z] in PELVIS frame
[14:17] body_eef_pos    = copy of right_eef_pos       (sim convention, not a separate link)
[17:21] body_eef_quat   = copy of right_eef_quat
[21:24] robot_pos       pelvis XYZ                   in WORLD frame   (SDK2 state estimator)
[24:28] robot_quat      pelvis [w,x,y,z]             in WORLD frame   (IMU)
```

> **Frame note:** EEF positions [0:21] are in the **pelvis frame** — this matches
> `get_target_link_position_in_target_frame(target_frame_name="pelvis")` used during training.
> Only `robot_pos` and `robot_quat` [21:28] are in world frame.

### Action (23D) layout:
```
[0]     left_hand_state   (0 = open — always ignored)
[1]     right_hand_state  (0 = open — always ignored)
[2:5]   left_wrist_pos    in pelvis frame → PINK IK
[5:9]   left_wrist_quat   [w,x,y,z] pelvis frame → PINK IK
[9:12]  right_wrist_pos   in pelvis frame → PINK IK
[12:16] right_wrist_quat  [w,x,y,z] pelvis frame → PINK IK
[16:19] nav_cmd           [vx, vy, vyaw] → LocoClient.Move()
[19]    base_height_cmd   metres → LocoClient.SetStandHeight()
[20:23] torso_rpy         (logged only)
```

---

## File Reference

| File | Purpose |
|---|---|
| [deploy.py](deploy.py) | Main entry point — 50 Hz control loop |
| [policy_loader.py](policy_loader.py) | Load EquivDP3 from checkpoint |
| [robot_interface.py](robot_interface.py) | Unitree SDK2 wrapper |
| [pcd_pipeline.py](pcd_pipeline.py) | RealSense → pelvis-frame PCD |
| [arm_ik.py](arm_ik.py) | PINK IK: wrist targets → joint angles |
| [obs_builder.py](obs_builder.py) | Build 28D agent_pos from FK + IMU |
| [safety_monitor.py](safety_monitor.py) | Joint limits, delta checks, E-stop |
| [scripts/dry_run.py](scripts/dry_run.py) | Offline inference latency test |
| [scripts/check_robot_state.py](scripts/check_robot_state.py) | Live robot state monitor |
| [scripts/check_pcd.py](scripts/check_pcd.py) | PCD pipeline calibration check |
| [checkpoint/final.ckpt](checkpoint/final.ckpt) | BC policy checkpoint |
| [config/g1_homie_v2.yaml](config/g1_homie_v2.yaml) | LLC config (future Option 2 use) |
