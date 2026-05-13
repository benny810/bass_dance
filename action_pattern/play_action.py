#!/usr/bin/env python3
"""MuJoCo player for individual PCA action primitive CSV files.

Usage:
    python action_pattern/play_action.py action_pattern/PC1_weighted_sway.csv
    python action_pattern/play_action.py action_pattern/PC1_weighted_sway.csv --speed 2
    python action_pattern/play_action.py action_pattern/PC2_lateral_step.csv --dynamics
    python action_pattern/play_action.py action_pattern/PC3_symmetric_squat.csv --dynamics --speed 0.5 --loop
"""

import argparse
import re
import struct
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


def load_trajectory(csv_path: str):
    """Load CSV trajectory. Returns (timestamps, joint_data dict)."""
    with open(csv_path, encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    timestamps = data[:, 0]
    joint_data = {}
    for i, name in enumerate(header[1:]):
        joint_data[name] = data[:, i + 1]
    return timestamps, joint_data


def build_scene():
    """Build MuJoCo model with floating base_link and position actuators."""
    xml_dir = Path(__file__).parent.parent / "casbot_band_urdf" / "xml"
    robot_xml_path = xml_dir / "CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.xml"
    robot_content = robot_xml_path.read_text()

    inner = re.search(r"<mujoco[^>]*>(.*)</mujoco>", robot_content, re.DOTALL)
    if not inner:
        print("Error: could not parse robot XML")
        sys.exit(1)
    inner_content = inner.group(1)

    # Strip sections we override
    for tag in ["compiler", "actuator", "sensor", "equality", "visual"]:
        inner_content = re.sub(
            rf'\s*<{tag}>.*?</{tag}>\s*', '\n', inner_content, flags=re.DOTALL
        )
    inner_content = re.sub(r'<camera[^/]*/>', '', inner_content)
    inner_content = re.sub(r'<body name="external_camera_body".*?</body>', '',
                           inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'<body name="target_marker_body".*?</body>', '',
                           inner_content, flags=re.DOTALL)

    inner_content = inner_content.replace(
        '<worldbody>',
        '<worldbody><body name="base_link" pos="0 0 0.8834"><freejoint/>'
    )
    inner_content = inner_content.replace('</worldbody>', '</body></worldbody>')

    # Position actuators for all 27 controlled joints
    _leg_joints = ["leg_pelvic_pitch", "leg_pelvic_roll", "leg_pelvic_yaw",
                   "leg_knee_pitch", "leg_ankle_pitch", "leg_ankle_roll"]
    _arm_joints = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                   "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll"]
    _act_lines = []
    for side in ["left", "right"]:
        for j in _leg_joints:
            _act_lines.append(
                f'    <position name="{side}_{j}_joint" joint="{side}_{j}_joint" kp="60" kv="4"/>'
            )
    _act_lines.append(
        '    <position name="waist_yaw_joint" joint="waist_yaw_joint" kp="40" kv="3"/>'
    )
    for side in ["left", "right"]:
        for j in _arm_joints:
            _act_lines.append(
                f'    <position name="{side}_{j}_joint" joint="{side}_{j}_joint" kp="40" kv="3"/>'
            )

    mesh_abs = str((xml_dir.parent / "meshes").resolve())

    scene_xml = f"""<mujoco model="action_pattern">
  <compiler angle="radian" meshdir="{mesh_abs}" texturedir="{mesh_abs}" autolimits="true"/>
  <asset>
    <texture name="checker_tex" type="2d" builtin="checker" width="512" height="512"
             rgb1="0.15 0.15 0.15" rgb2="0.85 0.85 0.85"/>
    <material name="checker_mat" texture="checker_tex" texrepeat="10 10" reflectance="0.05"/>
  </asset>
  {inner_content}
  <actuator>
{chr(10).join(_act_lines)}
  </actuator>
  <worldbody>
    <body pos="0 0 -0.8834">
      <geom name="floor" type="plane" material="checker_mat" size="0 0 1"/>
    </body>
    <light pos="5 5 4" dir="-1 -1 -1" diffuse="0.8 0.8 0.8"/>
  </worldbody>
</mujoco>"""

    scene_path = xml_dir / "_scene_action.xml"
    scene_path.write_text(scene_xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)

    ankle_ids = {}
    for side in ["left", "right"]:
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{side}_leg_ankle_roll_link")
        ankle_ids[side] = aid
    return model, ankle_ids


def _load_foot_verts():
    """Load foot STL mesh vertices (local frame relative to ankle_roll_link)."""
    mesh_dir = Path(__file__).parent.parent / "casbot_band_urdf" / "meshes"
    foot_verts = {}
    for side in ["left", "right"]:
        stl_path = mesh_dir / f"{side}_leg_ankle_roll_link.STL"
        verts = []
        with open(stl_path, "rb") as f:
            f.seek(80)
            n_tris = struct.unpack("<I", f.read(4))[0]
            for _ in range(n_tris):
                f.read(12)
                verts.append(struct.unpack("<3f", f.read(12)))
                verts.append(struct.unpack("<3f", f.read(12)))
                verts.append(struct.unpack("<3f", f.read(12)))
                f.read(2)
        foot_verts[side] = np.unique(np.array(verts, dtype=np.float32), axis=0)
    return foot_verts


def _compute_base_z(data, ankle_ids, foot_verts, floor_z):
    """Return qpos[2] so the lowest foot vertex touches floor_z."""
    foot_min = float("inf")
    for side, body_id in ankle_ids.items():
        xpos = data.xpos[body_id]
        z_row = data.xmat[body_id][6:9]
        world_z = xpos[2] + foot_verts[side] @ z_row
        foot_min = min(foot_min, world_z.min())
    return data.qpos[2] + floor_z - float(foot_min)


def build_qpos_map(model, csv_joint_names):
    """Map CSV column names to MuJoCo qpos indices and actuator indices."""
    qpos_map = {}
    ctrl_map = {}
    joint_to_actuator = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid][0])
        if jid >= 0:
            joint_to_actuator[jid] = aid

    skipped = 0
    for csv_name in csv_joint_names:
        if csv_name.endswith("_pos"):
            joint_name = csv_name[:-4]
        elif csv_name.endswith("_vel") or csv_name in ("left_foot_step", "right_foot_step"):
            skipped += 1
            continue
        elif csv_name.endswith("_joint"):
            joint_name = csv_name
        else:
            joint_name = csv_name + "_joint"

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            skipped += 1
            continue
        qpos_map[csv_name] = model.jnt_qposadr[jid]
        if jid in joint_to_actuator:
            ctrl_map[csv_name] = joint_to_actuator[jid]
    return qpos_map, ctrl_map


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo player for PCA action primitive trajectories"
    )
    parser.add_argument("csv_file", type=str, help="Path to action primitive CSV")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed: 1=normal, 2=2x, 0.5=half (default: 1)")
    parser.add_argument("--dynamics", action="store_true",
                        help="Use physics dynamics simulation (default: kinematic)")
    parser.add_argument("--loop", action="store_true",
                        help="Loop playback")
    args = parser.parse_args()

    csv_path = Path(args.csv_file)
    if not csv_path.exists():
        print(f"Error: file not found: {args.csv_file}")
        sys.exit(1)

    print(f"Loading: {csv_path.name}")
    timestamps, joint_data = load_trajectory(str(csv_path))
    n_frames = len(timestamps)
    dt_traj = timestamps[1] - timestamps[0]
    total_duration = timestamps[-1]
    print(f"  {n_frames} frames, dt={dt_traj:.4f}s ({1/dt_traj:.0f}Hz), duration={total_duration:.1f}s")

    if args.speed != 1.0:
        print(f"  Speed: {args.speed}x")

    mode_label = "dynamics" if args.dynamics else "kinematic"
    print(f"Building scene ({mode_label} mode)...")
    model, ankle_ids = build_scene()
    data = mujoco.MjData(model)
    print(f"  {model.njnt} joints, {model.nq} DOFs, {model.nu} actuators")

    csv_joint_names = list(joint_data.keys())
    qpos_map, ctrl_map = build_qpos_map(model, csv_joint_names)
    print(f"  Mapped {len(qpos_map)} joints to qpos, {len(ctrl_map)} to actuators")

    # Pre-compute qpos for each frame (joint angles only, base stays at qpos0)
    qpos_sequence = np.tile(model.qpos0.copy(), (n_frames, 1))
    for csv_name, qpos_idx in qpos_map.items():
        qpos_sequence[:, qpos_idx] = joint_data[csv_name]

    # Pre-compute floating-base Z per frame so feet stay on the ground (both modes)
    print("  Computing floating-base height (foot-ground contact)...")
    foot_verts = _load_foot_verts()
    floor_z = -0.8834
    base_z = np.zeros(n_frames)
    for frame in range(n_frames):
        data.qpos[:] = qpos_sequence[frame]
        mujoco.mj_kinematics(model, data)
        base_z[frame] = _compute_base_z(data, ankle_ids, foot_verts, floor_z)
        if (frame + 1) % 500 == 0 or frame == n_frames - 1:
            pct = (frame + 1) / n_frames * 100
            print(f"\r  ... {pct:.0f}% ({frame + 1}/{n_frames})", end="", flush=True)
    print()

    if args.dynamics:
        # Store corrected base Z into qpos_sequence for initial state each frame
        qpos_sequence[:, 2] = base_z
        ctrl_sequence = np.zeros((n_frames, model.nu))
        for csv_name, ctrl_idx in ctrl_map.items():
            ctrl_sequence[:, ctrl_idx] = joint_data[csv_name]
        n_substeps = max(1, round(dt_traj / model.opt.timestep))
        print(f"  Physics: {n_substeps} substeps/frame (timestep={model.opt.timestep:.4f}s)")
    else:
        qpos_sequence[:, 2] = base_z

    print(f"\nPlaying: {csv_path.stem} ({mode_label})")
    print("  Controls: drag to rotate, scroll to zoom, Esc to quit")

    sim_start_time = time.time()
    last_frame = -1

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10
        viewer.cam.lookat = [0, 0, -0.2]

        while viewer.is_running():
            elapsed = (time.time() - sim_start_time) * args.speed
            if args.loop:
                elapsed %= total_duration
            frame = min(int(elapsed / dt_traj), n_frames - 1)

            if frame != last_frame:
                if args.dynamics:
                    data.ctrl[:] = ctrl_sequence[frame]
                    for _ in range(n_substeps):
                        data.qpos[2] = base_z[frame]
                        mujoco.mj_step(model, data)
                else:
                    data.qpos[:] = qpos_sequence[frame]
                    mujoco.mj_forward(model, data)
                last_frame = frame

            viewer.sync()


if __name__ == "__main__":
    main()
