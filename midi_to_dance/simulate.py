#!/usr/bin/env python3
"""MuJoCo simulation player for robot dance trajectories with synced MIDI audio.

Usage:
    python midi_to_dance/simulate.py output.csv mid/yellow贝斯.mid
    python midi_to_dance/simulate.py output.csv mid/yellow贝斯.mid --slow 0.5
    python midi_to_dance/simulate.py output.csv mid/yellow贝斯.mid --dynamics
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


# ---------------------------------------------------------------------------
# Quaternion math helpers
# ---------------------------------------------------------------------------

def _quat_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def _apply_base_rotation(qpos, delta_pitch, delta_roll):
    """Apply pitch (Y-axis) and roll (X-axis) rotation to the base_link
    quaternion stored at qpos[3:7] (w,x,y,z)."""
    cp = np.cos(delta_pitch / 2.0)
    sp = np.sin(delta_pitch / 2.0)
    q_pitch = np.array([cp, 0.0, sp, 0.0])

    cr = np.cos(delta_roll / 2.0)
    sr = np.sin(delta_roll / 2.0)
    q_roll = np.array([cr, sr, 0.0, 0.0])

    q_delta = _quat_multiply(q_pitch, q_roll)
    q_current = qpos[3:7].copy()
    q_new = _quat_multiply(q_delta, q_current)
    q_new /= np.linalg.norm(q_new)
    qpos[3:7] = q_new


# ---------------------------------------------------------------------------
# ZMP / CoM balance helpers
# ---------------------------------------------------------------------------

def _compute_robot_com(model, data):
    """CoM of the robot subtree (base_link and descendants) in world frame.

    Uses MuJoCo subtree_com which is computed during mj_kinematics / mj_forward.
    """
    base_link_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    if base_link_id >= 0:
        return data.subtree_com[base_link_id].copy()
    return data.subtree_com[0].copy()


def _compute_support_center(data, left_ankle_id, right_ankle_id):
    """Support-polygon centre: midpoint of the two ankle-link world positions
    projected to the ground plane."""
    left = data.xpos[left_ankle_id][:2]
    right = data.xpos[right_ankle_id][:2]
    return (left + right) / 2.0


def load_trajectory(csv_path: str, fps: float = 50.0):
    """Load CSV trajectory into (timestamps, joint_data dict, joint_names).

    Handles two formats:
    - With timestamp column: "timestamp, joint_1, ..."
    - Without: "joint_1, joint_2, ..." — generates synthetic timestamps at `fps`.
    """
    with open(csv_path, encoding="utf-8-sig") as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

    has_timestamp = header[0].strip() == "timestamp"
    if has_timestamp:
        timestamps = data[:, 0]
        joint_names = header[1:]
        joint_offset = 1
    else:
        dt = 1.0 / fps
        timestamps = np.arange(len(data)) * dt
        joint_names = header
        joint_offset = 0

    joint_data = {}
    for i, name in enumerate(joint_names):
        joint_data[name] = data[:, i + joint_offset]
    return timestamps, joint_data, joint_names


def synthesize_midi_audio(midi_path: str, sample_rate: int = 44100):
    """Synthesize audio from MIDI file using additive synthesis.

    Returns (audio_samples, sample_rate).
    audio_samples is a 1D float32 numpy array in [-1, 1].
    """
    import mido

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Extract tempo map and notes from all tracks
    tempo = 500000  # default 120 BPM
    tempo_map = [(0, tempo)]
    notes = []  # (start_tick, end_sec, midi_note, velocity)

    # Track 0 typically contains tempo meta-events; subsequent tracks hold notes
    for track in mid.tracks:
        abs_tick = 0
        active = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo = msg.tempo
                tempo_map.append((abs_tick, tempo))
            elif msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = (abs_tick, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in active:
                    start_tick, vel = active.pop(msg.note)
                    notes.append((start_tick, abs_tick, msg.note, vel))

    # Convert ticks to seconds using tempo map
    def ticks_to_seconds(tick):
        sec = 0.0
        prev_tick = 0
        prev_tempo = 500000
        for t, tm in tempo_map:
            if t > tick:
                break
            sec += (t - prev_tick) / ticks_per_beat * prev_tempo / 1_000_000
            prev_tick = t
            prev_tempo = tm
        sec += (tick - prev_tick) / ticks_per_beat * prev_tempo / 1_000_000
        return sec

    total_duration = 0.0
    note_segments = []
    for start_tick, end_tick, midi_note, velocity in notes:
        t_start = ticks_to_seconds(start_tick)
        t_end = ticks_to_seconds(end_tick)
        if t_end > total_duration:
            total_duration = t_end
        note_segments.append((t_start, t_end, midi_note, velocity / 127.0))

    total_samples = int(total_duration * sample_rate) + sample_rate  # 1s padding
    audio = np.zeros(total_samples, dtype=np.float32)

    for t_start, t_end, midi_note, vel in note_segments:
        freq = 440.0 * 2 ** ((midi_note - 69) / 12.0)
        i_start = int(t_start * sample_rate)
        i_end = int(t_end * sample_rate)

        n_samples = i_end - i_start
        if n_samples <= 0:
            continue

        t = np.arange(n_samples) / sample_rate
        dur = t_end - t_start

        # ADSR envelope
        attack = min(0.02, dur * 0.1)
        decay = min(0.05, dur * 0.2)
        release = min(0.08, dur * 0.3)
        sustain_level = 0.6

        env = np.ones(n_samples)
        i_attack = int(attack * sample_rate)
        i_decay = int((attack + decay) * sample_rate)
        i_release_start = max(0, n_samples - int(release * sample_rate))

        if i_attack > 0:
            env[:i_attack] = np.linspace(0, 1, i_attack)
        if i_decay > i_attack:
            env[i_attack:i_decay] = np.linspace(1, sustain_level, i_decay - i_attack)
        env[i_decay:i_release_start] = sustain_level
        if i_release_start < n_samples:
            env[i_release_start:] = np.linspace(sustain_level, 0, n_samples - i_release_start)

        # Bass-like tone: fundamental + 3 harmonics
        wave = (0.7 * np.sin(2 * np.pi * freq * t) +
                0.2 * np.sin(2 * np.pi * freq * 2 * t) +
                0.07 * np.sin(2 * np.pi * freq * 3 * t) +
                0.03 * np.sin(2 * np.pi * freq * 4 * t))

        audio[i_start:i_end] += wave * env * vel * 0.8

    # Normalize
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak * 1.1

    return audio.astype(np.float32), sample_rate, total_duration


def main():
    parser = argparse.ArgumentParser(
        description="MuJoCo simulation of robot dance with synced MIDI audio"
    )
    parser.add_argument("csv_file", type=str, help="CSV trajectory file")
    parser.add_argument("midi_file", type=str, help="MIDI file for audio")
    parser.add_argument("--slow", type=float, default=1.0,
                        help="Playback speed factor (default: 1.0)")
    parser.add_argument("--no-audio", action="store_true",
                        help="Disable audio playback")
    parser.add_argument("--fps", type=float, default=50.0,
                        help="Frames per second for CSV without timestamp column (default: 50)")
    parser.add_argument("--dynamics", action="store_true",
                        help="Run physics dynamics simulation (default: kinematic)")
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory: {args.csv_file}")
    timestamps, joint_data, csv_joint_names = load_trajectory(args.csv_file, fps=args.fps)
    if len(timestamps) < 2:
        print("Error: trajectory needs at least 2 frames")
        sys.exit(1)
    dt_traj = timestamps[1] - timestamps[0]
    if dt_traj <= 0:
        print(f"Error: trajectory dt must be positive, got {dt_traj}")
        sys.exit(1)
    print(f"  {len(timestamps)} frames, dt={dt_traj:.3f}s, duration={timestamps[-1]:.1f}s")

    # Synthesize audio
    audio = None
    sample_rate = 44100
    if not args.no_audio:
        print(f"Synthesizing audio from: {args.midi_file}")
        audio, sample_rate, audio_duration = synthesize_midi_audio(args.midi_file)
        print(f"  {audio_duration:.1f}s audio, {sample_rate} Hz")

    # Build scene XML with floating base_link, checkerboard floor, and position actuators.
    # The robot XML is embedded inline so we can wrap its bodies in a free-floating base_link.
    import re

    xml_dir = Path(__file__).parent.parent / "casbot_band_urdf" / "xml"
    robot_xml_path = xml_dir / "CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.xml"
    robot_content = robot_xml_path.read_text()

    # Extract inner content from robot XML (<mujoco> wrapper -> inner content)
    inner = re.search(r"<mujoco[^>]*>(.*)</mujoco>", robot_content, re.DOTALL)
    if not inner:
        print("Error: could not parse robot XML")
        sys.exit(1)
    inner_content = inner.group(1)

    # Strip sections we override ourselves
    inner_content = re.sub(r'\s*<compiler[^/]*/>\s*', '\n', inner_content)
    inner_content = re.sub(r'\s*<actuator>.*?</actuator>\s*', '\n', inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'\s*<sensor>.*?</sensor>\s*', '\n', inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'\s*<equality>.*?</equality>\s*', '\n', inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'\s*<visual>.*?</visual>\s*', '\n', inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'<camera[^/]*/>', '', inner_content)
    inner_content = re.sub(r'<body name="external_camera_body".*?</body>', '', inner_content, flags=re.DOTALL)
    inner_content = re.sub(r'<body name="target_marker_body".*?</body>', '', inner_content, flags=re.DOTALL)

    # Wrap all worldbody children in a floating base_link body
    inner_content = inner_content.replace(
        '<worldbody>',
        '<worldbody><body name="base_link" pos="0 0 0.8834"><freejoint/>'
    )
    inner_content = inner_content.replace(
        '</worldbody>',
        '</body></worldbody>'
    )

    # Position actuators for all 27 controlled joints
    _leg_joints = ["leg_pelvic_pitch", "leg_pelvic_roll", "leg_pelvic_yaw",
                   "leg_knee_pitch", "leg_ankle_pitch", "leg_ankle_roll"]
    _arm_joints = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw",
                   "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll"]
    _act_lines = []
    for side in ["left", "right"]:
        for j in _leg_joints:
            _act_lines.append(f'    <position name="{side}_{j}_joint" joint="{side}_{j}_joint" kp="60" kv="4"/>')
    _act_lines.append('    <position name="waist_yaw_joint" joint="waist_yaw_joint" kp="40" kv="3"/>')
    for side in ["left", "right"]:
        for j in _arm_joints:
            _act_lines.append(f'    <position name="{side}_{j}_joint" joint="{side}_{j}_joint" kp="40" kv="3"/>')

    mesh_abs = str((xml_dir.parent / "meshes").resolve())

    scene_xml = f"""<mujoco model="scene">
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

    scene_path = xml_dir / "_scene_checker.xml"
    scene_path.write_text(scene_xml)
    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
    finally:
        scene_path.unlink(missing_ok=True)
    print(f"Loaded model: {model.njnt} joints, {model.nq} DOFs, {model.nu} actuators")

    data = mujoco.MjData(model)

    # Build joint mapping: CSV column name -> MuJoCo qpos index
    qpos_map = {}  # csv_name -> qpos_index
    ctrl_map = {}  # csv_name -> actuator_index (for dynamics mode)

    # Pre-build joint-id -> actuator-index lookup
    joint_to_actuator = {}
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid][0])
        if jid >= 0:
            joint_to_actuator[jid] = aid

    skipped = 0
    for csv_name in csv_joint_names:
        # Map CSV column name to MuJoCo joint name
        if csv_name.endswith("_pos"):
            joint_name = csv_name[:-4]  # strip _pos suffix
        elif csv_name.endswith("_vel"):
            skipped += 1
            continue
        elif csv_name.endswith("_joint"):
            joint_name = csv_name
        else:
            joint_name = csv_name + "_joint"  # legacy format from trajectory_writer

        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            skipped += 1
            continue
        qposadr = model.jnt_qposadr[jid]
        qpos_map[csv_name] = qposadr

        # Map to actuator for dynamics mode
        if jid in joint_to_actuator:
            ctrl_map[csv_name] = joint_to_actuator[jid]
    print(f"  Mapped {len(qpos_map)} joints ({skipped} columns skipped)")
    if args.dynamics:
        print(f"  Dynamics mode: {len(ctrl_map)} joints with actuators")

    # Pre-compute full qpos array for each frame
    n_frames = len(timestamps)
    qpos_sequence = np.tile(model.qpos0.copy(), (n_frames, 1))
    for csv_name, qpos_idx in qpos_map.items():
        qpos_sequence[:, qpos_idx] = joint_data[csv_name]

    # In kinematic mode, pre-compute base_link z and ankle angles so both feet
    # stay flat on the floor.
    if not args.dynamics:
        import struct

        FLOOR_Z = -0.8834
        left_ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                          "left_leg_ankle_roll_link")
        right_ankle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                           "right_leg_ankle_roll_link")

        # Ankle pitch/roll joint qpos indices and parent body IDs for foot flattening
        ankle_pitch_qpos = {}
        ankle_roll_qpos = {}
        ankle_pitch_parent = {}  # parent body of ankle_pitch_link
        ankle_pitch_range = {}
        ankle_roll_range = {}

        for side in ["left", "right"]:
            ap_body = f"{side}_leg_ankle_pitch_link"
            ar_body = f"{side}_leg_ankle_roll_link"
            ap_joint = f"{side}_leg_ankle_pitch_joint"
            ar_joint = f"{side}_leg_ankle_roll_joint"

            ap_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ap_joint)
            ar_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ar_joint)
            ankle_pitch_qpos[side] = model.jnt_qposadr[ap_jid]
            ankle_roll_qpos[side] = model.jnt_qposadr[ar_jid]
            ankle_pitch_range[side] = model.jnt_range[ap_jid]
            ankle_roll_range[side] = model.jnt_range[ar_jid]

            ap_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ap_body)
            ankle_pitch_parent[side] = model.body_parentid[ap_body_id]

        # Load foot mesh vertices (local frame relative to ankle_roll_link body)
        mesh_dir = Path(__file__).parent.parent / "casbot_band_urdf" / "meshes"
        foot_verts = {}  # side -> (N, 3) float32 array

        for side in ["left", "right"]:
            stl_path = mesh_dir / f"{side}_leg_ankle_roll_link.STL"
            verts = []
            with open(stl_path, "rb") as f:
                f.seek(80)
                n_tris = struct.unpack("<I", f.read(4))[0]
                for _ in range(n_tris):
                    f.read(12)  # normal
                    verts.append(struct.unpack("<3f", f.read(12)))
                    verts.append(struct.unpack("<3f", f.read(12)))
                    verts.append(struct.unpack("<3f", f.read(12)))
                    f.read(2)  # attribute
            foot_verts[side] = np.unique(np.array(verts, dtype=np.float32), axis=0)

        print(f"  Computing foot-ground constraints + ZMP balance "
              f"(left: {len(foot_verts['left'])} verts, right: {len(foot_verts['right'])} verts)...")

        for frame in range(n_frames):
            data.qpos[:] = qpos_sequence[frame]

            # Iterative IK: foot-flattening, Z-contact, and CoM balance
            for _iter in range(20):
                mujoco.mj_kinematics(model, data)

                # -- 1. Foot flattening: ankle pitch/roll so foot +Z points up --
                for side in ["left", "right"]:
                    z_k = data.xmat[ankle_pitch_parent[side]][6:9]
                    new_roll = np.arcsin(np.clip(-z_k[1], -1.0, 1.0))
                    new_pitch = np.arctan2(z_k[0], z_k[2])
                    new_pitch = np.clip(new_pitch, *ankle_pitch_range[side])
                    new_roll = np.clip(new_roll, *ankle_roll_range[side])
                    data.qpos[ankle_pitch_qpos[side]] = new_pitch
                    data.qpos[ankle_roll_qpos[side]] = new_roll

                mujoco.mj_kinematics(model, data)

                # -- 2. Foot-ground contact: base Z so lowest foot vertex touches floor --
                foot_min_z = float("inf")
                for side, body_id in [("left", left_ankle_id), ("right", right_ankle_id)]:
                    xpos = data.xpos[body_id]
                    z_row = data.xmat[body_id][6:9]
                    world_z = xpos[2] + foot_verts[side] @ z_row
                    foot_min_z = min(foot_min_z, world_z.min())
                data.qpos[2] += FLOOR_Z - float(foot_min_z)

                mujoco.mj_kinematics(model, data)

                # -- 3. ZMP / CoM balance: pitch & roll of the floating base --
                com = _compute_robot_com(model, data)
                support_center = _compute_support_center(
                    data, left_ankle_id, right_ankle_id,
                )
                com_error = support_center - com[:2]  # desired - actual

                if abs(com_error[0]) < 0.003 and abs(com_error[1]) < 0.003:
                    break

                com_height = max(com[2] - FLOOR_Z, 0.1)
                gain = 0.35
                delta_pitch = np.clip(com_error[0] / com_height * gain, -0.06, 0.06)
                delta_roll = np.clip(-com_error[1] / com_height * gain, -0.06, 0.06)
                _apply_base_rotation(data.qpos, delta_pitch, delta_roll)

            # Save balanced pose back to the sequence
            qpos_sequence[frame] = data.qpos.copy()

    # Audio playback via aplay/paplay
    audio_proc = None
    temp_wav = None

    if audio is not None:
        import tempfile
        import subprocess

        # Write WAV file
        temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wav_path = temp_wav.name
        temp_wav.close()

        # Convert float32 [-1,1] to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        import wave
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        # Find available player
        import shutil
        player = None
        for p in ["paplay", "aplay", "ffplay"]:
            if shutil.which(p):
                player = p
                break

        if player:
            print(f"  Starting audio: {player}")
            audio_proc = subprocess.Popen(
                [player, wav_path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            time.sleep(0.1)  # Let audio player buffer

    # MuJoCo viewer
    mode_label = "dynamics" if args.dynamics else "kinematic"
    print(f"\nStarting simulation ({mode_label})...")
    print("  Controls: drag mouse to rotate, scroll to zoom, Esc to quit")

    if args.dynamics:
        n_substeps = max(1, round(dt_traj / model.opt.timestep))
        print(f"  Physics: {n_substeps} substeps/frame (timestep={model.opt.timestep:.4f}s)")

        # ZMP balance controller state
        left_ankle_id_bal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                               "left_leg_ankle_roll_link")
        right_ankle_id_bal = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                                "right_leg_ankle_roll_link")
        BALANCE_KP = 1.2
        BALANCE_KD = 0.08
        BALANCE_COM_H = 0.85
        prev_com_error = np.zeros(2)

    sim_start_time = time.time()
    if audio_proc is None:
        print("  (no audio)")
    else:
        print("  Audio + motion synced")
        sim_start_time = time.time()  # reset after audio started

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -10
        viewer.cam.lookat = [0, 0, -0.2]

        last_frame = -1
        while viewer.is_running():
            elapsed = (time.time() - sim_start_time) * args.slow
            frame = min(int(elapsed / dt_traj), n_frames - 1)

            if frame != last_frame:
                if args.dynamics:
                    # Set position targets for all controlled joints
                    for csv_name, aid in ctrl_map.items():
                        data.ctrl[aid] = joint_data[csv_name][frame]

                    # ZMP / CoM balance feedback on pelvic pitch & roll
                    support_center = _compute_support_center(
                        data, left_ankle_id_bal, right_ankle_id_bal,
                    )
                    com = _compute_robot_com(model, data)
                    com_error = support_center - com[:2]
                    d_error = (com_error - prev_com_error) / max(dt_traj, 1e-6)

                    pitch_adj = (com_error[0] / BALANCE_COM_H * BALANCE_KP
                                 + d_error[0] * BALANCE_KD)
                    roll_adj = (-com_error[1] / BALANCE_COM_H * BALANCE_KP
                                - d_error[1] * BALANCE_KD)

                    for side in ["left", "right"]:
                        csv_pp = f"{side}_leg_pelvic_pitch"
                        if csv_pp in ctrl_map:
                            data.ctrl[ctrl_map[csv_pp]] += pitch_adj
                        csv_pr = f"{side}_leg_pelvic_roll"
                        if csv_pr in ctrl_map:
                            data.ctrl[ctrl_map[csv_pr]] += roll_adj

                    prev_com_error = com_error.copy()

                    # Step physics
                    for _ in range(n_substeps):
                        mujoco.mj_step(model, data)
                else:
                    # Kinematic: set joint positions + enforce foot-ground contact
                    data.qpos[:] = qpos_sequence[frame]
                    mujoco.mj_forward(model, data)
                last_frame = frame

            viewer.sync()

    # Cleanup
    if audio_proc is not None:
        audio_proc.terminate()
        audio_proc.wait()
    if temp_wav is not None:
        import os
        os.unlink(temp_wav.name)


if __name__ == "__main__":
    main()
