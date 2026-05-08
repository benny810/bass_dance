#!/usr/bin/env python3
"""MuJoCo simulation player for robot dance trajectories with synced MIDI audio.

Usage:
    python midi_to_dance/simulate.py output.csv mid/yellow贝斯.mid
    python midi_to_dance/simulate.py output.csv mid/yellow贝斯.mid --slow 0.5
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import mujoco
import mujoco.viewer


def load_trajectory(csv_path: str):
    """Load CSV trajectory into (timestamps, joint_data dict, header)."""
    with open(csv_path) as f:
        header = f.readline().strip().split(",")
    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    timestamps = data[:, 0]
    joint_data = {}
    for i, name in enumerate(header[1:]):
        joint_data[name] = data[:, i + 1]
    return timestamps, joint_data, header[1:]


def synthesize_midi_audio(midi_path: str, sample_rate: int = 44100):
    """Synthesize audio from MIDI file using additive synthesis.

    Returns (audio_samples, sample_rate).
    audio_samples is a 1D float32 numpy array in [-1, 1].
    """
    import mido

    mid = mido.MidiFile(midi_path)
    ticks_per_beat = mid.ticks_per_beat

    # Extract tempo map and notes
    abs_tick = 0
    tempo = 500000  # default 120 BPM
    tempo_map = [(0, tempo)]

    notes = []  # (start_sec, end_sec, midi_note, velocity)
    active = {}

    for msg in mid.tracks[0]:
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
    args = parser.parse_args()

    # Load trajectory
    print(f"Loading trajectory: {args.csv_file}")
    timestamps, joint_data, csv_joint_names = load_trajectory(args.csv_file)
    dt_traj = timestamps[1] - timestamps[0]
    print(f"  {len(timestamps)} frames, dt={dt_traj:.3f}s, duration={timestamps[-1]:.1f}s")

    # Synthesize audio
    audio = None
    sample_rate = 44100
    if not args.no_audio:
        print(f"Synthesizing audio from: {args.midi_file}")
        audio, sample_rate, audio_duration = synthesize_midi_audio(args.midi_file)
        print(f"  {audio_duration:.1f}s audio, {sample_rate} Hz")

    # Load MuJoCo model
    urdf_path = Path(__file__).parent.parent / "casbot_band_urdf" / "urdf" / \
                "CASBOT02_ENCOS_7dof_shell_20251015_P1L_bass.urdf"
    print(f"Loading URDF: {urdf_path}")
    model = mujoco.MjModel.from_xml_path(str(urdf_path))
    data = mujoco.MjData(model)

    # Build joint mapping: CSV column name -> MuJoCo qpos index
    qpos_map = {}  # csv_name -> qpos_index
    for csv_name in csv_joint_names:
        joint_name = csv_name + "_joint"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        qposadr = model.jnt_qposadr[jid]
        qpos_map[csv_name] = qposadr
    print(f"  Mapped {len(qpos_map)} joints")

    # Pre-compute full qpos array for each frame
    n_frames = len(timestamps)
    qpos_sequence = np.tile(model.qpos0.copy(), (n_frames, 1))
    for csv_name, qpos_idx in qpos_map.items():
        qpos_sequence[:, qpos_idx] = joint_data[csv_name]

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
    print("\nStarting simulation...")
    print("  Controls: drag mouse to rotate, scroll to zoom, Esc to quit")
    print("  Sync: audio and motion start together")

    sim_start_time = time.time()
    if audio_proc is None:
        print("  (no audio)")
    else:
        print("  Audio + motion synced")
        sim_start_time = time.time()  # reset after audio started

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 3.0
        viewer.cam.elevation = -5
        viewer.cam.lookat = [0, 0, 0.8]

        last_frame = -1
        while viewer.is_running():
            elapsed = (time.time() - sim_start_time) * args.slow
            frame = min(int(elapsed / dt_traj), n_frames - 1)

            if frame != last_frame:
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
