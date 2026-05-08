#!/usr/bin/env python3
"""CLI entry point: convert MIDI file to robot dance joint trajectories.

Usage:
    python -m midi_to_dance.main mid/yellow贝斯.mid -o output.csv
    python -m midi_to_dance.main mid/yellow贝斯.mid -o output.csv --scale 0.8 --plot
"""

import argparse
import sys
from pathlib import Path

from .midi_parser import parse_midi
from .feature_extractor import extract_features
from .trajectory_generator import generate_trajectory, trajectory_stats
from .trajectory_writer import write_csv


def main():
    parser = argparse.ArgumentParser(
        description="Generate robot leg dance trajectories from MIDI files"
    )
    parser.add_argument("midi_file", type=str, help="Path to input MIDI file")
    parser.add_argument("-o", "--output", type=str, default="dance_trajectory.csv",
                        help="Output CSV file path")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Time step in seconds (default: 0.02 = 50Hz)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Motion amplitude scale (default: 1.0)")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize the generated trajectory")
    parser.add_argument("--stats", action="store_true",
                        help="Print trajectory statistics")
    args = parser.parse_args()

    midi_path = Path(args.midi_file)
    if not midi_path.exists():
        print(f"Error: MIDI file not found: {args.midi_file}")
        sys.exit(1)

    print(f"Parsing MIDI: {midi_path}")
    midi_data = parse_midi(str(midi_path))
    print(f"  BPM: {midi_data.bpm:.1f}, Time sig: {midi_data.time_signature[0]}/{midi_data.time_signature[1]}")
    print(f"  Notes: {len(midi_data.notes)}, Duration: {midi_data.total_duration_seconds:.1f}s")

    print(f"\nGenerating trajectories (dt={args.dt}s, scale={args.scale})...")
    sample_times, trajectories = generate_trajectory(
        str(midi_path), dt=args.dt, scale=args.scale
    )

    if args.stats:
        print("\n" + trajectory_stats(sample_times, trajectories))

    print(f"\nWriting CSV: {args.output}")
    write_csv(args.output, sample_times, trajectories)
    print(f"  Done: {len(sample_times)} frames, {sample_times[-1]:.1f}s")

    if args.plot:
        _plot_trajectory(sample_times, trajectories, midi_data)


def _plot_trajectory(sample_times, trajectories, midi_data):
    import matplotlib.pyplot as plt
    from .trajectory_generator import JOINT_NAMES

    fig, axes = plt.subplots(5, 3, figsize=(18, 16))
    axes = axes.flatten()

    # Group by function
    groups = {
        "Pelvic Pitch": ["left_leg_pelvic_pitch", "right_leg_pelvic_pitch"],
        "Pelvic Roll": ["left_leg_pelvic_roll", "right_leg_pelvic_roll"],
        "Pelvic Yaw": ["left_leg_pelvic_yaw", "right_leg_pelvic_yaw"],
        "Knee Pitch": ["left_leg_knee_pitch", "right_leg_knee_pitch"],
        "Ankle Pitch": ["left_leg_ankle_pitch", "right_leg_ankle_pitch"],
        "Ankle Roll": ["left_leg_ankle_roll", "right_leg_ankle_roll"],
        "Waist Yaw": ["waist_yaw"],
    }

    for ax_idx, (title, joints) in enumerate(groups.items()):
        if ax_idx >= len(axes):
            break
        ax = axes[ax_idx]
        for jn in joints:
            label = "L" if "left" in jn else "R" if "right" in jn else "W"
            ax.plot(sample_times, trajectories[jn], label=f"{label}: {jn}", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (rad)")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3)

    # Mark note onsets
    seconds_per_beat = 60.0 / midi_data.bpm
    onset_times = [n.beat * seconds_per_beat for n in midi_data.notes]
    for ax in axes[:len(groups)]:
        for ot in onset_times:
            ax.axvline(x=ot, color="gray", alpha=0.08, linewidth=0.5)

    # Hide unused subplots
    for i in range(len(groups), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(f"Dance Trajectory (BPM={midi_data.bpm:.0f}, {len(midi_data.notes)} notes)",
                 fontsize=14)
    plt.tight_layout()
    plt.savefig("dance_trajectory.png", dpi=100)
    print("  Plot saved: dance_trajectory.png")
    plt.close()


if __name__ == "__main__":
    main()
