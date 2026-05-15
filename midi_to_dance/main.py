#!/usr/bin/env python3
"""CLI entry point: convert MIDI file to robot dance joint trajectories.

Usage:
    python -m midi_to_dance.main mid/yellow贝斯.mid -o output.csv
    python -m midi_to_dance.main mid/yellow贝斯.mid -o output.csv --scale 0.8 --plot
    python -m midi_to_dance.main --bpm 120 --duration 30 -o output.csv
"""

import argparse
import sys
from pathlib import Path

from .midi_parser import parse_midi
from .feature_extractor import extract_features, synthesize_features_from_bpm
from .trajectory_generator import (
    generate_trajectory,
    generate_trajectory_from_features,
    trajectory_stats,
)
from .trajectory_writer import write_csv


def main():
    parser = argparse.ArgumentParser(
        description="Generate robot leg dance trajectories from MIDI files "
                    "or from BPM alone (--bpm mode)"
    )
    parser.add_argument(
        "midi_file", type=str, nargs="?", default=None,
        help="Path to input MIDI file (omit when using --bpm)",
    )
    parser.add_argument("-o", "--output", type=str, default="dance_trajectory.csv",
                        help="Output CSV file path")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Time step in seconds (default: 0.02 = 50Hz)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Motion amplitude scale (default: 1.0)")
    parser.add_argument("--enable-steps", action="store_true",
                        help="Layer the optional hand-coded single-leg-lift "
                             "events on top of PCA motion (OFF by default). "
                             "This layer is not derived from action primitives "
                             "and triggers occasional asymmetric leg lifts on "
                             "accented downbeats.")
    parser.add_argument("--pc-weights", type=float, nargs="+", default=None,
                        metavar="W",
                        help="Per-PC weight multipliers, space-separated "
                             "(default: 1.0 for all).  Length should match "
                             "the number of PCs in pca_model.npz (typically "
                             "7).  Shorter lists are right-padded with 1.0; "
                             "longer lists are truncated.  PC semantics: "
                             "1=weight rock, 2=lateral knee shift, 3=squat, "
                             "4=yaw twist, 5=forward lean, 6=body sway, "
                             "7=subtle extension.  Example: "
                             "--pc-weights 1.0 1.0 1.8 0.5 1.0 1.0 1.0")
    parser.add_argument("--bpm", type=float, default=None,
                        help="BPM for metronomic dance generation (no MIDI "
                             "required; use with --duration)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Duration in seconds (required with --bpm)")
    parser.add_argument("--time-signature", type=str, default="4/4",
                        help="Time signature as 'N/D' (default: 4/4)")
    parser.add_argument("--plot", action="store_true",
                        help="Visualize the generated trajectory")
    parser.add_argument("--stats", action="store_true",
                        help="Print trajectory statistics")
    args = parser.parse_args()

    # Mutual exclusivity / requirement checks.
    if args.bpm is None and args.midi_file is None:
        print("Error: either provide a MIDI file or use --bpm with --duration")
        sys.exit(1)
    if args.bpm is not None and args.midi_file is not None:
        print("Error: --bpm and MIDI file are mutually exclusive")
        sys.exit(1)
    if args.bpm is not None and args.duration is None:
        print("Error: --duration is required with --bpm")
        sys.exit(1)

    pc_w_str = (
        " ".join(f"{w:.2f}" for w in args.pc_weights)
        if args.pc_weights else "default (all 1.0)"
    )
    midi_data = None  # only set in MIDI mode

    if args.bpm is not None:
        # -- BPM-only mode --
        ts_parts = args.time_signature.split("/")
        if len(ts_parts) != 2:
            print(f"Error: invalid --time-signature '{args.time_signature}' "
                  "(expected e.g. '4/4')")
            sys.exit(1)
        time_sig = (int(ts_parts[0]), int(ts_parts[1]))

        print(f"BPM-only mode: BPM={args.bpm:.0f}, duration={args.duration}s, "
              f"time sig={time_sig[0]}/{time_sig[1]}")
        print(f"\nGenerating trajectories (dt={args.dt}s, scale={args.scale}, "
              f"pc_weights={pc_w_str}, enable_steps={args.enable_steps})...")
        features = synthesize_features_from_bpm(
            args.bpm, args.duration, dt=args.dt, time_signature=time_sig,
        )
        sample_times, trajectories = generate_trajectory_from_features(
            features.sample_times, features,
            scale=args.scale, pc_weights=args.pc_weights,
            enable_steps=args.enable_steps,
        )
    else:
        # -- MIDI mode --
        midi_path = Path(args.midi_file)
        if not midi_path.exists():
            print(f"Error: MIDI file not found: {args.midi_file}")
            sys.exit(1)

        print(f"Parsing MIDI: {midi_path}")
        midi_data = parse_midi(str(midi_path))
        print(f"  BPM: {midi_data.bpm:.1f}, "
              f"Time sig: {midi_data.time_signature[0]}/{midi_data.time_signature[1]}")
        print(f"  Notes: {len(midi_data.notes)}, "
              f"Duration: {midi_data.total_duration_seconds:.1f}s")

        print(f"\nGenerating trajectories (dt={args.dt}s, scale={args.scale}, "
              f"pc_weights={pc_w_str}, enable_steps={args.enable_steps})...")
        sample_times, trajectories = generate_trajectory(
            str(midi_path), dt=args.dt, scale=args.scale,
            pc_weights=args.pc_weights, enable_steps=args.enable_steps,
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

    # Mark note onsets (MIDI mode only) and set title
    if midi_data is not None:
        seconds_per_beat = 60.0 / midi_data.bpm
        onset_times = [n.beat * seconds_per_beat for n in midi_data.notes]
        for ax in axes[:len(groups)]:
            for ot in onset_times:
                ax.axvline(x=ot, color="gray", alpha=0.08, linewidth=0.5)
        title_str = (f"Dance Trajectory (BPM={midi_data.bpm:.0f}, "
                     f"{len(midi_data.notes)} notes)")
    else:
        title_str = "Dance Trajectory (BPM-only mode)"

    # Hide unused subplots
    for i in range(len(groups), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle(title_str, fontsize=14)
    plt.tight_layout()
    plt.savefig("dance_trajectory.png", dpi=100)
    print("  Plot saved: dance_trajectory.png")
    plt.close()


if __name__ == "__main__":
    main()
