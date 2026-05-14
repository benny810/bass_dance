#!/usr/bin/env python3
"""Extract individual PCA action primitives as standalone trajectory CSV files.

Each of the 7 PCs is driven by its continuous carrier (3 incommensurate sines)
and saved as a separate CSV, making each coordination pattern visible in isolation.

Usage:
    python action_pattern/extract_primitives.py [--duration 30] [--scale 1.0]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from midi_to_dance.pca_motion import _load_pca_model, LOWER_BODY_JOINTS, _CARRIER_PERIODS
from midi_to_dance.trajectory_generator import JOINT_NAMES, JOINT_LIMITS, NEUTRAL_STANCE
from midi_to_dance.trajectory_writer import write_csv

PC_NAMES = [
    "PC1_weighted_sway",
    "PC2_lateral_step",
    "PC3_symmetric_squat",
    "PC4_pelvic_yaw",
    "PC5_forward_lean",
    "PC6_lateral_lean",
    "PC7_full_stretch",
]


def generate_primitive(pc_index: int, sample_times: np.ndarray, pca_model: dict,
                       scale: float = 1.0) -> dict:
    """Generate a single-PC trajectory using only continuous carrier modulation.

    Returns dict of joint_name -> angle_array for all 27 joints.
    """
    n = len(sample_times)
    mean_pose = pca_model["mean_pose"]         # (13,)
    component = pca_model["components"][pc_index]  # (13,)
    std_score = pca_model["std_scores"][pc_index]

    # Generate continuous carrier for this PC
    periods = _CARRIER_PERIODS[pc_index]
    weights = [0.55, 0.30, 0.15]
    carrier = np.zeros(n)
    for period, weight in zip(periods, weights):
        carrier += weight * np.sin(2 * np.pi * sample_times / period)

    amplitude = std_score * scale * 1.5  # slightly amplified for visibility
    activation = carrier * amplitude

    # Reconstruct: (n, 13) = mean + activation @ component (outer product)
    lower_trajs = mean_pose[np.newaxis, :] + np.outer(activation, component)

    # Pack into dict with all 27 joints
    lower_set = set(LOWER_BODY_JOINTS)
    trajectories = {}
    for joint_name in JOINT_NAMES:
        if joint_name in lower_set:
            idx = LOWER_BODY_JOINTS.index(joint_name)
            trajectories[joint_name] = lower_trajs[:, idx]
        else:
            trajectories[joint_name] = np.full(n, NEUTRAL_STANCE.get(joint_name, 0.0))

    # Clamp to joint limits
    for joint_name in JOINT_NAMES:
        lo, hi = JOINT_LIMITS[joint_name]
        trajectories[joint_name] = np.clip(trajectories[joint_name], lo, hi)

    return trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Extract individual PCA action primitives as CSV files"
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Duration in seconds per primitive (default: 30)")
    parser.add_argument("--dt", type=float, default=0.02,
                        help="Time step in seconds (default: 0.02 = 50Hz, matches example CSVs)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="Motion amplitude scale (default: 1.0)")
    parser.add_argument("-o", "--output-dir", type=str,
                        default=str(Path(__file__).parent),
                        help="Output directory (default: action_pattern/)")
    args = parser.parse_args()

    pca_model = _load_pca_model()
    n_comp = pca_model["n_components"]

    sample_times = np.arange(0, args.duration, args.dt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Extracting {n_comp} PCA action primitives "
          f"({args.duration}s each, dt={args.dt}s)...")

    for i in range(n_comp):
        name = PC_NAMES[i]
        ev = pca_model["explained_variance_ratio"][i] * 100
        top_joints = np.argsort(np.abs(pca_model["components"][i]))[::-1][:3]
        top_str = ", ".join(
            f"{LOWER_BODY_JOINTS[j]} ({pca_model['components'][i][j]:+.3f})"
            for j in top_joints
        )

        trajectories = generate_primitive(i, sample_times, pca_model,
                                          scale=args.scale)

        csv_path = out_dir / f"{name}.csv"
        write_csv(str(csv_path), sample_times, trajectories)
        print(f"  {name:30s}  var={ev:5.1f}%  top: {top_str}")


if __name__ == "__main__":
    main()
