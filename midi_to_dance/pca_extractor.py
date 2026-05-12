#!/usr/bin/env python3
"""Extract PCA motion primitives from mocap example CSV files.

Usage:
    python -m midi_to_dance.pca_extractor csv/example1.csv csv/example2.csv -o pca_model.npz
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.linalg import svd

# 13 lower-body joints (must match JOINT_NAMES[:13] in trajectory_generator.py)
LOWER_BODY_JOINTS = [
    "left_leg_pelvic_pitch", "left_leg_pelvic_roll", "left_leg_pelvic_yaw",
    "left_leg_knee_pitch", "left_leg_ankle_pitch", "left_leg_ankle_roll",
    "right_leg_pelvic_pitch", "right_leg_pelvic_roll", "right_leg_pelvic_yaw",
    "right_leg_knee_pitch", "right_leg_ankle_pitch", "right_leg_ankle_roll",
    "waist_yaw",
]


def load_mocap_data(csv_paths: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Load lower-body joint angles from mocap CSV files.

    Handles BOM via utf-8-sig encoding.  Columns are matched by appending
    '_joint_pos' to each of the 13 LOWER_BODY_JOINTS names.

    Returns:
        data: (total_frames, 13) float64 array in radians
        joint_names: the 13 joint names in column order
    """
    all_frames = []
    col_indices = None

    for path in csv_paths:
        with open(path, encoding="utf-8-sig") as f:
            header = f.readline().strip().split(",")

        if col_indices is None:
            col_indices = []
            for jn in LOWER_BODY_JOINTS:
                col_name = jn + "_joint_pos"
                try:
                    col_indices.append(header.index(col_name))
                except ValueError:
                    print(f"Warning: column '{col_name}' not found in {path}")
                    col_indices.append(-1)

        data = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
        valid_cols = [c for c in col_indices if c >= 0]
        if len(valid_cols) != 13:
            print(f"Error: only {len(valid_cols)}/13 lower-body joints found in {path}")
            sys.exit(1)

        all_frames.append(data[:, valid_cols])

    stacked = np.vstack(all_frames).astype(np.float64)
    return stacked, list(LOWER_BODY_JOINTS)


def compute_pca(data: np.ndarray, n_components: int = 7):
    """PCA via SVD on centered data.

    Returns dict with mean_pose, components, explained_variance_ratio,
    std_scores, n_frames, n_components.
    """
    n_frames = data.shape[0]
    mean_pose = data.mean(axis=0)
    centered = data - mean_pose

    U, s, Vt = svd(centered, full_matrices=False)

    n_comp = min(n_components, len(s))
    components = Vt[:n_comp]  # (n_comp, 13)
    scores = centered @ components.T  # (n_frames, n_comp)
    total_var = np.sum(s ** 2)
    ev_ratio = (s[:n_comp] ** 2) / total_var
    std_scores = np.std(scores, axis=0)

    return {
        "mean_pose": mean_pose.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance_ratio": ev_ratio.astype(np.float32),
        "std_scores": std_scores.astype(np.float32),
        "n_frames": n_frames,
        "n_components": n_comp,
    }


def save_pca_model(pca: dict, joint_names: List[str], output_path: str):
    """Save PCA model to .npz file."""
    np.savez(
        output_path,
        mean_pose=pca["mean_pose"],
        components=pca["components"],
        explained_variance_ratio=pca["explained_variance_ratio"],
        std_scores=pca["std_scores"],
        joint_names=np.array(joint_names, dtype="S"),
        n_frames=pca["n_frames"],
        n_components=pca["n_components"],
    )
    print(f"Saved PCA model to {output_path}")


def print_summary(pca: dict, joint_names: List[str]):
    """Print human-readable summary of PCA results."""
    print(f"\nPCA Summary ({pca['n_frames']} frames, "
          f"{pca['n_components']} components, {len(joint_names)} joints)")
    print(f"{'PC':>4s}  {'Var%':>7s}  {'Cum%':>7s}  Top-5 joint loadings")
    print("-" * 70)

    cum = 0.0
    for i in range(pca["n_components"]):
        ev = pca["explained_variance_ratio"][i]
        cum += ev
        comp = pca["components"][i]
        top5_idx = np.argsort(np.abs(comp))[::-1][:5]
        top5_str = "  ".join(
            f"{joint_names[j]}({comp[j]:+.3f})" for j in top5_idx
        )
        print(f"{i+1:4d}  {ev*100:6.1f}%  {cum*100:6.1f}%  {top5_str}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract PCA motion primitives from mocap CSV files"
    )
    parser.add_argument("csv_files", nargs="+", help="Mocap CSV files to process")
    parser.add_argument("-o", "--output", default="pca_model.npz",
                        help="Output .npz path (default: pca_model.npz)")
    parser.add_argument("-n", "--n-components", type=int, default=7,
                        help="Number of PCs to keep (default: 7)")
    args = parser.parse_args()

    print(f"Loading {len(args.csv_files)} CSV file(s)...")
    data, joint_names = load_mocap_data(args.csv_files)
    print(f"  Total frames: {data.shape[0]}, joints: {data.shape[1]}")

    print(f"Computing PCA ({args.n_components} components)...")
    pca = compute_pca(data, n_components=args.n_components)

    print_summary(pca, joint_names)
    save_pca_model(pca, joint_names, args.output)


if __name__ == "__main__":
    main()
