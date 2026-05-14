#!/usr/bin/env python3
"""Extract PCA motion primitives from mocap example CSV files.

Usage:
    python -m midi_to_dance.pca_extractor csv/example1.csv csv/example2.csv -o pca_model.npz

The pipeline applied to the raw 50 Hz mocap is, in order:

  1.  Load   — read the 13 lower-body joint angles from each CSV
  2.  Detrend (high-pass)
           — remove < 0.3 Hz drift so PCs capture dance motion, not slow
             postural drift.  Raw mocap has 40–87 % of its per-joint
             variance below 0.2 Hz; without this step PCA mostly fits
             pose drift instead of dance coordination patterns.
  3.  Static-frame drop
           — remove frames whose high-pass speed is below the 10 %
             quantile.  Idle moments contribute noise to the covariance
             without adding new motion information.
  4.  Mirror augmentation
           — append a left-right-mirrored copy with sign-flipped lateral
             joints (hip roll/yaw, ankle roll, waist yaw).  Makes the
             dataset bilaterally symmetric so the resulting PCs are
             cleanly either symmetric (e.g. squat) or anti-symmetric
             (e.g. stride) rather than biased toward whichever side
             happened to dominate the take.
  5.  PCA via SVD on centred data — preserves radians as the unit, so the
     downstream `mean_pose + Σ activation × component` reconstruction is
     unchanged.
  6.  Sign canonicalisation
           — fix every component's sign so its dominant entry is positive,
             giving reproducible PC signs across runs/datasets.

Optionally (`--canonical`) the PCs can be additionally reordered via a
Hungarian maximum-|cosine| assignment against semantic prototypes (squat,
yaw, rock, stride, …) so the
slot index matches the per-PC accent assignments hard-coded in
`pca_motion.py`.  This is off by default because the preprocessing
already yields physically clean primitives, and forcing a fixed slot
ordering can demote a high-variance PC into a low-priority slot.

All preprocessing steps can be disabled individually via CLI flags so the
extractor stays comparable to the prior baseline.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.linalg import svd
from scipy import signal

# 13 lower-body joints (must match JOINT_NAMES[:13] in trajectory_generator.py)
LOWER_BODY_JOINTS = [
    "left_leg_pelvic_pitch", "left_leg_pelvic_roll", "left_leg_pelvic_yaw",
    "left_leg_knee_pitch", "left_leg_ankle_pitch", "left_leg_ankle_roll",
    "right_leg_pelvic_pitch", "right_leg_pelvic_roll", "right_leg_pelvic_yaw",
    "right_leg_knee_pitch", "right_leg_ankle_pitch", "right_leg_ankle_roll",
    "waist_yaw",
]

# Indices used by mirror augmentation
_L_LEG = [0, 1, 2, 3, 4, 5]           # L: hip_p, hip_r, hip_y, knee, ankle_p, ankle_r
_R_LEG = [6, 7, 8, 9, 10, 11]
_LATERAL_FLIP = [1, 2, 5, 7, 8, 11, 12]  # hip_roll(L/R), hip_yaw(L/R), ankle_roll(L/R), waist_yaw


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


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def detrend_highpass(data: np.ndarray, fs: float, cutoff: float) -> np.ndarray:
    """Subtract per-joint mean, then high-pass with a zero-phase 2nd-order
    Butterworth filter, then add the mean back.  Removes slow postural
    drift (< `cutoff` Hz) while preserving dance frequencies.

    The reattached mean keeps the pose centred on the natural "ready"
    posture, which is what the downstream reconstruction uses.
    """
    if cutoff <= 0:
        return data
    mu = data.mean(axis=0)
    centred = data - mu
    sos = signal.butter(2, cutoff, btype="high", fs=fs, output="sos")
    centred = signal.sosfiltfilt(sos, centred, axis=0)
    return centred + mu


def drop_static_frames(
    data: np.ndarray, fs: float, quantile: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop frames whose centred-and-detrended speed falls below `quantile`.

    Speed is computed on the per-joint centred signal so that long
    stationary holds (which would all share the same drift component) get
    treated as static regardless of where they sit in the postural range.
    """
    if quantile <= 0:
        return data, np.ones(len(data), dtype=bool)
    centred = data - data.mean(axis=0)
    vel = np.gradient(centred, axis=0) * fs
    speed = np.linalg.norm(vel, axis=1)
    thr = float(np.quantile(speed, quantile))
    keep = speed >= thr
    return data[keep], keep


def mirror_pose(data: np.ndarray) -> np.ndarray:
    """Return a left-right mirrored copy of the data.

    Mirroring is a swap of L↔R leg columns combined with sign flips on
    laterally-directional joints (roll/yaw of hip and ankle, plus
    waist yaw).  Pitch axes (hip pitch, knee, ankle pitch) keep their
    sign because they describe forward-back motion that is mirror-
    invariant.
    """
    out = np.empty_like(data)
    out[:, _L_LEG] = data[:, _R_LEG]
    out[:, _R_LEG] = data[:, _L_LEG]
    out[:, 12] = data[:, 12]
    out[:, _LATERAL_FLIP] *= -1.0
    return out


def preprocess(
    data: np.ndarray,
    fs: float = 50.0,
    hp_cutoff: float = 0.3,
    static_quantile: float = 0.10,
    mirror: bool = True,
) -> Tuple[np.ndarray, dict]:
    """Apply the full preprocessing pipeline; return cleaned data and stats.

    Order: detrend → drop-static → mirror.  Detrending is done first so
    the speed used by `drop_static_frames` reflects dance motion only.
    Mirroring is last so each kept frame contributes both its original
    and its reflection equally to the PCA covariance.
    """
    stats = {"n_input": len(data)}
    cleaned = detrend_highpass(data, fs=fs, cutoff=hp_cutoff)
    cleaned, keep = drop_static_frames(cleaned, fs=fs, quantile=static_quantile)
    stats["n_after_static_drop"] = len(cleaned)
    stats["static_drop_fraction"] = 1.0 - keep.mean()
    if mirror:
        cleaned = np.vstack([cleaned, mirror_pose(cleaned)])
    stats["n_final"] = len(cleaned)
    stats["mirror_augmented"] = mirror
    return cleaned, stats


# ---------------------------------------------------------------------------
# PCA + sign canonicalisation
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical PC ordering: each slot has a semantic prototype that downstream
# `pca_motion.py` expects.  Joint indices below correspond to LOWER_BODY_JOINTS.
#   0..5  = L hip_p, hip_r, hip_y, knee, ankle_p, ankle_r
#   6..11 = R hip_p, hip_r, hip_y, knee, ankle_p, ankle_r
#   12    = waist_yaw
# ---------------------------------------------------------------------------

def _build_prototypes() -> np.ndarray:
    """Construct 7 unit-norm semantic prototypes for canonical reordering.

    Slot semantics match the per-PC accent assignments in `pca_motion.py`:
        0 onset knee flex  → symmetric knee bend  (squat / bounce)
        1 pitch modulation → pelvic yaw (waist+L+R)  (body twist)
        2 beat rocking     → anti-symmetric ankle / hip pitch  (weight rock)
        3 stride accent    → anti-symmetric knee bend  (lateral step)
        4 density twist    → anti-symmetric pelvic yaw  (counter-twist)
        5 register shift   → lateral hip roll (symmetric in URDF sign convention)
        6 phrase breathing → symmetric hip pitch  (forward lean)

    Each prototype only specifies the joints it *cares* about, with the
    URDF-aware sign convention.  Joints not relevant to a prototype are
    zero, so the cosine similarity rewards alignment on the right joints
    without penalising secondary loadings.
    """
    proto = np.zeros((7, 13))
    # 0 squat: knees flex together, hip pitches go forward together.
    proto[0, 3] = 1.0;  proto[0, 9] = 1.0
    proto[0, 0] = -0.3; proto[0, 6] = -0.3
    # 1 yaw body twist: waist + both pelvic yaws turn together.
    proto[1, 12] = 1.0
    proto[1, 2] = 0.7;  proto[1, 8] = 0.7
    # 2 weight rock: opposite-sign hip pitch + opposite-sign ankle pitch (and
    #   ankle moves opposite to hip on the same leg — this is what the data
    #   actually shows as the largest motion primitive).
    proto[2, 0] = -1.0; proto[2, 6] = 1.0
    proto[2, 4] = 1.0;  proto[2, 10] = -1.0
    # 3 lateral step: knees move opposite (one bends, one extends).
    proto[3, 3] = 1.0;  proto[3, 9] = -1.0
    # 4 counter-twist: pelvic yaws move opposite.
    proto[4, 2] = 1.0;  proto[4, 8] = -1.0
    # 5 body lean / sway: both hip rolls same sign (URDF convention =
    #   whole body leans laterally).
    proto[5, 1] = 1.0;  proto[5, 7] = 1.0
    # 6 forward lean: both hip pitches same sign (no anti-symmetric content).
    proto[6, 0] = 1.0;  proto[6, 6] = 1.0
    proto[6, 4] = -0.3; proto[6, 10] = -0.3

    for i in range(proto.shape[0]):
        n = np.linalg.norm(proto[i])
        if n > 1e-12:
            proto[i] /= n
    return proto


def _hungarian_maximise(cost: np.ndarray) -> List[int]:
    """Return an assignment of rows to columns that maximises the total
    cost (cost ≥ 0).  Uses scipy's linear_sum_assignment under the hood
    if available, otherwise falls back to a greedy permutation search
    for small problems.  Returns a list `assign[i] = column for row i`.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        # linear_sum_assignment minimises; negate to maximise.
        rows, cols = linear_sum_assignment(-cost)
        return [int(cols[i]) for i in range(len(rows))]
    except Exception:
        n_rows, n_cols = cost.shape
        used: set = set()
        out: List[int] = []
        for i in range(n_rows):
            best, best_v = -1, -np.inf
            for j in range(n_cols):
                if j in used:
                    continue
                if cost[i, j] > best_v:
                    best_v = cost[i, j]
                    best = j
            out.append(best)
            used.add(best)
        return out


def _canonical_reorder(
    components: np.ndarray, ev_ratio: np.ndarray, scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reorder PCs to match `_build_prototypes()` by global |cos| matching.

    Uses the Hungarian algorithm to maximise the total |cosine| across
    all (slot, PC) pairs, so a strong prototype-PC match isn't blocked by
    an earlier slot greedily claiming an only-marginally-better PC.

    Each output component is sign-flipped so its activation's positive
    direction aligns with the prototype's positive direction.

    Returns reordered (components, ev_ratio, scores) such that
    `mean + scores @ components` reconstructs the same data as before.
    """
    n_in = components.shape[0]
    n_slots = min(7, n_in)
    proto = _build_prototypes()[:n_slots]

    comp_norm = np.linalg.norm(components, axis=1, keepdims=True) + 1e-12
    comp_unit = components / comp_norm
    cos = proto @ comp_unit.T            # (n_slots, n_in)

    assign = _hungarian_maximise(np.abs(cos))
    leftover = [j for j in range(n_in) if j not in set(assign)]
    order = list(assign) + leftover

    new_components = components[order].copy()
    new_ev = ev_ratio[order].copy()
    new_scores = scores[:, order].copy()

    for slot in range(n_slots):
        if cos[slot, order[slot]] < 0:
            new_components[slot] *= -1.0
            new_scores[:, slot] *= -1.0

    return new_components, new_ev, new_scores


def compute_pca(
    data: np.ndarray,
    n_components: int = 7,
    canonical: bool = False,
) -> dict:
    """PCA via SVD on mean-centred data.

    When `canonical=True` the resulting PCs are reordered + sign-fixed so
    each slot matches a semantic prototype (see `_build_prototypes`).
    This keeps the downstream `pca_motion.py` per-PC accent assignments
    valid across re-extractions even though the raw variance ordering
    may change with cleaner data.

    Returns: mean_pose, components, explained_variance_ratio, std_scores,
    n_frames, n_components.
    """
    n_frames = data.shape[0]
    mean_pose = data.mean(axis=0)
    centred = data - mean_pose

    U, s, Vt = svd(centred, full_matrices=False)

    n_comp = int(min(n_components, len(s)))
    components = Vt[:n_comp].copy()  # (n_comp, n_joints)
    total_var = float(np.sum(s ** 2))
    ev_ratio = (s[:n_comp] ** 2) / total_var
    scores = centred @ components.T   # (n_frames, n_comp)

    if canonical:
        components, ev_ratio, scores = _canonical_reorder(
            components, ev_ratio, scores,
        )
    else:
        # Sign canonicalisation: largest |loading| in each PC is positive.
        for i in range(n_comp):
            peak = int(np.argmax(np.abs(components[i])))
            if components[i, peak] < 0:
                components[i] *= -1.0
                scores[:, i] *= -1.0

    std_scores = np.std(scores, axis=0)

    return {
        "mean_pose": mean_pose.astype(np.float32),
        "components": components.astype(np.float32),
        "explained_variance_ratio": ev_ratio.astype(np.float32),
        "std_scores": std_scores.astype(np.float32),
        "n_frames": n_frames,
        "n_components": n_comp,
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def symmetry_score(component: np.ndarray) -> Tuple[float, str]:
    """Decompose a 13-vector component into bilateral symmetric and
    anti-symmetric parts.  Returns (sym_fraction, label).

    sym_fraction in [0, 1]: 1 = pure symmetric (both legs together),
    0 = pure anti-symmetric (legs opposite).  Waist yaw is treated as
    anti-symmetric (it sign-flips under L↔R mirroring).
    """
    leg_L = component[_L_LEG]
    leg_R = component[_R_LEG].copy()
    # Apply mirror to right side: swap to compare in L's frame
    # Mirror of right leg in L's frame: flip sign of roll (R[1]), yaw (R[2]), ankle_roll (R[5])
    leg_R_in_L = leg_R.copy()
    leg_R_in_L[[1, 2, 5]] *= -1.0
    waist = component[12]

    leg_vec = np.concatenate([leg_L, leg_R_in_L])
    sym_part = (leg_L + leg_R_in_L) / 2.0
    asy_part = (leg_L - leg_R_in_L) / 2.0
    sym_e = float(np.dot(sym_part, sym_part) * 2.0)
    asy_e = float(np.dot(asy_part, asy_part) * 2.0)
    # Waist is anti-symmetric under mirror
    asy_e += float(waist ** 2)
    total = sym_e + asy_e + 1e-12
    frac = sym_e / total
    label = "sym " if frac > 0.7 else ("anti" if frac < 0.3 else "mix ")
    return frac, label


def print_summary(pca: dict, joint_names: List[str]) -> None:
    """Print human-readable summary of PCA results."""
    print(f"\nPCA Summary ({pca['n_frames']} frames, "
          f"{pca['n_components']} components, {len(joint_names)} joints)")
    print(f"{'PC':>4s}  {'Var%':>6s}  {'Cum%':>6s}  {'Sym':>5s}  Top loadings")
    print("-" * 95)

    cum = 0.0
    for i in range(pca["n_components"]):
        ev = float(pca["explained_variance_ratio"][i])
        cum += ev
        comp = pca["components"][i]
        sym_f, sym_lbl = symmetry_score(comp)
        top_idx = np.argsort(np.abs(comp))[::-1][:5]
        top_str = "  ".join(
            f"{joint_names[j].replace('_leg_', '_').replace('_joint', '')}({comp[j]:+.3f})"
            for j in top_idx
        )
        print(f"{i+1:4d}  {ev*100:5.1f}%  {cum*100:5.1f}%  "
              f"{sym_lbl} {sym_f*100:3.0f}%  {top_str}")


def save_pca_model(pca: dict, joint_names: List[str], output_path: str) -> None:
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract PCA motion primitives from mocap CSV files",
    )
    parser.add_argument("csv_files", nargs="+", help="Mocap CSV files to process")
    parser.add_argument("-o", "--output", default="pca_model.npz",
                        help="Output .npz path (default: pca_model.npz)")
    parser.add_argument("-n", "--n-components", type=int, default=7,
                        help="Number of PCs to keep (default: 7)")
    parser.add_argument("--fs", type=float, default=50.0,
                        help="Mocap sampling rate in Hz (default: 50)")
    parser.add_argument("--hp-cutoff", type=float, default=0.3,
                        help="High-pass cutoff in Hz (default: 0.3). "
                             "Set <=0 to disable detrending.")
    parser.add_argument("--static-quantile", type=float, default=0.10,
                        help="Drop frames below this speed quantile (default: 0.10). "
                             "Set <=0 to disable.")
    parser.add_argument("--no-mirror", action="store_true",
                        help="Disable left-right mirror augmentation.")
    parser.add_argument("--canonical", action="store_true",
                        help="Reorder PCs to match `pca_motion.py`'s "
                             "expected semantic slots (squat, yaw, rock, "
                             "stride, twist, sway, lean).  Off by default; "
                             "the cleaned-data PCs are already physically "
                             "interpretable in pure variance order.")
    args = parser.parse_args()

    print(f"Loading {len(args.csv_files)} CSV file(s)...")
    data, joint_names = load_mocap_data(args.csv_files)
    print(f"  Raw: {data.shape[0]} frames × {data.shape[1]} joints")

    cleaned, stats = preprocess(
        data,
        fs=args.fs,
        hp_cutoff=args.hp_cutoff,
        static_quantile=args.static_quantile,
        mirror=not args.no_mirror,
    )
    print(f"Preprocessing:")
    print(f"  high-pass {args.hp_cutoff} Hz, drop bottom {args.static_quantile*100:.0f}% "
          f"speed quantile, mirror={'on' if not args.no_mirror else 'off'}")
    print(f"  static-drop kept {1.0 - stats['static_drop_fraction']:.1%} "
          f"({stats['n_after_static_drop']} frames)")
    print(f"  after mirror: {stats['n_final']} frames")

    print(f"\nComputing PCA ({args.n_components} components, "
          f"canonical-order={'on' if args.canonical else 'off'})...")
    pca = compute_pca(
        cleaned,
        n_components=args.n_components,
        canonical=args.canonical,
    )

    print_summary(pca, joint_names)
    save_pca_model(pca, joint_names, args.output)


if __name__ == "__main__":
    main()
