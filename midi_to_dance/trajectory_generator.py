"""Compose motion primitives into full joint trajectories with constraints."""

from typing import Dict, List, Tuple, Optional
import numpy as np
from .midi_parser import parse_midi, MidiData
from .feature_extractor import extract_features, MusicalFeatures
from .pca_motion import generate_pca_motion, LOWER_BODY_JOINTS


JOINT_NAMES = [
    # Legs
    "left_leg_pelvic_pitch",
    "left_leg_pelvic_roll",
    "left_leg_pelvic_yaw",
    "left_leg_knee_pitch",
    "left_leg_ankle_pitch",
    "left_leg_ankle_roll",
    "right_leg_pelvic_pitch",
    "right_leg_pelvic_roll",
    "right_leg_pelvic_yaw",
    "right_leg_knee_pitch",
    "right_leg_ankle_pitch",
    "right_leg_ankle_roll",
    # Waist
    "waist_yaw",
    # Left arm
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow_pitch",
    "left_wrist_yaw",
    "left_wrist_pitch",
    "left_wrist_roll",
    # Right arm
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow_pitch",
    "right_wrist_yaw",
    "right_wrist_pitch",
    "right_wrist_roll",
]

JOINT_LIMITS = {
    "left_leg_pelvic_pitch": (-1.9199, 1.5708),
    "left_leg_pelvic_roll": (-0.17453, 1.5708),
    "left_leg_pelvic_yaw": (-1.5708, 1.5708),
    "left_leg_knee_pitch": (0.0, 2.5307),
    "left_leg_ankle_pitch": (-0.87266, 0.50614),
    "left_leg_ankle_roll": (-0.50614, 0.50614),
    "right_leg_pelvic_pitch": (-1.9199, 1.5708),
    "right_leg_pelvic_roll": (-1.5708, 0.17453),
    "right_leg_pelvic_yaw": (-1.5708, 1.5708),
    "right_leg_knee_pitch": (0.0, 2.5307),
    "right_leg_ankle_pitch": (-0.87266, 0.50614),
    "right_leg_ankle_roll": (-0.50614, 0.50614),
    "waist_yaw": (-1.5708, 1.5708),
    "left_shoulder_pitch": (-3.22886, 1.60570),
    "left_shoulder_roll": (-0.34907, 3.14159),
    "left_shoulder_yaw": (-1.5708, 1.5708),
    "left_elbow_pitch": (-2.26893, 0.34907),
    "left_wrist_yaw": (-1.5708, 1.5708),
    "left_wrist_pitch": (-1.0472, 1.0472),
    "left_wrist_roll": (-1.5708, 1.0472),
    "right_shoulder_pitch": (-3.22886, 1.60570),
    "right_shoulder_roll": (-3.14159, 0.34907),
    "right_shoulder_yaw": (-1.5708, 1.5708),
    "right_elbow_pitch": (-2.26893, 0.34907),
    "right_wrist_yaw": (-1.5708, 1.5708),
    "right_wrist_pitch": (-1.0472, 1.0472),
    "right_wrist_roll": (-1.0472, 1.5708),
}


# Neutral stance added to all trajectories before motion primitives.
# Knee bend gives a natural "ready" posture; hip abduction widens the stance.
NEUTRAL_STANCE = {
    # Pelvic pitch set to 0 — torso lean is handled by ZMP/CoM balance in simulation
    "left_leg_pelvic_pitch": 0.0,
    "left_leg_pelvic_roll": 0.20,     # left leg abduct (wider stance)
    "left_leg_pelvic_yaw": 0.0,
    "left_leg_knee_pitch": 0.35,      # baseline knee bend
    "left_leg_ankle_pitch": 0.0,
    "left_leg_ankle_roll": -0.12,     # evert to keep foot flat on wider stance
    "right_leg_pelvic_pitch": 0.0,
    "right_leg_pelvic_roll": -0.20,   # right leg abduct (wider stance)
    "right_leg_pelvic_yaw": 0.0,
    "right_leg_knee_pitch": 0.35,     # baseline knee bend
    "right_leg_ankle_pitch": 0.0,
    "right_leg_ankle_roll": 0.12,     # evert to keep foot flat on wider stance
    "waist_yaw": 0.0,
    # Left arm (from example.csv mid frame: bass-playing pose)
    "left_shoulder_pitch": 0.1707,
    "left_shoulder_roll": 0.3270,
    "left_shoulder_yaw": 0.4422,
    "left_elbow_pitch": -1.0927,
    "left_wrist_yaw": 1.2570,
    "left_wrist_pitch": -0.0577,
    "left_wrist_roll": -1.5140,
    # Right arm (from example.csv mid frame: bass-playing pose)
    "right_shoulder_pitch": -0.1487,
    "right_shoulder_roll": -0.7554,
    "right_shoulder_yaw": 0.3063,
    "right_elbow_pitch": -1.8647,
    "right_wrist_yaw": 0.2585,
    "right_wrist_pitch": 0.1556,
    "right_wrist_roll": 1.0734,
}


def generate_trajectory(
    midi_path: str,
    dt: float = 0.02,
    scale: float = 1.0,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Generate full joint trajectories from a MIDI file.

    Lower-body joints (legs + waist) are generated from PCA motion primitives
    extracted from mocap data.  Arm joints stay at their NEUTRAL_STANCE values.

    Returns (sample_times, dict of joint_name -> angle_array).
    """
    midi_data = parse_midi(midi_path)
    features = extract_features(midi_data, dt=dt)
    sample_times = features.sample_times

    # PCA-based lower-body motion (absolute joint angles, not offsets)
    pca_output = generate_pca_motion(sample_times, features, scale=scale)

    lower_set = set(LOWER_BODY_JOINTS)
    trajectories: Dict[str, np.ndarray] = {}
    for joint_name in JOINT_NAMES:
        if joint_name in lower_set:
            trajectories[joint_name] = pca_output[joint_name]
        else:
            # Arm joints stay at constant NEUTRAL_STANCE pose
            trajectories[joint_name] = np.full(
                len(sample_times), NEUTRAL_STANCE.get(joint_name, 0.0)
            )

    # Clamp to joint limits
    for joint_name in JOINT_NAMES:
        low, high = JOINT_LIMITS[joint_name]
        trajectories[joint_name] = np.clip(trajectories[joint_name], low, high)

    # Light smoothing
    from scipy.ndimage import gaussian_filter1d
    sigma = 2  # ~40ms at 50Hz
    for joint_name in JOINT_NAMES:
        trajectories[joint_name] = gaussian_filter1d(
            trajectories[joint_name], sigma=sigma
        )

    # Pass through foot-step phase signals for simulation IK
    for key in ("left_foot_step", "right_foot_step"):
        if key in pca_output:
            trajectories[key] = pca_output[key]

    return sample_times, trajectories


def trajectory_stats(
    sample_times: np.ndarray,
    trajectories: Dict[str, np.ndarray],
) -> str:
    """Generate a human-readable statistics report."""
    lines = [f"Trajectory: {len(sample_times)} samples, "
             f"{sample_times[-1]:.1f}s at dt={sample_times[1]-sample_times[0]:.3f}s"]
    lines.append(f"{'Joint':35s} {'min':>8s} {'max':>8s} {'range':>8s} {'limit':>12s}")
    lines.append("-" * 75)
    for name in JOINT_NAMES:
        t = trajectories[name]
        lo, hi = JOINT_LIMITS[name]
        lines.append(
            f"{name:35s} {np.min(t):8.4f} {np.max(t):8.4f} "
            f"{np.max(t)-np.min(t):8.4f} [{lo:7.4f}, {hi:7.4f}]"
        )
    return "\n".join(lines)
