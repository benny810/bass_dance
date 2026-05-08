"""Dance motion primitives parameterized by musical features.

Each primitive function takes a time array plus MusicalFeatures and returns
a dict mapping joint_name -> angle_offset array (same length as time array).

Joint names match the CASBOT URDF:
  left_leg_pelvic_pitch, left_leg_pelvic_roll, left_leg_pelvic_yaw,
  left_leg_knee_pitch, left_leg_ankle_pitch, left_leg_ankle_roll,
  right_leg_pelvic_pitch, right_leg_pelvic_roll, right_leg_pelvic_yaw,
  right_leg_knee_pitch, right_leg_ankle_pitch, right_leg_ankle_roll,
  waist_yaw

All offsets are ADDITIVE to neutral stance (zeros).
The trajectory_generator sums them and clamps to joint limits.
"""

from typing import Dict, List
import numpy as np
from .feature_extractor import MusicalFeatures


def generate_bounce(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    depth: float = 0.20,
    decay_time: float = 0.12,
) -> Dict[str, np.ndarray]:
    """Generate bounce (knee spring) synced to note onsets.

    On each onset, knees flex proportionally to velocity.
    Decays exponentially between onsets.
    """
    n = len(sample_times)
    dt = sample_times[1] - sample_times[0] if n > 1 else 0.02

    # Build a bounce envelope from onset impulses
    bounce_env = np.zeros(n)
    waist_env = np.zeros(n)
    onset_indices = features.onset_indices

    for i, idx in enumerate(onset_indices):
        if idx < n:
            # Downbeat (beat phase ~0) gets deeper bounce than upbeat (~0.5)
            phase_at_onset = features.beat_phase[idx]
            is_downbeat_like = phase_at_onset < 0.25
            onset_depth = depth if is_downbeat_like else depth * 0.6

            # Scale by onset strength
            strength = features.onset_strength[idx]
            onset_depth *= 0.7 + 0.3 * strength

            # Add accent boost
            onset_depth += features.accent[idx] * depth * 0.4

            # Inject impulse and let it decay
            decay_samples = int(decay_time / dt)
            for j in range(min(decay_samples, n - idx)):
                envelope = onset_depth * np.exp(-j * dt / decay_time)
                bounce_env[idx + j] += envelope
                # Alternating waist twist on each onset
                sign = 1.0 if i % 2 == 0 else -1.0
                waist_env[idx + j] += sign * envelope * 0.15

    # Clamp to reasonable range
    bounce_env = np.clip(bounce_env, 0, depth * 1.5)

    return {
        "left_leg_knee_pitch": bounce_env.copy(),
        "right_leg_knee_pitch": bounce_env.copy(),
        "left_leg_pelvic_pitch": bounce_env * 0.3,
        "right_leg_pelvic_pitch": bounce_env * 0.3,
        "waist_yaw": waist_env,
    }


def generate_sway(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    amplitude: float = 0.12,
) -> Dict[str, np.ndarray]:
    """Generate side-to-side sway driven by pitch contour and measure boundaries.

    Positive sway = weight on right leg (right pelvic_roll positive).
    """
    n = len(sample_times)
    dt = sample_times[1] - sample_times[0] if n > 1 else 0.02

    # Primary sway driver: pitch contour
    pitch_driven = features.pitch_contour * amplitude

    # Secondary: slow periodic sway on measure boundaries
    beats_per_measure = features.time_signature[0]
    sec_per_measure = beats_per_measure * 60.0 / features.bpm
    measure_wave = np.sin(2 * np.pi * sample_times / sec_per_measure) * amplitude * 0.5

    # Detect phrase boundaries to trigger larger sways
    phrase_env = np.zeros(n)
    for boundary_idx in features.phrase_boundaries:
        if boundary_idx < n:
            # Inject a half-sine pulse
            pulse_samples = int(sec_per_measure / dt)
            for j in range(min(pulse_samples, n - boundary_idx)):
                phrase_env[boundary_idx + j] += np.sin(np.pi * j / pulse_samples)
    phrase_env *= amplitude * 0.8

    sway = pitch_driven + measure_wave + phrase_env
    sway = np.clip(sway, -amplitude * 2, amplitude * 2)

    result = {}
    # Right leg: positive sway -> right pelvic_roll increases
    result["right_leg_pelvic_roll"] = np.clip(sway, 0, None)
    result["left_leg_pelvic_roll"] = np.clip(-sway, 0, None)
    # Ankle roll compensates in opposite direction
    result["right_leg_ankle_roll"] = -np.clip(sway, 0, None) * 0.6
    result["left_leg_ankle_roll"] = -np.clip(-sway, 0, None) * 0.6
    # Waist counter-rotates against hips for upper body expression
    result["waist_yaw"] = -measure_wave * 1.2 - phrase_env * 0.6

    return result


def generate_step(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    step_depth: float = 0.18,
) -> Dict[str, np.ndarray]:
    """Generate forward/backward steps triggered at phrase boundaries.

    Alternates left and right leg as the stepping leg.
    """
    n = len(sample_times)
    dt = sample_times[1] - sample_times[0] if n > 1 else 0.02

    beats_per_second = features.bpm / 60.0
    seconds_per_beat = 60.0 / features.bpm

    # Find step candidates: measure downbeats OR phrase boundaries, minimum 4 beats apart
    step_candidates = set(features.phrase_boundaries)
    for i in range(n):
        if features.is_downbeat[i] > 0.5:
            step_candidates.add(i)

    step_candidates = sorted(step_candidates)
    filtered_steps = []
    last_step_beat = -10.0
    leg_side = 0  # 0 = left, 1 = right

    for idx in step_candidates:
        beat_at_idx = sample_times[idx] * beats_per_second
        if beat_at_idx - last_step_beat >= 4.0 and idx < n - 10:
            filtered_steps.append((idx, leg_side))
            last_step_beat = beat_at_idx
            leg_side = 1 - leg_side

    # Build step contributions
    step_duration = int(0.5 * seconds_per_beat / dt)

    result = {
        "left_leg_pelvic_pitch": np.zeros(n),
        "left_leg_pelvic_yaw": np.zeros(n),
        "left_leg_knee_pitch": np.zeros(n),
        "left_leg_ankle_pitch": np.zeros(n),
        "right_leg_pelvic_pitch": np.zeros(n),
        "right_leg_pelvic_yaw": np.zeros(n),
        "right_leg_knee_pitch": np.zeros(n),
        "right_leg_ankle_pitch": np.zeros(n),
    }

    for idx, side in filtered_steps:
        side_str = "left" if side == 0 else "right"
        other_str = "right" if side == 0 else "left"

        for j in range(min(step_duration, n - idx)):
            phase = j / step_duration
            envelope = np.sin(np.pi * phase)

            # Leading leg: knee flex, pelvic pitch forward, ankle adjust
            result[f"{side_str}_leg_knee_pitch"][idx + j] += step_depth * envelope
            result[f"{side_str}_leg_pelvic_pitch"][idx + j] += step_depth * 0.5 * envelope
            result[f"{side_str}_leg_ankle_pitch"][idx + j] += step_depth * 0.4 * envelope
            # Leading leg rotates outward slightly
            result[f"{side_str}_leg_pelvic_yaw"][idx + j] += step_depth * 0.3 * envelope

            # Support leg: slight opposite adjustment
            result[f"{other_str}_leg_knee_pitch"][idx + j] += step_depth * 0.1 * envelope
            result[f"{other_str}_leg_pelvic_pitch"][idx + j] += step_depth * 0.05 * envelope
            result[f"{other_str}_leg_ankle_pitch"][idx + j] -= step_depth * 0.2 * envelope

    return result


def generate_squat(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    squat_depth: float = 0.40,
    hold_beats: float = 3.0,
) -> Dict[str, np.ndarray]:
    """Generate deep squat on strong accented low notes with proper buildup.

    Only triggers when accent + low note coincide on a downbeat, spaced >= 8 beats apart.
    """
    n = len(sample_times)
    dt = sample_times[1] - sample_times[0] if n > 1 else 0.02
    seconds_per_beat = 60.0 / features.bpm
    beats_per_second = features.bpm / 60.0

    squat_env = np.zeros(n)

    last_squat_beat = -20.0
    for i in range(n):
        if (
            features.is_downbeat[i] > 0.5
            and features.accent[i] > 0.2
            and features.is_low_note[i] > 0.5
        ):
            beat_at_i = sample_times[i] * beats_per_second
            if beat_at_i - last_squat_beat >= 8.0:
                last_squat_beat = beat_at_i

                # Attack: go down over ~0.3 beats
                attack_samples = int(0.3 * seconds_per_beat / dt)
                for j in range(min(attack_samples, n - i)):
                    phase = j / attack_samples
                    squat_env[i + j] = squat_depth * (phase ** 0.5)  # fast start, slow settle

                # Hold: maintain squat depth
                hold_samples = int(hold_beats * seconds_per_beat / dt)
                hold_start = i + attack_samples
                for j in range(min(hold_samples, n - hold_start)):
                    squat_env[hold_start + j] = squat_depth * (1.0 - 0.3 * (j / hold_samples))

                # Release: come back up over ~0.5 beats
                release_start = hold_start + hold_samples
                release_samples = int(0.5 * seconds_per_beat / dt)
                for j in range(min(release_samples, n - release_start)):
                    phase = j / release_samples
                    remaining = squat_env[max(0, release_start - 1)]
                    squat_env[release_start + j] = remaining * (1.0 - phase) ** 2

    squat_env = np.clip(squat_env, 0, squat_depth)

    return {
        "left_leg_knee_pitch": squat_env.copy(),
        "right_leg_knee_pitch": squat_env.copy(),
        "left_leg_pelvic_pitch": squat_env * 0.25,
        "right_leg_pelvic_pitch": squat_env * 0.25,
    }
