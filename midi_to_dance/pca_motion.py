"""Generate lower-body dance motion from music features using PCA primitives.

The PCA model is pre-computed by pca_extractor.py from mocap example data.
Each principal component captures a natural motion coordination pattern.
Music features modulate the activation (coefficient) of each PC over time.

Architecture:
    Continuous multi-sine carriers (one per PC, incommensurate frequencies)
    provide non-repeating baseline motion.  A musical-energy envelope derived
    from onset density and accent strength scales the carriers so that dense /
    energetic passages produce larger movements and sparse passages stay subtle.
    Event-driven accents (onset flex, accent stride, phrase pulses) are added
    on top, contributing ~30% of total energy.

Reconstruction:
    lower_trajs[t] = mean_pose + sum_i(activation_i[t] * component[i])
"""

from pathlib import Path
from typing import Dict, Optional, Set

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .feature_extractor import MusicalFeatures

# 13 lower-body joints (must match JOINT_NAMES[:13] in trajectory_generator.py)
LOWER_BODY_JOINTS = [
    "left_leg_pelvic_pitch", "left_leg_pelvic_roll", "left_leg_pelvic_yaw",
    "left_leg_knee_pitch", "left_leg_ankle_pitch", "left_leg_ankle_roll",
    "right_leg_pelvic_pitch", "right_leg_pelvic_roll", "right_leg_pelvic_yaw",
    "right_leg_knee_pitch", "right_leg_ankle_pitch", "right_leg_ankle_roll",
    "waist_yaw",
]

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _load_pca_model(pca_model_path: Optional[str] = None) -> dict:
    """Load PCA model from .npz file.

    Search order: 1) provided path, 2) module directory, 3) project root.
    """
    if pca_model_path is not None:
        return dict(np.load(pca_model_path, allow_pickle=True))

    candidates = [
        Path(__file__).parent / "pca_model.npz",
        Path(__file__).parent.parent / "pca_model.npz",
    ]
    for p in candidates:
        if p.exists():
            return dict(np.load(str(p), allow_pickle=True))

    raise FileNotFoundError(
        "pca_model.npz not found. Run: python -m midi_to_dance.pca_extractor "
        "csv/example1.csv csv/example2.csv -o pca_model.npz"
    )


# ---------------------------------------------------------------------------
# Continuous carriers  (non-repeating baseline motion for every PC)
# ---------------------------------------------------------------------------

# Each PC gets 3 incommensurate sine periods (seconds).  The ratios are chosen
# so the combined signal never exactly repeats within a typical song length.
# Frequencies are in the 0.02-0.35 Hz range, matching mocap dominant spectra.
_CARRIER_PERIODS = [
    [8.3, 17.7, 35.1],    # PC1 – stance depth
    [5.7, 13.3, 29.5],    # PC2 – pelvic yaw
    [3.1, 7.9, 19.3],     # PC3 – ankle rock (faster, beat-like)
    [11.2, 23.4, 41.7],   # PC4 – stride accent
    [6.5, 15.1, 31.8],    # PC5 – asymmetric hip twist
    [9.7, 21.1, 37.3],    # PC6 – weight shift
    [14.3, 27.6, 43.9],   # PC7 – breathing pelvic tilt
]


def _continuous_carriers(
    n: int,
    sample_times: np.ndarray,
    n_comp: int,
) -> np.ndarray:
    """Generate continuous unit-amplitude carriers for each PC.

    Each carrier is a weighted sum of 3 sine waves with incommensurate periods.
    Returns (n, n_comp) array with values roughly in [-1, 1].
    """
    carriers = np.zeros((n, n_comp))
    for i in range(min(n_comp, len(_CARRIER_PERIODS))):
        periods = _CARRIER_PERIODS[i]
        weights = [0.55, 0.30, 0.15]  # decreasing weight for faster components
        for period, weight in zip(periods, weights):
            carriers[:, i] += weight * np.sin(2 * np.pi * sample_times / period)
    return carriers


# ---------------------------------------------------------------------------
# Musical energy envelope  (onset density + accent  ->  amplitude modulator)
# ---------------------------------------------------------------------------

def _musical_energy_envelope(
    n: int,
    dt: float,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    accent: np.ndarray,
    bpm: float,
) -> np.ndarray:
    """Smooth musical-energy envelope in [0, 1] from onset density and accent.

    Smoothed over ~2 beats so it rises/falls with musical phrases rather than
    individual notes.
    """
    spb = 60.0 / bpm if bpm > 0 else 0.5
    raw = np.zeros(n)
    for idx in onset_indices:
        if idx < n:
            raw[idx] += onset_strength[idx] * 0.7 + accent[idx] * 0.3
    sigma = max(int(2.0 * spb / dt), 2)
    energy = gaussian_filter1d(raw, sigma=sigma)
    e_max = np.max(energy)
    if e_max > 1e-10:
        energy /= e_max
    return energy


# ---------------------------------------------------------------------------
# Event-driven accent envelopes  (~30% of total energy)
# ---------------------------------------------------------------------------

def _pc1_onset_accent(
    n: int,
    dt: float,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    beat_phase: np.ndarray,
    accent: np.ndarray,
    depth: float,
    decay_time: float = 0.15,
) -> np.ndarray:
    """PC1 accent: onset-triggered stance-flex impulse (negative direction)."""
    activation = np.zeros(n)
    decay_samples = int(decay_time / dt)

    for idx in onset_indices:
        if idx >= n:
            continue
        is_downbeat_like = beat_phase[idx] < 0.25
        amp = depth if is_downbeat_like else depth * 0.6
        amp *= 0.7 + 0.3 * onset_strength[idx]
        amp += accent[idx] * depth * 0.4
        amp = -amp  # negative = flex

        for j in range(min(decay_samples, n - idx)):
            activation[idx + j] += amp * np.exp(-j * dt / decay_time)

    return np.clip(activation, -depth * 2.0, 0.0)


def _pc4_accent_envelope(
    n: int,
    sample_times: np.ndarray,
    dt: float,
    accent: np.ndarray,
    is_low_note: np.ndarray,
    is_downbeat: np.ndarray,
    bpm: float,
    amplitude: float,
    hold_beats: float = 3.0,
) -> np.ndarray:
    """PC4 accent: accented low-note downbeats trigger ADSR stride."""
    bps = bpm / 60.0
    spb = 60.0 / bpm if bpm > 0 else 0.5

    trigger_indices = []
    last_trigger_beat = -20.0
    for i in range(n):
        if is_downbeat[i] > 0.5 and accent[i] > 0.2 and is_low_note[i] > 0.5:
            beat_at_i = sample_times[i] * bps
            if beat_at_i - last_trigger_beat >= 8.0:
                trigger_indices.append(i)
                last_trigger_beat = beat_at_i

    activation = np.zeros(n)
    for ti in trigger_indices:
        attack_s = int(0.3 * spb / dt)
        for j in range(min(attack_s, n - ti)):
            activation[ti + j] = amplitude * np.sqrt(j / max(attack_s - 1, 1))

        hold_s = int(hold_beats * spb / dt)
        hold_start = ti + attack_s
        for j in range(min(hold_s, n - hold_start)):
            activation[hold_start + j] = amplitude * (1.0 - 0.3 * j / max(hold_s, 1))

        release_start = hold_start + hold_s
        release_s = int(0.5 * spb / dt)
        for j in range(min(release_s, n - release_start)):
            remaining = activation[max(0, release_start - 1)] if release_start > 0 else 0.0
            if remaining > 0:
                activation[release_start + j] = remaining * (1.0 - j / max(release_s, 1)) ** 2

    return np.clip(activation, 0.0, amplitude * 1.2)


def _pc2_pitch_accent(
    n: int,
    sample_times: np.ndarray,
    dt: float,
    pitch_level: np.ndarray,
    bpm: float,
    time_signature: tuple,
    phrase_boundaries: Set[int],
    amplitude: float,
) -> np.ndarray:
    """PC2 accent: pitch-level modulation + measure wave + phrase pulses."""
    pitch_dev = pitch_level - np.mean(pitch_level)

    beats_per_measure = time_signature[0]
    sec_per_measure = beats_per_measure * 60.0 / bpm
    measure_wave = np.sin(2 * np.pi * sample_times / sec_per_measure) * 0.4

    phrase_pulse = np.zeros(n)
    pulse_samples = int(sec_per_measure / dt)
    for bi in phrase_boundaries:
        if bi < n:
            for j in range(min(pulse_samples, n - bi)):
                phrase_pulse[bi + j] += np.sin(np.pi * j / pulse_samples)
    phrase_pulse *= 0.6

    activation = (pitch_dev * 2.0 + measure_wave + phrase_pulse) * amplitude * 0.6
    return np.clip(activation, -amplitude * 1.5, amplitude * 1.5)


def _pc3_beat_accent(
    n: int,
    dt: float,
    beat_phase: np.ndarray,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """PC3 accent: beat-synced heel-toe rocking, modulated by onset envelope."""
    activation = np.sin(2 * np.pi * beat_phase) * amplitude * 0.25

    onset_env = np.zeros(n)
    for idx in onset_indices:
        if idx < n:
            onset_env[idx] += onset_strength[idx]
    sigma = max(int(0.1 / dt), 2)
    onset_env = gaussian_filter1d(onset_env, sigma=sigma)

    activation *= 0.3 + 0.7 * onset_env
    return activation


def _pc5_density_accent(
    n: int,
    sample_times: np.ndarray,
    dt: float,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    bpm: float,
    amplitude: float,
) -> np.ndarray:
    """PC5 accent: rhythmic density drives asymmetric hip twist."""
    impulse = np.zeros(n)
    for idx in onset_indices:
        if idx < n:
            impulse[idx] += onset_strength[idx]

    spb = 60.0 / bpm if bpm > 0 else 0.5
    sigma = max(int(2.0 * spb / dt), 2)
    density = gaussian_filter1d(impulse, sigma=sigma)
    d_max = np.max(density)
    if d_max > 1e-10:
        density /= d_max

    sign_period = 48.0 * spb
    sign_osc = np.sin(2 * np.pi * sample_times / sign_period)

    activation = density * sign_osc * amplitude * 0.6
    return np.clip(activation, -amplitude * 0.9, amplitude * 0.9)


def _pc6_register_accent(
    n: int,
    pitch_level: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """PC6 accent: pitch register  ->  right-leg weight-shift."""
    centered = pitch_level - np.mean(pitch_level)
    std = np.std(pitch_level)
    if std > 1e-10:
        centered /= std
    activation = np.tanh(centered) * amplitude * 0.6
    return np.clip(activation, -amplitude * 1.2, amplitude * 1.2)


def _pc7_phrase_accent(
    n: int,
    sample_times: np.ndarray,
    bpm: float,
    time_signature: tuple,
    phrase_boundaries: Set[int],
    amplitude: float,
) -> np.ndarray:
    """PC7 accent: phrase-level breathing arc."""
    boundaries = sorted(phrase_boundaries)
    if len(boundaries) == 0 or boundaries[0] != 0:
        boundaries = [0] + boundaries

    activation = np.zeros(n)

    if len(boundaries) >= 2:
        for i in range(len(boundaries)):
            start = boundaries[i]
            end = boundaries[i + 1] if i + 1 < len(boundaries) else n
            length = end - start
            if length <= 1:
                continue
            phase = np.arange(length) / length
            activation[start:end] = np.cos(np.pi * phase)
    else:
        beats_per_measure = time_signature[0]
        sec_per_measure = beats_per_measure * 60.0 / bpm
        activation[:] = np.cos(2 * np.pi * sample_times / (sec_per_measure * 4.0))

    activation *= amplitude * 0.5
    return np.clip(activation, -amplitude * 0.6, amplitude * 0.6)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_pca_motion(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    scale: float = 1.0,
    pca_model: Optional[dict] = None,
) -> Dict[str, np.ndarray]:
    """Generate lower-body joint trajectories from PCA model + music features.

    Returns dict mapping lower-body joint_name -> angle_array (radians, absolute).
    """
    if pca_model is None:
        pca_model = _load_pca_model()

    n = len(sample_times)
    dt = sample_times[1] - sample_times[0] if n > 1 else 0.02

    components = pca_model["components"]  # (n_comp, 13)
    mean_pose = pca_model["mean_pose"]     # (13,)
    std_scores = pca_model["std_scores"]   # (n_comp,)
    n_comp = pca_model["n_components"]

    amps = [std_scores[i] * scale for i in range(n_comp)]

    # -- Continuous carriers (always-on, never-repeating baseline motion) --
    carriers = _continuous_carriers(n, sample_times, n_comp)

    # -- Musical energy envelope (0-1, scales carrier amplitude) --
    energy = _musical_energy_envelope(
        n, dt, features.onset_indices, features.onset_strength,
        features.accent, features.bpm,
    )

    # -- Build activation matrix --
    # activation = carrier * modulated_energy * amplitude + accent
    # Energy is squared to increase section-to-section contrast (quiet
    # sections get much quieter, matching the 13:1 range seen in mocap).
    activations = np.zeros((n, n_comp))
    base_level = 0.05
    dynamic_range = 0.95

    for i in range(n_comp):
        envelope = base_level + dynamic_range * (energy ** 1.5)
        activations[:, i] = carriers[:, i] * envelope * amps[i] * 1.2

    # -- Event-driven accents (30% of energy) --
    if n_comp >= 1:
        activations[:, 0] += _pc1_onset_accent(
            n, dt, features.onset_indices, features.onset_strength,
            features.beat_phase, features.accent,
            depth=amps[0] * 0.35,
        )

    if n_comp >= 2:
        activations[:, 1] += _pc2_pitch_accent(
            n, sample_times, dt, features.pitch_level, features.bpm,
            features.time_signature, features.phrase_boundaries,
            amplitude=amps[1],
        )

    if n_comp >= 3:
        activations[:, 2] += _pc3_beat_accent(
            n, dt, features.beat_phase, features.onset_indices,
            features.onset_strength,
            amplitude=amps[2] * 0.35,
        )

    if n_comp >= 4:
        activations[:, 3] += _pc4_accent_envelope(
            n, sample_times, dt, features.accent, features.is_low_note,
            features.is_downbeat, features.bpm,
            amplitude=amps[3],
        )

    if n_comp >= 5:
        activations[:, 4] += _pc5_density_accent(
            n, sample_times, dt, features.onset_indices,
            features.onset_strength, features.bpm,
            amplitude=amps[4],
        )

    if n_comp >= 6:
        activations[:, 5] += _pc6_register_accent(
            n, features.pitch_level,
            amplitude=amps[5],
        )

    if n_comp >= 7:
        activations[:, 6] += _pc7_phrase_accent(
            n, sample_times, features.bpm, features.time_signature,
            features.phrase_boundaries,
            amplitude=amps[6],
        )

    # -- Reconstruct: (n, 13) = mean + activations @ components --
    lower_trajs = mean_pose[np.newaxis, :] + activations @ components

    # -- Pack into dict --
    result = {}
    jn_array = pca_model.get("joint_names", None)
    if jn_array is not None:
        joint_names = []
        for j in jn_array:
            if isinstance(j, (bytes, np.bytes_)):
                joint_names.append(j.decode("utf-8") if isinstance(j, bytes) else j.tobytes().decode("utf-8"))
            else:
                joint_names.append(str(j))
    else:
        joint_names = list(LOWER_BODY_JOINTS)

    for i, jn in enumerate(joint_names):
        result[jn] = lower_trajs[:, i]

    return result
