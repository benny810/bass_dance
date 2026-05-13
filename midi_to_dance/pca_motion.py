"""Generate lower-body dance motion from music features using PCA primitives.

The PCA model is pre-computed by `pca_extractor.py` from mocap example data
after high-pass detrending + mirror augmentation, so each PC is either purely
bilaterally symmetric or purely anti-symmetric.  The default variance-ordered
PCs (see README) carry the following semantics, which are used as fixed slot
assignments below:

    PC1 ─ anti-sym front/back weight rock (R/L ankle pitch & hip pitch flip)
    PC2 ─ anti-sym lateral knee shift     (one knee bends while the other
                                          extends; ankle roll same direction)
    PC3 ─ sym squat                        (both knees flex, ankles pitch back)
    PC4 ─ anti-sym whole-body yaw twist    (both pelvic yaws turn together)
    PC5 ─ sym forward lean                 (both hip pitches forward)
    PC6 ─ anti-sym body sway / hip roll    (both hip rolls same direction =
                                          full-body lateral lean)
    PC7 ─ sym subtle extension             (ankles + knees rise)

Each PC is driven by the musical event that naturally fits its motion type:

    PC1 ← beat-phase sine          (alternates L/R every half beat)
    PC2 ← accented-onset stride    (ADSR, sign alternates per trigger)
    PC3 ← onset knee-flex impulse  (exponentially decaying flex on each onset)
    PC4 ← pitch + measure twist    (pitch deviation × measure-wave × phrase)
    PC5 ← phrase breathing arc     (slow cosine per musical phrase)
    PC6 ← density × slow sway      (note density × ~48-beat-period sine)
    PC7 ← pitch register           (high notes raise the body via tanh)

Architecture:
    Continuous multi-sine carriers (one per PC, incommensurate frequencies)
    provide non-repeating baseline motion.  The pre-computed
    `features.energy` envelope from `feature_extractor.py` (which already
    blends smoothed note-density and metric-aware accent) modulates the
    carriers so that dense / energetic passages produce larger movements
    and sparse passages stay subtle.  Event-driven accents (per the table
    above) are added on top.  An explicit `_identify_step_events` layer
    triggers visible foot-lifts on strong accented downbeats; while a
    step is active a rhythm-suppression mask silences all PC activations
    and groove patterns so the step reads cleanly.  Finally, beat-synced
    groove patterns target hip/knee/waist joints directly with absolute-
    radian amplitudes.

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

# Joint indices within the 13-element lower-body array
_L_HIP_P, _L_HIP_R, _L_HIP_Y = 0, 1, 2
_L_KNEE = 3
_L_ANK_P, _L_ANK_R = 4, 5
_R_HIP_P, _R_HIP_R, _R_HIP_Y = 6, 7, 8
_R_KNEE = 9
_R_ANK_P, _R_ANK_R = 10, 11
_WAIST = 12

_ANKLE_INDICES = [_L_ANK_P, _L_ANK_R, _R_ANK_P, _R_ANK_R]

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

# Per-PC incommensurate sine periods (seconds).  Frequency ratios are
# irrational so each carrier's 3-sine sum never exactly repeats within song
# length.  Period magnitudes are tuned per-PC to match the new (variance-
# ordered) semantics: dominant rocking/stepping PCs get visible mid-tempo
# baseline motion; lean/extension PCs sit on slow breathing-tempo carriers.
_CARRIER_PERIODS = [
    [4.7, 11.3, 23.5],    # PC1 anti-sym weight rock  — visible groove tempo
    [7.9, 17.1, 31.2],    # PC2 anti-sym lateral step — slower deliberate sway
    [9.1, 19.7, 35.4],    # PC3 sym squat             — slow continuous flex
    [5.7, 13.3, 29.5],    # PC4 anti-sym yaw twist    — mid-tempo body twist
    [14.3, 27.6, 43.9],   # PC5 sym forward lean      — slow breathing
    [10.5, 22.3, 39.1],   # PC6 anti-sym body sway    — slow lateral lean
    [16.7, 30.2, 47.3],   # PC7 sym extension         — slowest, subtle rise
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
# Musical energy
# ---------------------------------------------------------------------------
#
# `feature_extractor.extract_features` already computes a per-sample
# `energy` signal (≈ 0.6·smoothed_note_density + 0.4·smoothed_accent) that
# captures section-level musical energy more cleanly than re-smoothing raw
# onsets here.  We therefore consume `features.energy` directly in
# `generate_pca_motion` instead of recomputing it.


# ---------------------------------------------------------------------------
# Event-driven accent envelopes  (~30% of total energy)
#
# Each function returns an (n,)-shaped activation array intended to be added
# to one specific PC's column.  Function name describes the musical driver;
# its docstring records which PC slot it is wired to and what the activation
# sign means in joint space.  Wiring lives in `generate_pca_motion`.
# ---------------------------------------------------------------------------

def _beat_rock_accent(
    n: int,
    dt: float,
    beat_phase: np.ndarray,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Beat-phase sine wave, gated by local onset density.  Wired to PC1
    (anti-sym weight rock): activation > 0 lifts L heel / R toe, activation
    < 0 inverts, giving an alternating L↔R weight rock that lands on
    beats.
    """
    activation = np.sin(2 * np.pi * beat_phase) * amplitude * 0.6

    onset_env = np.zeros(n)
    for idx in onset_indices:
        if idx < n:
            onset_env[idx] += onset_strength[idx]
    sigma = max(int(0.1 / dt), 2)
    onset_env = gaussian_filter1d(onset_env, sigma=sigma)
    env_max = float(np.max(onset_env))
    if env_max > 1e-10:
        onset_env /= env_max

    activation *= 0.4 + 0.6 * onset_env
    return np.clip(activation, -amplitude * 1.2, amplitude * 1.2)


def _stride_accent(
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
    """ADSR envelope triggered by accented low-note downbeats, with sign
    alternating per trigger.  Wired to PC2 (anti-sym lateral knee shift):
    +activation bends L knee while extending R (weight on right foot);
    next trigger flips the sign so successive accents lean the body to
    alternating sides.

    Most triggers also coincide with `_identify_step_events`; the global
    rhythm-suppression mask zeros this out during an explicit step so the
    two layers don't fight.  The remaining triggers (low + downbeat but
    not strong enough for an explicit step) still produce a subtle
    side-lean cue.
    """
    bps = bpm / 60.0
    spb = 60.0 / bpm if bpm > 0 else 0.5

    triggers = []
    last_beat = -20.0
    for i in range(n):
        if is_downbeat[i] > 0.5 and accent[i] > 0.2 and is_low_note[i] > 0.5:
            beat_at_i = sample_times[i] * bps
            if beat_at_i - last_beat >= 8.0:
                triggers.append(i)
                last_beat = beat_at_i

    activation = np.zeros(n)
    attack_s = max(int(0.3 * spb / dt), 1)
    hold_s = max(int(hold_beats * spb / dt), 1)
    release_s = max(int(0.5 * spb / dt), 1)

    sign = 1.0
    for ti in triggers:
        amp_signed = sign * amplitude

        for j in range(min(attack_s, n - ti)):
            activation[ti + j] = amp_signed * np.sqrt(j / max(attack_s - 1, 1))

        hold_start = ti + attack_s
        for j in range(min(hold_s, n - hold_start)):
            activation[hold_start + j] = amp_signed * (1.0 - 0.3 * j / hold_s)

        release_start = hold_start + hold_s
        if release_start > 0 and release_start < n:
            peak_at_release = activation[release_start - 1]
            for j in range(min(release_s, n - release_start)):
                activation[release_start + j] = (
                    peak_at_release * (1.0 - j / release_s) ** 2
                )

        sign = -sign

    return np.clip(activation, -amplitude * 1.2, amplitude * 1.2)


def _squat_flex_accent(
    n: int,
    dt: float,
    onset_indices: np.ndarray,
    onset_strength: np.ndarray,
    beat_phase: np.ndarray,
    accent: np.ndarray,
    depth: float,
    decay_time: float = 0.15,
) -> np.ndarray:
    """Exponentially-decaying impulse on each note onset (downbeats deeper).
    Wired to PC3 (sym squat): activation > 0 drives both knees toward
    deeper bend and both ankle pitches toward "back" — a coordinated
    bilateral squat dip on each onset.  Signs intentionally produce
    positive activations only (a flex, never an over-extension).
    """
    activation = np.zeros(n)
    decay_samples = max(int(decay_time / dt), 1)

    for idx in onset_indices:
        if idx >= n:
            continue
        is_downbeat_like = beat_phase[idx] < 0.25
        amp = depth if is_downbeat_like else depth * 0.6
        amp *= 0.7 + 0.3 * onset_strength[idx]
        amp += accent[idx] * depth * 0.4

        for j in range(min(decay_samples, n - idx)):
            activation[idx + j] += amp * np.exp(-j * dt / decay_time)

    return np.clip(activation, 0.0, depth * 2.0)


def _yaw_twist_accent(
    n: int,
    sample_times: np.ndarray,
    dt: float,
    pitch_level: np.ndarray,
    bpm: float,
    time_signature: tuple,
    phrase_boundaries: Set[int],
    amplitude: float,
) -> np.ndarray:
    """Pitch deviation × measure-wave × phrase pulse.  Wired to PC4
    (anti-sym whole-body yaw twist): both pelvic yaws turn together,
    rotating the lower body about the vertical axis to follow the melodic
    contour and punctuate phrase boundaries.
    """
    pitch_mean = float(np.mean(pitch_level))
    pitch_dev = pitch_level - pitch_mean

    beats_per_measure = time_signature[0]
    sec_per_measure = beats_per_measure * 60.0 / bpm if bpm > 0 else 2.0
    measure_wave = np.sin(2 * np.pi * sample_times / sec_per_measure) * 0.4

    phrase_pulse = np.zeros(n)
    pulse_samples = max(int(sec_per_measure / dt), 1)
    for bi in phrase_boundaries:
        if bi < n:
            for j in range(min(pulse_samples, n - bi)):
                phrase_pulse[bi + j] += np.sin(np.pi * j / pulse_samples)
    phrase_pulse *= 0.6

    activation = (pitch_dev * 2.0 + measure_wave + phrase_pulse) * amplitude * 0.6
    return np.clip(activation, -amplitude * 1.5, amplitude * 1.5)


def _phrase_breath_accent(
    n: int,
    sample_times: np.ndarray,
    bpm: float,
    time_signature: tuple,
    phrase_boundaries: Set[int],
    amplitude: float,
) -> np.ndarray:
    """Slow cosine arc spanning each musical phrase.  Wired to PC5
    (sym forward lean): activation > 0 → both hip pitches forward
    (upright), activation < 0 → slight stoop.  Produces a breathing-like
    rise and fall that tracks musical phrasing rather than individual
    notes.
    """
    boundaries = sorted(phrase_boundaries)
    if len(boundaries) == 0 or boundaries[0] != 0:
        boundaries = [0] + list(boundaries)

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
        sec_per_measure = beats_per_measure * 60.0 / bpm if bpm > 0 else 2.0
        activation[:] = np.cos(2 * np.pi * sample_times / (sec_per_measure * 4.0))

    activation *= amplitude * 0.7
    return np.clip(activation, -amplitude * 0.9, amplitude * 0.9)


def _density_sway_accent(
    sample_times: np.ndarray,
    note_density: np.ndarray,
    bpm: float,
    amplitude: float,
    sway_period_beats: float = 48.0,
) -> np.ndarray:
    """Density-modulated slow lateral oscillation.  Wired to PC6
    (anti-sym hip roll body sway): a very slow sine (~`sway_period_beats`
    period) carries the body left/right, and rhythmic note density
    determines how much sway is allowed.  Quiet passages stay near
    upright; dense passages get full sway.

    `note_density` comes from `features.note_density` (pre-smoothed in the
    feature extractor); we consume it directly instead of re-smoothing
    raw onsets here.
    """
    spb = 60.0 / bpm if bpm > 0 else 0.5
    sign_period = sway_period_beats * spb
    sign_osc = np.sin(2 * np.pi * sample_times / sign_period)
    activation = note_density * sign_osc * amplitude * 0.8
    return np.clip(activation, -amplitude * 1.1, amplitude * 1.1)


def _register_extension_accent(
    n: int,
    pitch_level: np.ndarray,
    amplitude: float,
) -> np.ndarray:
    """Pitch register → small body rise/fall.  Wired to PC7 (sym subtle
    extension): high notes (positive activation) lift ankles toward toe-up
    plus slight knee bend; low notes settle the pose down.  `tanh` keeps
    the response bounded for melodies with extreme leaps.
    """
    centered = pitch_level - float(np.mean(pitch_level))
    std = float(np.std(pitch_level))
    if std > 1e-10:
        centered /= std
    activation = np.tanh(centered) * amplitude * 0.6
    return np.clip(activation, -amplitude * 1.0, amplitude * 1.0)


# ---------------------------------------------------------------------------
# Beat-synced groove patterns  (variety & musicality layer)
# ---------------------------------------------------------------------------

# Five groove emphasis modes, cycled across musical phrases.
# Columns: [bounce, double_bounce, twist, sway, pump]
_MODE_WEIGHTS = np.array([
    [1.0, 0.2, 0.5, 0.3, 0.4],   # bounce-heavy
    [0.4, 0.5, 1.0, 0.5, 0.3],   # twist-heavy
    [0.5, 0.3, 0.3, 1.0, 0.6],   # sway-heavy
    [0.6, 0.4, 0.6, 0.4, 1.0],   # pump-heavy
    [0.8, 0.6, 0.7, 0.7, 0.7],   # full groove
])


def _groove_patterns(
    lower_trajs: np.ndarray,
    mean_pose: np.ndarray,
    sample_times: np.ndarray,
    features: MusicalFeatures,
    energy: np.ndarray,
    scale: float,
) -> None:
    """Add beat-synced groove patterns to joint trajectories (in-place).

    Targets only hip (pelvic pitch/roll/yaw), knee, and waist joints.
    Ankle joints are not modified (handled by IK foot-flattening).

    Five patterns are blended via section-based groove mode cycling:
      1. Beat bounce     — knee flex on each beat
      2. Double-time     — eighth-note bounce in high-energy sections
      3. Hip twist       — waist/pelvic yaw on half-note & measure cycles
      4. Body sway       — pelvic roll alternation on 2-beat cycle
      5. Forward pump    — pelvic pitch rocking on beat cycle
    Plus accent-triggered snaps for punctuation.
    """
    n = len(sample_times)
    if n < 2:
        return
    dt = float(sample_times[1] - sample_times[0])
    bpm = features.bpm
    spb = 60.0 / bpm if bpm > 0 else 0.5

    beat_phase = features.beat_phase
    beats = sample_times * bpm / 60.0
    beats_per_measure = max(features.time_signature[0], 1)

    # ---- Section-based groove mode cycling ----
    boundaries = sorted(features.phrase_boundaries)
    if not boundaries or boundaries[0] != 0:
        boundaries = [0] + list(boundaries)

    mode_idx = np.zeros(n, dtype=int)
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else n
        mode_idx[start:end] = (i * 3 + 1) % len(_MODE_WEIGHTS)

    weights = _MODE_WEIGHTS[mode_idx]                       # (n, 5)
    cf_sigma = max(int(spb / dt), 2)
    for j in range(weights.shape[1]):
        weights[:, j] = gaussian_filter1d(weights[:, j], sigma=cf_sigma)

    # ---- Pattern 1: Beat bounce (knee flex on downbeats) ----
    bounce = np.cos(2 * np.pi * beat_phase) * 0.08 * scale * energy * weights[:, 0]
    lower_trajs[:, _L_KNEE] += bounce
    lower_trajs[:, _R_KNEE] += bounce
    lower_trajs[:, _L_HIP_P] -= bounce * 0.35
    lower_trajs[:, _R_HIP_P] -= bounce * 0.35

    # ---- Pattern 2: Double-time bounce (eighth notes, high energy only) ----
    high_energy = np.clip((energy - 0.35) * 2.0, 0.0, 1.0)
    dbl = np.cos(4 * np.pi * beat_phase) * 0.04 * scale * high_energy * weights[:, 1]
    lower_trajs[:, _L_KNEE] += dbl
    lower_trajs[:, _R_KNEE] += dbl

    # ---- Pattern 3: Hip twist (waist/pelvic yaw, half-note & measure) ----
    half_note_phase = (beats % 2.0) / 2.0
    twist_half = np.sin(2 * np.pi * half_note_phase) * 0.06 * scale
    measure_phase = (beats % beats_per_measure) / beats_per_measure
    twist_measure = np.sin(2 * np.pi * measure_phase) * 0.04 * scale
    twist = (twist_half + twist_measure) * energy * weights[:, 2]
    lower_trajs[:, _WAIST] += twist
    lower_trajs[:, _L_HIP_Y] += twist * 0.35
    lower_trajs[:, _R_HIP_Y] += twist * 0.35

    # ---- Pattern 4: Side-to-side sway (pelvic roll, 2-beat cycle) ----
    sway = np.sin(2 * np.pi * half_note_phase) * 0.04 * scale * energy * weights[:, 3]
    lower_trajs[:, _L_HIP_R] += sway
    lower_trajs[:, _R_HIP_R] -= sway

    # ---- Pattern 5: Forward-back pump (pelvic pitch, beat-synced) ----
    pump_phase = 2 * np.pi * beat_phase + np.pi / 6
    pump = np.sin(pump_phase) * 0.05 * scale * energy * weights[:, 4]
    lower_trajs[:, _L_HIP_P] += pump
    lower_trajs[:, _R_HIP_P] += pump

    # ---- Accent snap (quick hip/waist pop on strong accents) ----
    snap = np.zeros(n)
    decay_s = int(0.12 / dt)
    for idx in features.onset_indices:
        if idx >= n:
            continue
        if features.accent[idx] > 0.3:
            amp = features.onset_strength[idx] * 0.05 * scale
            end_idx = min(idx + decay_s, n)
            t_decay = np.arange(end_idx - idx) * dt
            snap[idx:end_idx] += amp * np.exp(-t_decay / 0.08)
    snap *= energy
    snap_dir = np.sign(np.sin(np.pi * beats))
    lower_trajs[:, _WAIST] += snap * snap_dir * 0.8
    lower_trajs[:, _L_HIP_Y] += snap * snap_dir * 0.3
    lower_trajs[:, _R_HIP_Y] += snap * snap_dir * 0.3


# ---------------------------------------------------------------------------
# Explicit stepping accents  (visible foot-lift on accented beats)
# ---------------------------------------------------------------------------

def _identify_step_events(
    sample_times: np.ndarray,
    features: MusicalFeatures,
    energy: np.ndarray,
) -> tuple:
    """Identify step trigger frames and compute foot-lift phase signals.

    Triggers on accented low-note downbeats (>=8 beats apart).  Each step
    is a 1.5-beat bell curve; left and right alternate.

    Returns (left_step, right_step) arrays in [0, 1], where 0 = planted and
    1 = peak of step.  This function does not modify any trajectory; callers
    use the returned signals both to drive step motion and to suppress
    rhythm contributions during the step interval.
    """
    n = len(sample_times)
    left_step = np.zeros(n)
    right_step = np.zeros(n)
    if n < 2:
        return left_step, right_step

    dt = float(sample_times[1] - sample_times[0])
    bpm = features.bpm
    spb = 60.0 / bpm if bpm > 0 else 0.5
    bps = bpm / 60.0

    triggers = []
    last_beat = -20.0
    for i in range(n):
        if (features.is_downbeat[i] > 0.5
                and features.accent[i] > 0.15
                and features.is_low_note[i] > 0.5
                and energy[i] > 0.25):
            beat = sample_times[i] * bps
            if beat - last_beat >= 8.0:
                triggers.append(i)
                last_beat = beat

    step_dur = 1.5 * spb
    step_samples = max(int(step_dur / dt), 1)
    is_left = True

    for ti in triggers:
        step_arr = left_step if is_left else right_step
        for j in range(min(step_samples, n - ti)):
            bell = np.sin(np.pi * j / step_samples) ** 2
            step_arr[ti + j] = max(step_arr[ti + j], bell)
        is_left = not is_left

    sm = max(int(0.03 / dt), 1)
    if np.any(left_step > 0):
        left_step = gaussian_filter1d(left_step, sigma=sm)
    if np.any(right_step > 0):
        right_step = gaussian_filter1d(right_step, sigma=sm)

    return np.clip(left_step, 0.0, 1.0), np.clip(right_step, 0.0, 1.0)


def _rhythm_suppression_mask(
    step_active: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Build a [0, 1] mask that is 0 during stepping (rhythm suppressed) and
    1 elsewhere (full rhythm).  The mask widens the raw step phase so rhythm
    is silenced for a brief lead-in / tail-out around each step, then ramps
    smoothly back to full activity."""
    if not np.any(step_active > 0):
        return np.ones_like(step_active)

    widen_sigma = max(int(0.25 / dt), 1)
    expanded = gaussian_filter1d(step_active, sigma=widen_sigma)
    expanded = np.clip(expanded * 3.0, 0.0, 1.0)
    ramp_sigma = max(int(0.12 / dt), 1)
    expanded = gaussian_filter1d(expanded, sigma=ramp_sigma)
    return 1.0 - np.clip(expanded, 0.0, 1.0)


def _apply_step_motion(
    lower_trajs: np.ndarray,
    left_step: np.ndarray,
    right_step: np.ndarray,
    scale: float,
) -> None:
    """Add asymmetric knee bend + hip flex on the lifting leg (in-place).

    Applied AFTER rhythm/groove have been masked off, so the step is the
    only motion happening during a step phase.  Amplitudes are larger than
    before so the lift is clearly visible.
    """
    extra_knee = 0.55 * scale
    hip_flex = 0.20 * scale

    lower_trajs[:, _L_KNEE] += extra_knee * left_step
    lower_trajs[:, _L_HIP_P] -= hip_flex * left_step

    lower_trajs[:, _R_KNEE] += extra_knee * right_step
    lower_trajs[:, _R_HIP_P] -= hip_flex * right_step


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

    # -- Musical energy envelope (0-1, scales carrier amplitude).
    #    Use the pre-computed, metric-aware energy from feature extraction
    #    (≈0.6·smoothed_note_density + 0.4·smoothed_accent).  If an older
    #    MusicalFeatures lacking this field is passed in, fall back to
    #    smoothed onset_strength so external callers don't break.
    energy = getattr(features, "energy", None)
    if energy is None or len(energy) != n:
        spb = 60.0 / features.bpm if features.bpm > 0 else 0.5
        smooth_sigma = max(int(2.0 * spb / dt), 2)
        energy = gaussian_filter1d(features.onset_strength, sigma=smooth_sigma)
        e_max = float(np.max(energy))
        if e_max > 1e-10:
            energy = energy / e_max

    # -- Step events: identified up front so rhythm can be suppressed --
    left_step, right_step = _identify_step_events(sample_times, features, energy)
    step_active = np.maximum(left_step, right_step)
    rhythm_mask = _rhythm_suppression_mask(step_active, dt)
    masked_energy = energy * rhythm_mask

    # -- Build activation matrix --
    # activation_i = carrier_i × envelope × std_score_i × CARRIER_GAIN
    # The energy envelope is exponentiated (γ > 1) to widen the dynamic
    # range so quiet passages stay subtle while loud ones bloom.
    activations = np.zeros((n, n_comp))
    base_level = 0.05
    dynamic_range = 0.95
    envelope_gamma = 1.5
    CARRIER_GAIN = 2.4   # boost slightly: detrended std_scores are ~0.4× of old

    envelope = base_level + dynamic_range * (energy ** envelope_gamma)
    for i in range(n_comp):
        activations[:, i] = carriers[:, i] * envelope * amps[i] * CARRIER_GAIN

    # -- Event-driven accents: each musical driver wired to the PC whose
    #    natural coordination matches its motion (see module docstring) --
    if n_comp >= 1:
        # PC1: anti-sym weight rock ← beat-phase sine
        activations[:, 0] += _beat_rock_accent(
            n, dt, features.beat_phase, features.onset_indices,
            features.onset_strength,
            amplitude=amps[0] * 1.5,
        )

    if n_comp >= 2:
        # PC2: anti-sym lateral step ← accented-onset stride ADSR
        activations[:, 1] += _stride_accent(
            n, sample_times, dt, features.accent, features.is_low_note,
            features.is_downbeat, features.bpm,
            amplitude=amps[1] * 1.5,
        )

    if n_comp >= 3:
        # PC3: sym squat ← onset knee-flex impulse (positive = bend)
        activations[:, 2] += _squat_flex_accent(
            n, dt, features.onset_indices, features.onset_strength,
            features.beat_phase, features.accent,
            depth=amps[2] * 2.0,
        )

    if n_comp >= 4:
        # PC4: anti-sym whole-body yaw twist ← pitch + measure wave
        activations[:, 3] += _yaw_twist_accent(
            n, sample_times, dt, features.pitch_level, features.bpm,
            features.time_signature, features.phrase_boundaries,
            amplitude=amps[3] * 1.5,
        )

    if n_comp >= 5:
        # PC5: sym forward lean ← phrase breathing arc
        activations[:, 4] += _phrase_breath_accent(
            n, sample_times, features.bpm, features.time_signature,
            features.phrase_boundaries,
            amplitude=amps[4] * 1.4,
        )

    if n_comp >= 6:
        # PC6: anti-sym hip-roll body sway ← density × slow sine
        note_density = getattr(features, "note_density", None)
        if note_density is None or len(note_density) != n:
            note_density = energy
        activations[:, 5] += _density_sway_accent(
            sample_times, note_density, features.bpm,
            amplitude=amps[5] * 1.4,
        )

    if n_comp >= 7:
        # PC7: sym subtle extension ← pitch register
        activations[:, 6] += _register_extension_accent(
            n, features.pitch_level,
            amplitude=amps[6] * 1.2,
        )

    # -- Suppress all rhythm activations during step intervals --
    activations *= rhythm_mask[:, np.newaxis]

    # -- Reconstruct: (n, 13) = mean + activations @ components --
    lower_trajs = mean_pose[np.newaxis, :] + activations @ components

    # -- Feet planted: zero ankle motion, let IK handle foot-flattening --
    for idx in _ANKLE_INDICES:
        lower_trajs[:, idx] = mean_pose[idx]

    # -- Beat-synced groove patterns (energy gated by rhythm_mask) --
    _groove_patterns(
        lower_trajs, mean_pose, sample_times, features, masked_energy, scale,
    )

    # -- Hip pitch constraint: prevent bass-thigh collision --
    _HIP_PITCH_MIN = -0.35
    lower_trajs[:, _L_HIP_P] = np.maximum(lower_trajs[:, _L_HIP_P], _HIP_PITCH_MIN)
    lower_trajs[:, _R_HIP_P] = np.maximum(lower_trajs[:, _R_HIP_P], _HIP_PITCH_MIN)

    # -- Apply step motion last so it's the only motion during a step --
    _apply_step_motion(lower_trajs, left_step, right_step, scale)

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

    result["left_foot_step"] = left_step
    result["right_foot_step"] = right_step

    return result
