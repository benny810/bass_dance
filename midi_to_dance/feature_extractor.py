"""Extract per-sample musical features from parsed MIDI data.

The output `MusicalFeatures` is consumed by `pca_motion.py` to drive the
PCA-based dance trajectory.  Each feature is sampled at a uniform `dt`
(default 0.02 s = 50 Hz) and aligned with the same `sample_times` axis,
so downstream code can index any feature by frame.

Algorithm highlights / fixes over the prior version:

  1.  **Sustained pitch & low-note** — previously only the single onset
      frame of each note carried information, so after Gaussian smoothing
      `pitch_level` collapsed to ~0.01 and `is_low_note` was almost never
      true.  We now fill these signals across each note's `[onset, offset)`
      interval, giving correct sustained values whose smoothed form
      reflects actual melodic motion.

  2.  **Layered accent** — quantised MIDIs have near-flat velocity
      (std ≈ 2 across the 4 example songs), so velocity-only accent
      produces random / sparse signals.  The new accent combines:
        - metric position (downbeat ≫ on-beat ≫ off-beat)
        - velocity deviation (when it varies)
        - duration anomaly (long notes feel accented)
        - pitch leap from the previous note
      Mixed with weights tuned so a typical bass downbeat gets accent ≈ 0.5
      even on flat-velocity material.

  3.  **Note density envelope** — exposed as `note_density` and
      `energy` (≈ density × accent, smoothed over ~2 beats).  `pca_motion`
      can use `energy` directly instead of recomputing.

  4.  **Vectorised beat / phase / downbeat** — single-pass numpy ops.

  5.  **Richer phrase boundaries** — combines gap-based detection with
      metric-grid candidates (every N measures, default 8) and silence
      regions (>= 3-beat note-free spans), then dedupes by proximity.

  6.  **Tempo-aware timing** — note start times come from `NoteEvent.
      time_seconds` (tempo-map-resolved by the parser), so tempo changes
      don't bias feature alignment.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .midi_parser import MidiData, NoteEvent


@dataclass
class MusicalFeatures:
    """Per-sample musical features.  All arrays are aligned on `sample_times`."""
    sample_times: np.ndarray      # seconds, shape (n,)
    bpm: float                    # dominant BPM from parser
    time_signature: Tuple[int, int]

    # Onset / rhythm
    onset_strength: np.ndarray    # 0–1 spike at note onsets (max-merged per dt)
    note_density: np.ndarray      # smoothed notes/beat (0–1 normalised)
    energy: np.ndarray            # smoothed musical energy (0–1)

    # Beat structure
    beat_phase: np.ndarray        # position within beat in [0, 1)
    is_beat: np.ndarray           # 1.0 at any beat boundary, else 0.0
    is_downbeat: np.ndarray       # 1.0 at measure downbeats, else 0.0
    metric_accent: np.ndarray     # 0–1 pure-metric accent at each sample

    # Melody
    pitch_level: np.ndarray       # sustained normalised pitch (0=lowest, 1=highest)
    pitch_contour: np.ndarray     # local slope of pitch_level, clipped to [-1, 1]
    accent: np.ndarray            # 0–1 composite accent (metric + velocity + …)
    is_low_note: np.ndarray       # 1.0 whenever a low note is sounding

    # Event indices
    onset_indices: np.ndarray     # frame indices where onsets occur
    phrase_boundaries: Set[int]   # frame indices of phrase starts (always incl. 0)


# ---------------------------------------------------------------------------
# Time-signature-aware metric accent profile
# ---------------------------------------------------------------------------

def _metric_accent_for_beat(beat_in_measure: float, beats_per_measure: int) -> float:
    """Return the metric-accent weight in [0, 1] for a position within a
    measure (0 = downbeat).  Off-beats decay smoothly so a slightly-late
    note doesn't fall off a cliff.
    """
    bpm_int = max(beats_per_measure, 1)
    nearest = round(beat_in_measure) % bpm_int
    off = min(
        abs(beat_in_measure - nearest),
        abs(beat_in_measure - (nearest + bpm_int)),
    )
    # Strong/weak hierarchy per common time signatures.
    if bpm_int == 4:
        hierarchy = [1.0, 0.30, 0.65, 0.30]
    elif bpm_int == 3:
        hierarchy = [1.0, 0.40, 0.40]
    elif bpm_int == 2:
        hierarchy = [1.0, 0.45]
    elif bpm_int == 6:
        hierarchy = [1.0, 0.25, 0.40, 0.60, 0.25, 0.40]
    else:
        hierarchy = [1.0] + [0.4 if i % 2 == 0 else 0.25 for i in range(1, bpm_int)]
    on_beat = hierarchy[nearest]
    # Smoothly fall off when the note is between beats.
    falloff = np.exp(-(off * 6.0) ** 2)
    return float(on_beat * falloff)


def _build_metric_accent_array(
    sample_times: np.ndarray,
    bpm: float,
    beats_per_measure: int,
) -> np.ndarray:
    """Vectorised metric-accent over all samples (one value per dt frame)."""
    bps = bpm / 60.0 if bpm > 0 else 2.0
    beats = sample_times * bps
    beats_in_measure = beats % max(beats_per_measure, 1)
    # Compute per-sample metric accent by routing through scalar helper —
    # the helper is cheap and unique values per song are few, but we just
    # call it elementwise for clarity.  n ~ 8000 in practice.
    out = np.fromiter(
        (_metric_accent_for_beat(b, beats_per_measure) for b in beats_in_measure),
        dtype=np.float64,
        count=len(sample_times),
    )
    return out


# ---------------------------------------------------------------------------
# BPM-only feature synthesis (no MIDI)
# ---------------------------------------------------------------------------

def synthesize_features_from_bpm(
    bpm: float,
    duration: float,
    dt: float = 0.02,
    time_signature: Tuple[int, int] = (4, 4),
) -> MusicalFeatures:
    """Synthesize a `MusicalFeatures` driven only by a metronomic beat grid.

    No MIDI file is required — onsets are placed at every beat and downbeat
    with synthetic velocities, and all derivative features (energy, accent,
    phrase boundaries) are built from this uniform grid.  The result feeds
    the same `pca_motion.generate_pca_motion()` pipeline unchanged.
    """
    bpm = float(max(bpm, 1.0))
    beats_per_measure = max(time_signature[0], 1)
    bps = bpm / 60.0
    spb = 60.0 / bpm

    n_samples = int(np.ceil(duration / dt)) + 1
    sample_times = np.arange(n_samples) * dt

    # -- Beat / downbeat grid --
    beats = sample_times * bps
    nearest_beat = np.round(beats)
    dist_to_beat = np.abs(beats - nearest_beat)
    beat_window = dt * bps * 0.5 + 1e-9
    is_beat = (dist_to_beat < beat_window).astype(np.float64)
    is_downbeat = is_beat * (
        nearest_beat.astype(np.int64) % beats_per_measure == 0
    )
    beat_phase = beats % 1.0

    # -- Synthetic onsets at each beat (velocity=100), boosted at downbeats (110) --
    onset_strength = np.zeros(n_samples)
    onset_strength[is_beat > 0.5] = 100.0 / 127.0
    onset_strength[is_downbeat > 0.5] = 110.0 / 127.0

    onset_indices = np.where(is_beat > 0.5)[0].astype(np.int64)

    # -- Metric accent --
    metric_accent = _build_metric_accent_array(sample_times, bpm, beats_per_measure)

    # -- Accent: metric_accent only (no velocity/deviation/duration/leap data) --
    accent = metric_accent.copy()

    # -- Note density & energy (smoothed onset track, ~1.5 beats wide) --
    onset_count = np.zeros(n_samples)
    onset_count[is_beat > 0.5] = 1.0
    density_sigma = max(int(1.5 * spb / dt), 3)
    note_density = gaussian_filter1d(onset_count, sigma=density_sigma)
    nd_max = float(note_density.max())
    if nd_max > 1e-9:
        note_density /= nd_max

    energy = note_density.copy()

    # -- Pitch level: constant mid-register (no melodic variation) --
    pitch_level = np.full(n_samples, 0.5)

    # -- Pitch contour & low-note: flat --
    pitch_contour = np.zeros(n_samples)
    is_low_note = np.zeros(n_samples)

    # -- Phrase boundaries: every 8 measures --
    sec_per_measure = beats_per_measure * spb
    section_seconds = sec_per_measure * 8
    phrase_boundaries: Set[int] = {0}
    if section_seconds > 0:
        t = section_seconds
        while t < duration:
            idx = int(round(t / dt))
            if 0 <= idx < n_samples:
                phrase_boundaries.add(idx)
            t += section_seconds

    return MusicalFeatures(
        sample_times=sample_times,
        bpm=bpm,
        time_signature=time_signature,
        onset_strength=onset_strength,
        note_density=note_density,
        energy=energy,
        beat_phase=beat_phase,
        is_beat=is_beat,
        is_downbeat=is_downbeat,
        metric_accent=metric_accent,
        pitch_level=pitch_level,
        pitch_contour=pitch_contour,
        accent=accent,
        is_low_note=is_low_note,
        onset_indices=onset_indices,
        phrase_boundaries=phrase_boundaries,
    )


# ---------------------------------------------------------------------------
# Main feature extraction
# ---------------------------------------------------------------------------

def extract_features(midi_data: MidiData, dt: float = 0.02) -> MusicalFeatures:
    """Extract dance-driving features from `MidiData` at sample period `dt`."""
    total_time = max(midi_data.total_duration_seconds, dt)
    n_samples = int(np.ceil(total_time / dt)) + 1
    sample_times = np.arange(n_samples) * dt

    bpm = midi_data.bpm if midi_data.bpm > 0 else 120.0
    beats_per_second = bpm / 60.0
    seconds_per_beat = 60.0 / bpm
    beats_per_measure = max(midi_data.time_signature[0], 1)

    notes: List[NoteEvent] = list(midi_data.notes)

    # Trivial early exit so empty MIDIs don't crash downstream.
    if not notes:
        zeros = np.zeros(n_samples)
        return MusicalFeatures(
            sample_times=sample_times, bpm=bpm,
            time_signature=midi_data.time_signature,
            onset_strength=zeros.copy(), note_density=zeros.copy(),
            energy=zeros.copy(),
            beat_phase=(sample_times * beats_per_second) % 1.0,
            is_beat=zeros.copy(), is_downbeat=zeros.copy(),
            metric_accent=zeros.copy(),
            pitch_level=zeros.copy(), pitch_contour=zeros.copy(),
            accent=zeros.copy(), is_low_note=zeros.copy(),
            onset_indices=np.array([], dtype=np.int64),
            phrase_boundaries={0},
        )

    # -- Pitch range over the whole song (for normalisation) --
    pitches = np.array([n.note for n in notes], dtype=np.int32)
    velocities = np.array([n.velocity for n in notes], dtype=np.int32)
    durations = np.array([n.duration_seconds for n in notes], dtype=np.float64)

    min_pitch = int(pitches.min())
    max_pitch = int(pitches.max())
    pitch_span = max(max_pitch - min_pitch, 1)
    # Bottom 40% of the pitch range counts as "low".
    low_pitch_thresh = min_pitch + 0.4 * pitch_span

    mean_vel = float(velocities.mean())
    vel_span = float(velocities.max() - mean_vel) + 1e-9
    median_dur = float(np.median(durations)) if len(durations) > 0 else 0.0

    # -- Per-note frame indices --
    starts_s = np.array([n.time_seconds for n in notes], dtype=np.float64)
    ends_s = starts_s + np.maximum(durations, dt)   # at least one frame wide
    start_idx = np.clip(np.round(starts_s / dt).astype(np.int64), 0, n_samples - 1)
    end_idx = np.clip(np.round(ends_s / dt).astype(np.int64), 0, n_samples)

    # -- Onset spikes (max-merged when 2 onsets collide on one frame) --
    onset_strength = np.zeros(n_samples)
    np.maximum.at(onset_strength, start_idx, velocities / 127.0)

    # -- Sustained pitch & low-note signals: fill across each note --
    pitch_norm_per_note = (pitches - min_pitch) / pitch_span
    is_low_per_note = (pitches <= low_pitch_thresh).astype(np.float64)
    pitch_level_raw = np.zeros(n_samples)
    is_low_note = np.zeros(n_samples)
    for i in range(len(notes)):
        s, e = int(start_idx[i]), int(max(end_idx[i], start_idx[i] + 1))
        pitch_level_raw[s:e] = np.maximum(pitch_level_raw[s:e], pitch_norm_per_note[i])
        if is_low_per_note[i] > 0.5:
            is_low_note[s:e] = 1.0
    # Light smoothing on pitch level (~0.25 beat) so the polyphonic-max
    # jumps don't translate into pitch_contour spikes.
    pitch_sigma = max(int(0.25 * seconds_per_beat / dt), 2)
    pitch_level = np.clip(
        gaussian_filter1d(pitch_level_raw, sigma=pitch_sigma), 0.0, 1.0,
    )
    pitch_contour_raw = np.gradient(pitch_level) * (1.0 / dt) * seconds_per_beat
    # Normalise per-beat slope to [-1, 1] then smooth.
    pitch_contour = np.clip(pitch_contour_raw / 1.0, -1.0, 1.0)
    pitch_contour = gaussian_filter1d(pitch_contour, sigma=pitch_sigma)

    # -- Beat / downbeat: vectorised nearest-beat detection --
    beats = sample_times * beats_per_second
    nearest_beat = np.round(beats)
    dist_to_beat = np.abs(beats - nearest_beat)
    beat_window = dt * beats_per_second * 0.5 + 1e-9
    is_beat = (dist_to_beat < beat_window).astype(np.float64)
    is_downbeat = is_beat * (nearest_beat.astype(np.int64) % beats_per_measure == 0)
    beat_phase = beats % 1.0

    # -- Metric accent per sample --
    metric_accent = _build_metric_accent_array(sample_times, bpm, beats_per_measure)

    # -- Per-note metric accent (used for composite accent) --
    note_beat_in_measure = (np.array([n.beat for n in notes]) % beats_per_measure)
    note_metric = np.array(
        [_metric_accent_for_beat(b, beats_per_measure) for b in note_beat_in_measure]
    )

    # -- Composite per-note accent (metric + velocity + duration + leap) --
    velocity_dev = np.clip((velocities - mean_vel) / vel_span, 0.0, 1.0)
    if median_dur > 1e-9:
        duration_ratio = np.clip(durations / median_dur - 1.0, 0.0, 2.0) / 2.0
    else:
        duration_ratio = np.zeros_like(durations)
    # Pitch leap: |Δsemitone| from previous note, normalised by an octave.
    leap_per_note = np.zeros(len(notes))
    if len(notes) > 1:
        leap_per_note[1:] = np.clip(np.abs(np.diff(pitches)) / 12.0, 0.0, 1.0)
    note_accent = np.clip(
        0.50 * note_metric
        + 0.25 * velocity_dev
        + 0.15 * duration_ratio
        + 0.10 * leap_per_note,
        0.0,
        1.0,
    )

    # -- Sample accent: place each note's accent on its onset frame.
    accent = np.zeros(n_samples)
    np.maximum.at(accent, start_idx, note_accent)
    # Brief decay so the accent persists ~80 ms beyond onset, surviving
    # the dt grid even when other features look at neighbouring frames.
    decay_sigma = max(int(0.04 / dt), 1)
    accent = gaussian_filter1d(accent, sigma=decay_sigma)
    a_max = float(accent.max())
    if a_max > 1e-9:
        accent /= a_max

    # -- Note density & energy envelope --
    # Use a Hann-like onset impulse track (sum of per-frame onset counts
    # weighted by accent), smoothed over ~1.5 beats.
    onset_count = np.zeros(n_samples)
    np.add.at(onset_count, start_idx, 1.0)
    density_sigma = max(int(1.5 * seconds_per_beat / dt), 3)
    note_density = gaussian_filter1d(onset_count, sigma=density_sigma)
    nd_max = float(note_density.max())
    if nd_max > 1e-9:
        note_density /= nd_max

    energy_raw = 0.6 * note_density + 0.4 * gaussian_filter1d(
        accent, sigma=density_sigma,
    )
    e_max = float(energy_raw.max())
    energy = energy_raw / e_max if e_max > 1e-9 else energy_raw

    # -- Phrase boundaries (gap-based ∪ metric-grid ∪ silence) --
    phrase_boundaries = _detect_phrase_boundaries(
        n_samples=n_samples,
        dt=dt,
        start_idx=start_idx,
        sample_times=sample_times,
        bpm=bpm,
        beats_per_measure=beats_per_measure,
        energy=energy,
    )

    # -- Onset indices (unique frame indices of onsets) --
    onset_indices = np.array(sorted(set(start_idx.tolist())), dtype=np.int64)

    return MusicalFeatures(
        sample_times=sample_times,
        bpm=bpm,
        time_signature=midi_data.time_signature,
        onset_strength=onset_strength,
        note_density=note_density,
        energy=energy,
        beat_phase=beat_phase,
        is_beat=is_beat,
        is_downbeat=is_downbeat,
        metric_accent=metric_accent,
        pitch_level=pitch_level,
        pitch_contour=pitch_contour,
        accent=accent,
        is_low_note=is_low_note,
        onset_indices=onset_indices,
        phrase_boundaries=phrase_boundaries,
    )


# ---------------------------------------------------------------------------
# Phrase-boundary detection
# ---------------------------------------------------------------------------

def _detect_phrase_boundaries(
    *,
    n_samples: int,
    dt: float,
    start_idx: np.ndarray,
    sample_times: np.ndarray,
    bpm: float,
    beats_per_measure: int,
    energy: np.ndarray,
    measures_per_section: int = 8,
    min_gap_beats: float = 2.0,
    min_separation_seconds: float = 4.0,
) -> Set[int]:
    """Combine onset-gap, silence, and metric-grid candidates into a
    deduped set of frame indices that approximate musical phrase starts.
    """
    seconds_per_beat = 60.0 / bpm if bpm > 0 else 0.5
    candidates: List[int] = [0]

    # 1) Onset-gap boundaries: a gap of ≥ min_gap_beats followed by a new
    #    onset is a phrase start at that onset.
    if len(start_idx) >= 2:
        sorted_idx = np.unique(start_idx)
        gaps_s = np.diff(sorted_idx) * dt
        gaps_beats = gaps_s / seconds_per_beat
        for i, gap_b in enumerate(gaps_beats):
            if gap_b >= min_gap_beats:
                candidates.append(int(sorted_idx[i + 1]))

    # 2) Metric-grid candidates: every `measures_per_section` measures.
    sec_per_measure = beats_per_measure * seconds_per_beat
    section_seconds = sec_per_measure * measures_per_section
    if section_seconds > 0:
        t = section_seconds
        while t < sample_times[-1]:
            idx = int(round(t / dt))
            if 0 <= idx < n_samples:
                candidates.append(idx)
            t += section_seconds

    # 3) Silence-based candidates: regions where energy drops below a
    #    low threshold for ≥ 2 beats, with the boundary at the energy
    #    return point.
    low_thresh = max(0.05, float(np.quantile(energy, 0.2)))
    quiet = energy < low_thresh
    if quiet.any():
        # Find rising edges (quiet → loud) — these are phrase entries.
        rising = np.where((~quiet[1:]) & (quiet[:-1]))[0] + 1
        for r in rising:
            candidates.append(int(r))

    # Dedup + enforce minimum separation.
    candidates.sort()
    min_sep = max(int(min_separation_seconds / dt), 1)
    deduped: List[int] = []
    last = -10 * min_sep
    for c in candidates:
        if c - last >= min_sep:
            deduped.append(c)
            last = c
    return set(deduped)
