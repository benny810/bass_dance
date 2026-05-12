"""Extract musical features from parsed MIDI data for driving dance motions."""

from dataclasses import dataclass
from typing import List, Dict, Set
import numpy as np
from .midi_parser import MidiData, NoteEvent


@dataclass
class MusicalFeatures:
    """Musical features extracted from MIDI data at a given time resolution."""
    sample_times: np.ndarray      # time in seconds for each sample point
    bpm: float
    time_signature: tuple

    # Per-sample features
    onset_strength: np.ndarray    # weighted note onset intensity at each sample
    beat_phase: np.ndarray        # position within beat (0.0 to 1.0)
    is_downbeat: np.ndarray       # 1.0 at measure downbeats, 0.0 elsewhere
    is_beat: np.ndarray           # 1.0 at any beat boundary
    pitch_contour: np.ndarray     # smoothed pitch change trend (-1 to 1)
    pitch_level: np.ndarray       # smoothed absolute pitch level (0 to 1)
    accent: np.ndarray            # accent strength (0 to 1)
    is_low_note: np.ndarray       # 1.0 when low notes (F2/A2) are sounding

    # Event lists (indices into sample_times)
    onset_indices: np.ndarray     # indices where notes start
    phrase_boundaries: Set[int]   # indices of phrase boundaries


def extract_features(midi_data: MidiData, dt: float = 0.02) -> MusicalFeatures:
    """Extract musical features from MIDI data at given time resolution."""
    total_time = midi_data.total_duration_seconds
    n_samples = int(total_time / dt) + 1
    sample_times = np.arange(n_samples) * dt

    beats_per_second = midi_data.bpm / 60.0
    seconds_per_beat = 60.0 / midi_data.bpm

    onset_strength = np.zeros(n_samples)
    accent = np.zeros(n_samples)
    pitch_values = np.zeros(n_samples)
    is_low_note = np.zeros(n_samples)
    onset_events = []  # list of (sample_idx, velocity)

    note_numbers = [n.note for n in midi_data.notes]
    velocities = [n.velocity for n in midi_data.notes]
    mean_vel = np.mean(velocities)
    std_vel = np.std(velocities)
    accent_threshold = mean_vel + 0.5 * std_vel

    min_note = min(note_numbers)
    max_note = max(note_numbers)
    low_note_threshold = min_note + (max_note - min_note) * 0.4

    for note in midi_data.notes:
        onset_time = note.beat * seconds_per_beat
        idx = int(onset_time / dt)
        if idx >= n_samples:
            continue

        # Onset strength: velocity scaled to 0-1
        strength = note.velocity / 127.0
        onset_strength[idx] = max(onset_strength[idx], strength)
        onset_events.append(idx)

        # Accent detection
        if note.velocity >= accent_threshold:
            accent[idx] = max(accent[idx], (note.velocity - mean_vel) / (max(velocities) - mean_vel + 1e-9))

        # Pitch contour: fill pitch values at onset and create smoothed contour
        pitch_norm = (note.note - min_note) / (max_note - min_note + 1e-9)
        pitch_values[idx] = pitch_norm

        # Low note detection
        if note.note <= low_note_threshold:
            is_low_note[idx] = 1.0

    # Smooth pitch values to create contour
    if n_samples > 10:
        sigma = int(seconds_per_beat * 2 / dt)  # smooth over ~2 beats
        sigma = max(sigma, 3)
        from scipy.ndimage import gaussian_filter1d
        pitch_level = gaussian_filter1d(pitch_values, sigma=sigma)
        pitch_level = np.clip(pitch_level, 0, 1)
        pitch_contour = np.gradient(pitch_level)
        pitch_contour = np.clip(pitch_contour * 10, -1, 1)  # amplify small changes
        # Also smooth the contour itself
        pitch_contour = gaussian_filter1d(pitch_contour, sigma=sigma)
    else:
        pitch_level = np.zeros(n_samples)
        pitch_contour = np.zeros(n_samples)

    # Beat phase (where we are within each beat, 0.0 to 1.0)
    beat_phase = (sample_times * beats_per_second) % 1.0

    # Beat markers
    is_beat = np.zeros(n_samples)
    is_downbeat = np.zeros(n_samples)
    beats_per_measure = midi_data.time_signature[0]
    for i, t in enumerate(sample_times):
        beat_num = t * beats_per_second
        dist_to_beat = abs(beat_num - round(beat_num))
        if dist_to_beat < dt * beats_per_second * 0.5:
            is_beat[i] = 1.0
            if round(beat_num) % beats_per_measure == 0:
                is_downbeat[i] = 1.0

    # Detect phrase boundaries: look for gaps in onsets >= 2 beats
    phrase_boundaries = set()
    if onset_events:
        onset_events_sorted = sorted(set(onset_events))
        for i in range(1, len(onset_events_sorted)):
            gap_samples = onset_events_sorted[i] - onset_events_sorted[i - 1]
            gap_beats = gap_samples * dt * beats_per_second
            if gap_beats >= 2.0:
                phrase_boundaries.add(onset_events_sorted[i])

    return MusicalFeatures(
        sample_times=sample_times,
        bpm=midi_data.bpm,
        time_signature=midi_data.time_signature,
        onset_strength=onset_strength,
        beat_phase=beat_phase,
        is_downbeat=is_downbeat,
        is_beat=is_beat,
        pitch_contour=pitch_contour,
        pitch_level=pitch_level,
        accent=accent,
        is_low_note=is_low_note,
        onset_indices=np.array(sorted(set(onset_events))),
        phrase_boundaries=phrase_boundaries,
    )
