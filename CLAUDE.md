# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

MIDI-to-Dance: generates robot (CASBOT 02) lower-body dance trajectories from MIDI files, then visualizes in MuJoCo with real-time synced audio. The robot has 27 DOFs — 13 lower-body joints are PCA-generated from mocap data, 14 arm joints stay at a fixed bass-playing pose.

## Commands

```bash
# Install dependencies
pip install mido numpy scipy mujoco matplotlib

# (One-time) Extract PCA model from mocap CSVs (high-pass detrend, static-frame drop,
# mirror augmentation by default; see README / pca_extractor.py for --hp-cutoff, etc.)
python -m midi_to_dance.pca_extractor csv/example1.csv csv/example2.csv -o pca_model.npz

# Generate dance trajectory from MIDI (pure PCA, no leg-lift)
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv --scale 1.2 --plot --stats

# Per-PC amplitude weights (length matches n_components in pca_model.npz; pads/truncates as needed)
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv \
    --pc-weights 1.0 1.0 1.8 0.4 1.0 1.0 1.0   # boost PC3 (squat), damp PC4 (yaw)

# Opt-in to the hand-coded single-leg-lift layer (NOT a PCA primitive; OFF by default)
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv --enable-steps

# MuJoCo simulation (kinematic mode, audio synced)
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid

# Dynamics mode (PD actuators + physics)
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --dynamics

# Half-speed, no audio, or save audio to WAV
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --slow 0.5 --no-audio
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --save-audio output.wav

# Audio/motion sync compensation (seconds; positive delays motion vs audio)
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --audio-offset 0.05
```

## Architecture

Data flows through a sequential pipeline (`main.py` is the entry point):

```
MIDI file  →  midi_parser.py (MidiData)  →  feature_extractor.py (MusicalFeatures)
  →  pca_motion.py (PCA joint angles)  →  trajectory_generator.py (clamp + smooth)
  →  trajectory_writer.py (CSV)  →  simulate.py (MuJoCo viewer + audio)
```

### Key modules

- **`midi_parser.py`** — Parses MIDI via `mido`, returns `MidiData` (`bpm`, `time_signature`, `tempo_map`, list of `NoteEvent`). Maintains a full tempo map across tracks and resolves each note's `time_seconds` / `duration_seconds` via piecewise `tick → seconds` integration (correct under mid-song tempo changes). `bpm` is the duration-weighted dominant tempo; `time_signature` is the FIRST event encountered (conductor-track convention). Note_on/note_off pairs are matched with a per-pitch FIFO so re-attacks before release are handled.
- **`feature_extractor.py`** — Converts `MidiData` → `MusicalFeatures` at sample period `dt` (default 0.02 s). Key signals: `onset_strength` (max-merged velocity per frame), `note_density` (smoothed onset count), `energy` (≈0.6·density + 0.4·smoothed accent), `beat_phase` / `is_beat` / `is_downbeat` (vectorised), `metric_accent` (time-signature-aware 0–1 weight per sample), `pitch_level` (filled across each note's `[onset, offset)`, then smoothed — fixes the prior bug where smoothing a sparse onset-only signal collapsed it to ~0.01), `pitch_contour` (slope of `pitch_level`), `accent` (composite: 0.50·metric + 0.25·velocity_dev + 0.15·duration_anomaly + 0.10·pitch_leap, robust to flat-velocity quantised MIDIs), `is_low_note` (also sustained for the note duration), `phrase_boundaries` (union of onset-gap ≥ 2 beats, every-N-measures metric grid, and silence-recovery edges, deduped by minimum separation).
- **`pca_extractor.py`** — Offline primitive extraction from `example*.csv`. Pipeline (defaults): load 13 lower-body joints → Butterworth high-pass detrend (0.3 Hz, zero-phase, mean reattached) → drop lowest-speed frames (10th percentile on high-pass velocity) → left-right mirror augmentation (swap legs + flip lateral joint signs) → mean-center + SVD → sign-canonical PCs by variance order; optional `--canonical` reorders PCs to semantic prototypes via Hungarian `|cosine|` matching for tighter alignment with hard-coded PC-slot accents in `pca_motion.py`. Saves `pca_model.npz` (`mean_pose`, `components`, `std_scores`, `explained_variance_ratio`, `joint_names`, counts).
- **`pca_motion.py`** — The core motion algorithm. Reconstructs joint angles as `mean_pose + Σ(activation_i × component_i)`. The default pipeline runs three layers per frame; a fourth (step) layer is opt-in via `enable_steps=True`:
  1. **Continuous carriers** — 7 PC-specific sums of 3 incommensurate sines (faster periods for high-variance rock/step PCs, slower for lean/extension PCs); modulated by the pre-computed `features.energy` envelope from `feature_extractor.py` (γ-shaped via `energy ** 1.5`, baseline 0.05, range 0.95) scaled by each PC's `std_score × scale × pc_weights[i] × CARRIER_GAIN`. Falls back to smoothed `features.onset_strength` if the upstream energy field is missing.
  2. **Event accents** — musical drivers wired to the PC whose default-variance semantic they match: PC1 anti-sym weight rock ← `_beat_rock_accent` (beat-phase sine × onset density); PC2 anti-sym lateral lean ← `_stride_accent` (accented low-downbeat ADSR with ~40 ms instant attack so the peak lands ON the trigger frame; sign alternates per trigger); PC3 sym squat ← `_squat_flex_accent` (positive-only onset flex impulse, exponentially decaying); PC4 anti-sym whole-body yaw twist ← `_yaw_twist_accent` (pitch deviation + measure wave + phrase pulse); PC5 sym forward lean ← `_phrase_breath_accent` (cosine arc per musical phrase); PC6 anti-sym body sway ← `_density_sway_accent` (note density × ~48-beat slow sine); PC7 sym extension ← `_register_extension_accent` (`tanh` of pitch register). All event peaks are aligned to ≤40 ms of their triggers (was up to 320 ms with the old √-attack).
  3. **Beat-synced groove patterns** — direct hip/knee/waist additions in absolute radians (bounce, double-time, twist, sway, pump, accent snap), section-cycled across phrase boundaries; gated by an energy envelope masked by rhythm-suppression when the step layer is on (no-op when off).
  4. **Step layer (opt-in)** — gated behind `enable_steps` (default `False`). `_identify_step_events` triggers on strong accented low-note downbeats and emits per-foot `sin²` bell-curve step phases **centred on the trigger frame** (`[ti − N/2, ti + N/2)`, so the peak lift coincides with the beat instead of trailing it by 0.75 beats). `_apply_step_motion` adds asymmetric knee bend + hip flex on the lifting leg; `_rhythm_suppression_mask` widens the step window and zeros every PC activation + groove pattern during the step so the lift reads cleanly. When disabled, both step arrays stay zero, the rhythm mask is all-ones, and `_apply_step_motion` is skipped — the CSV omits the `left_foot_step` / `right_foot_step` columns and `simulate.py` keeps both feet anchored at every frame.
  Public knobs: `scale` (global), `pc_weights` (per-PC, length-`n_comp`, pad/truncate to 1.0), `enable_steps` (bool). `pc_weights[i]` multiplies into `amps[i]` so it scales *both* the carrier and the event accent for PC*i* uniformly. Smoothness layer: right before the matmul, each PC activation is gaussian-filtered with σ ≈ 1 frame (≈ 20 ms) — this softens the corners of `_squat_flex_accent` / `_stride_accent` / beat-rock gates without phase-shifting their peaks (kept inside the ≤ 40 ms event-peak budget).
- **`trajectory_generator.py`** — Orchestrates the pipeline. Adds arm joints at `NEUTRAL_STANCE` constant values. Clamps all joints to `JOINT_LIMITS` (from URDF). Two-stage smoothing per joint: gaussian (σ=3 samples ≈ 60 ms) followed by Savitzky–Golay (order 3, 9-frame window) — the gaussian kills remaining HF jitter, the savgol preserves the rounded peak shape of surviving accents so rhythm crispness is retained (measured: frame-jerk std drops 27–36% across slow/mid/uptempo MIDIs while per-joint range shrinks <2%). Forwards `scale`, `pc_weights`, `enable_steps` to `generate_pca_motion`. Exports `JOINT_NAMES` (27 joints) and `LOWER_BODY_JOINTS` (13 joints).
- **`trajectory_writer.py`** — Writes CSV with `timestamp` + 27 joint columns. `left_foot_step` / `right_foot_step` are only appended when present in the trajectory dict (i.e. when `enable_steps=True`); absence is graceful — `simulate.py` defaults to zero (both feet planted) via `joint_data.get(name, np.zeros(...))`.
- **`simulate.py`** — MuJoCo viewer with kinematic or dynamics mode. In kinematic mode: foot-flattening IK, ZMP/CoM balance, foot anchoring with Jacobian-based hip correction (skipped per-foot when that foot's step phase > 0.1). Synthesizes audio from MIDI via additive synthesis (plucked-string model). Builds scene XML on-the-fly from `casbot_band_urdf/xml/` and `casbot_band_urdf/meshes/`. **Audio/motion sync:** records `audio_start_time = time.time()` immediately before `subprocess.Popen(player, wav)`, runs a 50 ms verify-sleep (was 500 ms — which silently delayed motion by half a second), then sets `sim_start_time = audio_start_time + args.audio_offset`. `--audio-offset SEC` (default 0.0) compensates for player buffering: positive delays motion, negative advances it. **Post-IK smoothing (`_smooth_kinematic_display_sequence`, on by default; `--no-display-smoothing` to disable):** low-pass the floating-base XYZ + quaternion (component-wise + renormalise), `waist_yaw`, and now also `pelvic_pitch/roll/yaw` + `knee_pitch` with mild Gaussians (~30–60 ms) to remove Stage-B Jacobian residual that otherwise shows up as thigh/upper-body wobble. A final settle pass right after re-flattens ankles and re-anchors base Z to `FLOOR_Z` so the foot smoothing doesn't introduce sub-mm foot drift.
- **`motion_primitives.py`** — Deprecated hand-tuned primitives, kept for reference. Not used in the current pipeline.

### Joint mapping

- 13 lower-body (PCA-generated): 6 left leg + 6 right leg (pelvic_pitch/roll/yaw, knee_pitch, ankle_pitch/roll) + waist_yaw
- 14 arm (static): 7 left + 7 right (shoulder_pitch/roll/yaw, elbow_pitch, wrist_yaw/pitch/roll)
- All angles in radians. Frame rate: 50 Hz (dt=0.02).

### Data files

- `pca_model.npz` — precomputed PCA model (7 components; trained on mirrored + filtered frames, see `pca_extractor.py`)
- `csv/example1.csv`, `csv/example2.csv` — mocap source data
- `mid/yellow_bass.mid` — sample MIDI
- `casbot_band_urdf/` — robot URDF/MJCF definitions and STL meshes
