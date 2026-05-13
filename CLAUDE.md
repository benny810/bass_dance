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

# Generate dance trajectory from MIDI
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv
python -m midi_to_dance.main mid/yellow_bass.mid -o csv/output.csv --scale 1.2 --plot --stats

# MuJoCo simulation (kinematic mode, audio synced)
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid

# Dynamics mode (PD actuators + physics)
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --dynamics

# Half-speed, no audio, or save audio to WAV
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --slow 0.5 --no-audio
python midi_to_dance/simulate.py csv/output.csv mid/yellow_bass.mid --save-audio output.wav
```

## Architecture

Data flows through a sequential pipeline (`main.py` is the entry point):

```
MIDI file  →  midi_parser.py (MidiData)  →  feature_extractor.py (MusicalFeatures)
  →  pca_motion.py (PCA joint angles)  →  trajectory_generator.py (clamp + smooth)
  →  trajectory_writer.py (CSV)  →  simulate.py (MuJoCo viewer + audio)
```

### Key modules

- **`midi_parser.py`** — Parses MIDI via `mido`, returns `MidiData` (bpm, time_sig, list of `NoteEvent` with beat-aligned timing). Matches note_on/note_off pairs across all tracks.
- **`feature_extractor.py`** — Converts `MidiData` → `MusicalFeatures` at a given `dt` (default 0.02s). Per-sample arrays: onset_strength, beat_phase, is_downbeat, pitch_level, accent, phrase_boundaries, etc.
- **`pca_extractor.py`** — Offline primitive extraction from `example*.csv`. Pipeline (defaults): load 13 lower-body joints → Butterworth high-pass detrend (0.3 Hz, zero-phase, mean reattached) → drop lowest-speed frames (10th percentile on high-pass velocity) → left-right mirror augmentation (swap legs + flip lateral joint signs) → mean-center + SVD → sign-canonical PCs by variance order; optional `--canonical` reorders PCs to semantic prototypes via Hungarian `|cosine|` matching for tighter alignment with hard-coded PC-slot accents in `pca_motion.py`. Saves `pca_model.npz` (`mean_pose`, `components`, `std_scores`, `explained_variance_ratio`, `joint_names`, counts).
- **`pca_motion.py`** — The core motion algorithm. Reconstructs joint angles as `mean_pose + Σ(activation_i × component_i)`. Three-layer architecture:
  1. **Continuous carriers** — 7×3=21 incommensurate sine waves (irrational frequency ratios), never-repeating baseline
  2. **Musical energy envelope** — onset density + accent strength → gaussian-smoothed amplitude modulator (0.05 baseline, ~11:1 dynamic range)
  3. **Event accents** — per-PC musical events (~30% of energy): onset knee flex (PC1), pitch modulation (PC2), beat rocking (PC3), accented stride (PC4), density twist (PC5), register shift (PC6), phrase breathing (PC7)
- **`trajectory_generator.py`** — Orchestrates the pipeline. Adds arm joints at `NEUTRAL_STANCE` constant values. Clamps all joints to `JOINT_LIMITS` (from URDF). Applies gaussian smoothing (σ=2 samples). Exports `JOINT_NAMES` (27 joints) and `LOWER_BODY_JOINTS` (13 joints).
- **`trajectory_writer.py`** — Writes CSV with `timestamp` + 27 joint columns + optional `left_foot_step`/`right_foot_step` columns.
- **`simulate.py`** — MuJoCo viewer with kinematic or dynamics mode. In kinematic mode: foot-flattening IK, ZMP/CoM balance, foot anchoring with Jacobian-based hip correction. Synthesizes audio from MIDI via additive synthesis (plucked-string model). Builds scene XML on-the-fly from `casbot_band_urdf/xml/` and `casbot_band_urdf/meshes/`.
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
