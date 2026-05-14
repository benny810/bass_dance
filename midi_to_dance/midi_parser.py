"""Parse MIDI files into structured note event data.

Key correctness improvements over a naive parser:
  * Full tempo map: every `set_tempo` event is recorded; `tick → seconds`
    is computed piecewise so mid-song tempo changes produce correct
    timing rather than silently shifting events.
  * Duration-weighted dominant tempo for `bpm` (not last-seen), which is
    what a downstream constant-bpm consumer expects.
  * Time-signature taken from the FIRST event (conductor-track-style),
    matching how almost all DAWs author meta tracks.
  * `NoteEvent.time_seconds` / `duration_seconds` resolved at parse-time
    via the tempo map so feature extraction never re-derives them under
    a constant-bpm assumption.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

import mido


# Default tempo when a MIDI omits set_tempo entirely (MIDI spec: 120 BPM).
_DEFAULT_USPB = 500_000   # microseconds per beat (= 120 BPM)


@dataclass
class NoteEvent:
    tick: int
    beat: float                # start tick / ticks_per_beat (musical beat index)
    measure: int
    note: int
    note_name: str
    velocity: int
    duration_beats: float
    time_seconds: float = 0.0      # tempo-map-resolved start time
    duration_seconds: float = 0.0  # tempo-map-resolved duration

    @staticmethod
    def _note_to_name(note: int) -> str:
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        return f"{names[note % 12]}{note // 12 - 1}"


@dataclass
class MidiData:
    bpm: float
    ticks_per_beat: int
    time_signature: Tuple[int, int]
    notes: List[NoteEvent] = field(default_factory=list)
    total_duration_beats: float = 0.0
    total_duration_seconds: float = 0.0
    # (tick, microsec_per_beat) sorted by tick, deduped (last wins on ties).
    # Always begins with `(0, uspb_at_t0)` so consumers can binary-search.
    tempo_map: List[Tuple[int, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Tempo-map utilities
# ---------------------------------------------------------------------------

def _build_tempo_map(events: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """Normalise a raw list of (tick, uspb) into a clean piecewise schedule.

    The output is sorted by tick, deduplicated (later events override
    earlier ones at the same tick), and guaranteed to start with an
    entry at tick=0 so binary-search lookups always have a left anchor.
    """
    events = sorted(events, key=lambda e: e[0])
    cleaned: List[Tuple[int, int]] = []
    seen_ticks: set = set()
    for tick, uspb in events:
        if tick in seen_ticks:
            cleaned[-1] = (tick, uspb)
        else:
            cleaned.append((tick, uspb))
            seen_ticks.add(tick)
    if not cleaned or cleaned[0][0] != 0:
        cleaned.insert(0, (0, cleaned[0][1] if cleaned else _DEFAULT_USPB))
    return cleaned


def _tick_to_seconds(
    tick: int, tempo_map: List[Tuple[int, int]], ticks_per_beat: int,
) -> float:
    """Convert an absolute tick to seconds under a piecewise tempo schedule.

    Walks each tempo segment up to `tick` accumulating
    `(segment_ticks × uspb / ticks_per_beat / 1e6)`.  O(len(tempo_map))
    per call; small in practice (typically 1–3 tempos).
    """
    if tick <= 0:
        return 0.0
    seconds = 0.0
    for i, (seg_tick, uspb) in enumerate(tempo_map):
        seg_end = tempo_map[i + 1][0] if i + 1 < len(tempo_map) else tick + 1
        if tick <= seg_tick:
            break
        span_end = min(tick, seg_end)
        seconds += (span_end - seg_tick) * uspb / (ticks_per_beat * 1_000_000.0)
        if tick <= seg_end:
            break
    return seconds


def _dominant_bpm(
    tempo_map: List[Tuple[int, int]],
    end_tick: int,
    ticks_per_beat: int,
) -> float:
    """Pick the BPM that covers the largest number of ticks.

    Falls back to the first tempo if there's only one segment.  Using
    duration weighting (rather than last-seen) keeps `bpm` semantically
    "the song's tempo" even when a final ritardando appears.
    """
    if not tempo_map:
        return 60_000_000 / _DEFAULT_USPB
    if len(tempo_map) == 1 or end_tick <= 0:
        return 60_000_000 / tempo_map[0][1]
    durations = {}
    for i, (seg_tick, uspb) in enumerate(tempo_map):
        seg_end = tempo_map[i + 1][0] if i + 1 < len(tempo_map) else end_tick
        span = max(0, seg_end - seg_tick)
        durations[uspb] = durations.get(uspb, 0) + span
    best = max(durations.items(), key=lambda kv: kv[1])
    return 60_000_000 / best[0]


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_midi(filepath: str) -> MidiData:
    """Parse a MIDI file into a `MidiData` with tempo-map-resolved timing.

    Notes are sorted by start tick (then by pitch as a stable tiebreaker
    so chord notes appear in a deterministic order).  Channel-10 drum
    notes are kept (they may still drive rhythm features); callers can
    filter if desired.
    """
    mid = mido.MidiFile(filepath)
    ticks_per_beat = max(int(mid.ticks_per_beat), 1)

    tempo_events_raw: List[Tuple[int, int]] = []
    time_sig: Tuple[int, int] = (4, 4)
    time_sig_set = False

    # Per-track absolute-tick scan: collect tempo, time-sig, and note events.
    raw_note_events = []  # (abs_tick, type, note, velocity)
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_events_raw.append((abs_tick, msg.tempo))
            elif msg.type == "time_signature":
                if not time_sig_set:
                    time_sig = (msg.numerator, msg.denominator)
                    time_sig_set = True
            elif msg.type == "note_on" and msg.velocity > 0:
                raw_note_events.append((abs_tick, "on", msg.note, msg.velocity))
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                raw_note_events.append((abs_tick, "off", msg.note, 0))

    tempo_map = _build_tempo_map(tempo_events_raw)
    end_tick_guess = max((t for t, *_ in raw_note_events), default=0)

    # Match note_on → note_off pairs.  Allow stacked notes of the same
    # pitch by using a per-pitch FIFO (rare in bass tracks but possible
    # when a sustained note is re-attacked before the prior note_off).
    active: dict = {}  # note_num -> list[(start_tick, velocity)] FIFO
    notes: List[NoteEvent] = []
    raw_note_events.sort(key=lambda e: (e[0], 0 if e[1] == "off" else 1))
    for tick, etype, note_num, velocity in raw_note_events:
        if etype == "on":
            active.setdefault(note_num, []).append((tick, velocity))
        else:
            stack = active.get(note_num)
            if stack:
                start_tick, vel = stack.pop(0)
                duration_ticks = tick - start_tick
                if duration_ticks > 0:
                    beat = start_tick / ticks_per_beat
                    measure = int(beat // max(time_sig[0], 1))
                    start_s = _tick_to_seconds(start_tick, tempo_map, ticks_per_beat)
                    end_s = _tick_to_seconds(tick, tempo_map, ticks_per_beat)
                    notes.append(NoteEvent(
                        tick=start_tick,
                        beat=beat,
                        measure=measure,
                        note=note_num,
                        note_name=NoteEvent._note_to_name(note_num),
                        velocity=vel,
                        duration_beats=duration_ticks / ticks_per_beat,
                        time_seconds=start_s,
                        duration_seconds=end_s - start_s,
                    ))

    # Stable sort: time first, then pitch, so chord notes are ordered.
    notes.sort(key=lambda n: (n.time_seconds, n.note))

    if notes:
        last = notes[-1]
        total_beats = last.beat + last.duration_beats
        total_seconds = last.time_seconds + last.duration_seconds
    else:
        total_beats = 0.0
        total_seconds = 0.0

    bpm = _dominant_bpm(tempo_map, end_tick_guess, ticks_per_beat)

    return MidiData(
        bpm=bpm,
        ticks_per_beat=ticks_per_beat,
        time_signature=time_sig,
        notes=notes,
        total_duration_beats=total_beats,
        total_duration_seconds=total_seconds,
        tempo_map=tempo_map,
    )
