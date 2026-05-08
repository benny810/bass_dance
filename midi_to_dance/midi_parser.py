"""Parse MIDI files into structured note event data."""

from dataclasses import dataclass, field
from typing import List, Tuple
import mido


@dataclass
class NoteEvent:
    tick: int
    beat: float
    measure: int
    note: int
    note_name: str
    velocity: int
    duration_beats: float

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


def parse_midi(filepath: str) -> MidiData:
    """Parse a MIDI file and return structured MidiData."""
    mid = mido.MidiFile(filepath)
    ticks_per_beat = mid.ticks_per_beat

    bpm = 120.0
    time_sig = (4, 4)

    abs_tick = 0
    tempo_at_tick = [(0, 500000)]  # (tick, microseconds per beat)

    raw_events = []  # (tick, type, note, velocity)
    for msg in mid.tracks[0]:
        abs_tick += msg.time
        if msg.type == "set_tempo":
            tempo_at_tick.append((abs_tick, msg.tempo))
        elif msg.type == "time_signature":
            time_sig = (msg.numerator, msg.denominator)
        elif msg.type == "note_on" and msg.velocity > 0:
            raw_events.append((abs_tick, "on", msg.note, msg.velocity))
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            raw_events.append((abs_tick, "off", msg.note, 0))

    # Use the most common tempo
    if len(tempo_at_tick) > 1:
        dominant = tempo_at_tick[-1][1]
        bpm = 60_000_000 / dominant
    else:
        bpm = 120.0

    # Match note_on to note_off to compute durations
    active = {}  # note -> (tick, velocity)
    notes = []
    for tick, etype, note_num, velocity in raw_events:
        if etype == "on":
            active[note_num] = (tick, velocity)
        else:  # off
            if note_num in active:
                start_tick, vel = active.pop(note_num)
                duration_ticks = tick - start_tick
                if duration_ticks > 0:
                    beat = start_tick / ticks_per_beat
                    measure = int(beat // time_sig[0])
                    note_name = NoteEvent._note_to_name(note_num)
                    notes.append(NoteEvent(
                        tick=start_tick,
                        beat=beat,
                        measure=measure,
                        note=note_num,
                        note_name=note_name,
                        velocity=vel,
                        duration_beats=duration_ticks / ticks_per_beat,
                    ))

    notes.sort(key=lambda n: n.beat)

    total_beats = notes[-1].beat + notes[-1].duration_beats if notes else 0.0
    total_seconds = total_beats * 60.0 / bpm

    return MidiData(
        bpm=bpm,
        ticks_per_beat=ticks_per_beat,
        time_signature=time_sig,
        notes=notes,
        total_duration_beats=total_beats,
        total_duration_seconds=total_seconds,
    )
