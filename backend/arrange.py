"""Piano arrangement engine: detected chords + sections -> piano MIDI.

The harmony comes entirely from the chord list (detected or user-corrected).
This module only decides *how* a piano would play those chords.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pretty_midi

from analysis import QUALITIES, chord_label, FLAT_KEYS_MAJOR

STYLES = {
    "ambient": "Ambient Worship Piano",
    "soft": "Soft Piano",
    "cinematic": "Cinematic Piano",
    "simple": "Simple Piano",
}

TAIL_RING = 3.0  # seconds the final chord keeps ringing


@dataclass
class ArrangeParams:
    style: str = "ambient"
    transpose: int = 0          # semitones
    target_bpm: float | None = None
    sustain: float = 0.8        # 0..1
    seed: int = 7


def transpose_chord(root: int, semis: int) -> int:
    return (root + semis) % 12


def uses_flats(key_tonic: int, mode: str, semis: int) -> bool:
    tonic = (key_tonic + semis) % 12
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    return major_tonic in FLAT_KEYS_MAJOR


def chord_pcs(root: int, quality: str) -> list[int]:
    return [(root + iv) % 12 for iv in QUALITIES[quality][1]]


def _place(pcs: list[int], low: int, order_start: int) -> list[int]:
    """Stack pitch classes upward starting at the first note >= low, rotated by order_start."""
    rot = pcs[order_start:] + pcs[:order_start]
    notes = []
    cur = low - 1
    for pc in rot:
        n = cur + 1
        while n % 12 != pc:
            n += 1
        notes.append(n)
        cur = n
    return notes


def rh_voicing(pcs: list[int], prev: list[int] | None, lo: int = 57, hi: int = 77, max_notes: int = 4) -> list[int]:
    """Pick the inversion (and octave) that moves least from the previous voicing."""
    pcs = pcs[:max_notes]
    best, best_cost = None, 1e9
    for inv in range(len(pcs)):
        for base in range(lo, lo + 12):
            v = _place(pcs, base, inv)
            if v[-1] > hi or v[0] < lo:
                continue
            if prev:
                cost = abs(sum(v) / len(v) - sum(prev) / len(prev)) + 0.3 * sum(abs(a - b) for a, b in zip(sorted(v), sorted(prev)))
            else:
                cost = abs(sum(v) / len(v) - 66)
            if cost < best_cost:
                best, best_cost = v, cost
    return best or _place(pcs, 60, 0)


def bass_note(root: int) -> int:
    return 36 + ((root - 36) % 12)  # C2..B2


def _section_at(sections: list[dict], t: float) -> tuple[dict | None, int]:
    for i, s in enumerate(sections):
        if s["start"] <= t < s["end"]:
            return s, i
    return (sections[-1], len(sections) - 1) if sections else (None, -1)


def _section_gain(sec: dict | None, t: float) -> float:
    if not sec:
        return 1.0
    kind = sec.get("kind", "verse")
    g = {"intro": 0.82, "verse": 0.95, "chorus": 1.10, "bridge": 1.02, "outro": 0.85, "instrumental": 1.0}.get(kind, 1.0)
    if kind == "outro":  # gentle fade across the outro
        span = max(sec["end"] - sec["start"], 1e-6)
        g *= 1.0 - 0.35 * (t - sec["start"]) / span
    return g


def build_midi(chords: list[dict], sections: list[dict], orig_bpm: float, beats_per_bar: int,
               first_beat: float, params: ArrangeParams) -> tuple[pretty_midi.PrettyMIDI, dict]:
    if not chords:
        raise ValueError("No chords to arrange.")
    if params.style not in STYLES:
        raise ValueError(f"Unknown style '{params.style}'.")

    target = params.target_bpm or orig_bpm
    k = orig_bpm / target                      # time-stretch factor
    beat = 60.0 / target
    bar = beat * beats_per_bar
    rng = random.Random(params.seed)
    style = params.style
    sustain = max(0.0, min(1.0, params.sustain))

    # scaled copies (times in seconds of the NEW tempo)
    segs = [{**c, "t0": c["start"] * k, "t1": c["end"] * k} for c in chords]
    secs = [{**s, "start": s["start"] * k, "end": s["end"] * k} for s in sections]
    fb = first_beat * k

    pm = pretty_midi.PrettyMIDI(initial_tempo=target)
    piano = pretty_midi.Instrument(program=0, name="Piano")

    def add(pitch, t0, t1, vel):
        pitch = int(max(21, min(108, pitch)))
        vel = int(max(18, min(112, vel + rng.randint(-3, 3))))
        if t1 - t0 < 0.05:
            t1 = t0 + 0.05
        piano.notes.append(pretty_midi.Note(velocity=vel, pitch=pitch, start=max(0.0, t0), end=t1))

    prev_rh: list[int] | None = None
    n_segs = len(segs)
    arps_added = 0

    for si, seg in enumerate(segs):
        root = transpose_chord(seg["root"], params.transpose)
        q = seg["quality"]
        pcs = chord_pcs(root, q)
        t0, t1 = seg["t0"], seg["t1"]
        last = si == n_segs - 1
        if last:
            t1 += TAIL_RING
        length = t1 - t0
        sec, sec_i = _section_at(secs, t0 + 1e-3)
        energy = float(seg.get("energy", 0.5))
        gain = _section_gain(sec, t0)
        rel = 0.55 + 0.45 * sustain           # fraction of the segment that notes ring

        # shape of this style
        if style == "simple":
            max_notes, base_v, span_v = 3, 66, 22
        elif style == "soft":
            max_notes, base_v, span_v = 3, 40, 18
        elif style == "cinematic":
            max_notes, base_v, span_v = 4, 52, 40
        else:
            max_notes, base_v, span_v = 4, 40, 26
        vel = (base_v + energy * span_v) * gain

        rh = rh_voicing(pcs, prev_rh, max_notes=max_notes)
        prev_rh = rh
        b = bass_note(root)

        # strike points inside this chord: the change itself, then each bar line (not for ambient/soft holds)
        strikes = [t0]
        if style in ("simple", "cinematic", "soft"):
            t = t0 + bar          # chord changes sit on bar lines, so count bars from the change
            while t < t1 - beat * 0.5:
                strikes.append(t)
                t += bar
        if style == "ambient" and length > bar * 4:
            # very long chord: re-touch softly every 4 bars so it does not die away
            t = t0 + bar * 4
            while t < t1 - bar:
                strikes.append(t)
                t += bar * 4

        for ti, ts in enumerate(strikes):
            te_full = (strikes[ti + 1] if ti + 1 < len(strikes) else t1)
            dur = max(0.2, (te_full - ts) * rel + (0.5 if te_full == t1 else 0.0))
            if last and ti == len(strikes) - 1:
                dur = (te_full - ts) * 0.98
            first = ti == 0
            v = vel * (1.0 if first else 0.82)

            stack: dict[int, tuple[float, float]] = {}   # pitch -> (time offset, velocity factor)

            def put(pitch: int, off: float, vf: float) -> None:
                stack.setdefault(int(pitch), (off, vf))      # one note per pitch, first wins

            fifth = b + 7 if b + 7 < 55 else b + 12
            if style == "ambient":
                put(b, 0.0, 0.9)
                put(fifth, 0.02, 0.62)
                for j, n_ in enumerate(rh):                  # slow upward roll, soft attack
                    put(n_, 0.05 + 0.045 * j, 0.8 + 0.05 * j)
            elif style == "soft":
                put(b, 0.0, 0.85)
                for j, n_ in enumerate(rh):
                    put(n_, 0.03 * j, 0.9)
            elif style == "cinematic":
                put(b, 0.0, 1.0)
                put(b + 12, 0.0, 0.8)
                put(fifth, 0.02, 0.6)
                wide = [n_ - 12 if j == 0 else n_ for j, n_ in enumerate(rh)]
                for j, n_ in enumerate(wide):
                    put(n_, 0.04 + 0.03 * j, 0.8)
            else:  # simple
                put(b, 0.0, 1.0)
                for n_ in rh:
                    put(n_, 0.0, 0.85)
            for pitch, (off, vf) in stack.items():
                add(pitch, ts + off, ts + dur, v * vf)
            held_now = set(stack)

        # flowing top-line figures
        if style in ("ambient", "cinematic") and length > beat * 3:
            step = beat * (2 if style == "ambient" else 1)
            if style == "ambient" and (sec and sec.get("kind") in ("intro", "outro")):
                step = beat * 4
            tones = [n_ + 12 for n_ in rh] + [rh[0] + 24 if rh[0] + 24 <= 96 else rh[0] + 12]
            if style == "cinematic":
                tones = tones + tones[-2:0:-1]
            held = set(held_now)
            idx = 0
            t = t0 + step
            while t < t1 - step * 0.5 and (not last or t < t1 - TAIL_RING):
                n_ = tones[idx % len(tones)]
                while n_ in held and n_ < 100:   # never re-strike a note that is still held
                    n_ += 12
                av = vel * (0.45 if style == "ambient" else 0.55 + 0.25 * energy)
                add(n_, t, min(t + beat * (3 if style == "ambient" else 1.2), t1), av)
                arps_added += 1
                idx += 1
                t += step

        # section lift: short rising run into a louder section
        if style in ("ambient", "cinematic") and not last:
            nxt = segs[si + 1]
            nsec, nsec_i = _section_at(secs, nxt["t0"] + 1e-3)
            if nsec_i != sec_i and sec and nsec and nsec.get("energy", 0) > sec.get("energy", 0) + 0.05 and length > beat * 3:
                run_t = t1 - beat * 1.5
                for j, n_ in enumerate(rh[:3]):
                    add(n_ + 12, run_t + j * beat * 0.4, t1 + 0.2, vel * 0.55)

        # sustain pedal: down with each chord, brief lift just before the next chord
        pedal_on = sustain > 0.12
        if pedal_on:
            piano.control_changes.append(pretty_midi.ControlChange(64, 127, max(0.0, t0)))
            lift = min(t1 - 0.02, t1 - (0.05 + (1 - sustain) * 0.2)) if not last else t1
            if si + 1 < n_segs or sustain < 0.95:
                piano.control_changes.append(pretty_midi.ControlChange(64, 0, max(t0 + 0.05, lift)))
            if last:
                piano.control_changes.append(pretty_midi.ControlChange(64, 0, t1))

        prev_rh = rh

    # A note-off for pitch P would also silence a newer note on the same pitch P, so
    # trim any note that is still ringing when the same pitch is struck again.
    by_pitch: dict[int, list] = {}
    for n in piano.notes:
        by_pitch.setdefault(n.pitch, []).append(n)
    for notes in by_pitch.values():
        notes.sort(key=lambda n: n.start)
        for prev, nxt in zip(notes, notes[1:]):
            if prev.end > nxt.start - 0.01:
                prev.end = max(prev.start + 0.03, nxt.start - 0.01)
    piano.notes.sort(key=lambda n: (n.start, n.pitch))

    pm.instruments.append(piano)
    end_time = max(n.end for n in piano.notes)
    info = {
        "notes": len(piano.notes),
        "duration": round(end_time, 2),
        "tempo": round(target, 1),
        "time_factor": k,
        "sections": [{"label": s["label"], "kind": s.get("kind"), "start": round(s["start"], 2), "end": round(s["end"], 2)} for s in secs],
    }
    return pm, info


def transposed_chord_names(chords: list[dict], key_tonic: int, mode: str, semis: int) -> list[str]:
    flats = uses_flats(key_tonic, mode, semis)
    return [chord_label(transpose_chord(c["root"], semis), c["quality"], flats) for c in chords]
