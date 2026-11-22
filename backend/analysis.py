"""Song analysis: key, BPM, time signature, chords, sections, dynamics.

Everything here is derived from the audio signal. Nothing is guessed from a
song title. Each value is tagged "detected" (measured directly) or
"estimated" (a heuristic that can be wrong).
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import librosa
import numpy as np

SR = 22050
HOP = 512
MAX_SECONDS = 15 * 60

NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
FLAT_KEYS_MAJOR = {5, 10, 3, 8, 1, 6}  # F Bb Eb Ab Db Gb

# quality -> (suffix, intervals, template weight bonus)
QUALITIES = {
    "maj": ("", [0, 4, 7], 1.00),
    "min": ("m", [0, 3, 7], 1.00),
    "7": ("7", [0, 4, 7, 10], 0.93),
    "maj7": ("maj7", [0, 4, 7, 11], 0.93),
    "min7": ("m7", [0, 3, 7, 10], 0.93),
    "sus4": ("sus4", [0, 5, 7], 0.92),
    "sus2": ("sus2", [0, 2, 7], 0.92),
}
SUFFIX_TO_Q = {v[0]: k for k, v in QUALITIES.items()}

# Krumhansl-Kessler key profiles
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def note_name(pc: int, flats: bool) -> str:
    return (NOTE_FLAT if flats else NOTE_SHARP)[pc % 12]


def chord_label(root: int, quality: str, flats: bool) -> str:
    return note_name(root, flats) + QUALITIES[quality][0]


def parse_chord(label: str) -> tuple[int, str]:
    """'F#m7' -> (6, 'min7'). Raises ValueError for anything unrecognised."""
    s = label.strip()
    if not s:
        raise ValueError("empty chord")
    root_txt = s[0].upper()
    rest = s[1:]
    if rest[:1] in ("#", "b"):
        root_txt += rest[0]
        rest = rest[1:]
    table = {n: i for i, n in enumerate(NOTE_SHARP)}
    table.update({n: i for i, n in enumerate(NOTE_FLAT)})
    if root_txt not in table:
        raise ValueError(f"unknown root in '{label}'")
    if rest not in SUFFIX_TO_Q:
        raise ValueError(f"unsupported chord type '{rest}' in '{label}'")
    return table[root_txt], SUFFIX_TO_Q[rest]


def _templates():
    names, mats = [], []
    for root in range(12):
        for q, (_, ivs, w) in QUALITIES.items():
            v = np.zeros(12)
            for iv in ivs:
                v[(root + iv) % 12] = 1.0
            # root slightly emphasised: helps separate relative major/minor
            v[root] += 0.25
            v /= np.linalg.norm(v)
            names.append((root, q, w))
            mats.append(v)
    return names, np.array(mats)


TEMPL_NAMES, TEMPL = _templates()


def decode_to_wav(src: Path, dst: Path) -> None:
    """Decode any audio/video container to mono 22.05 kHz wav with ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-v", "error", "-i", str(src), "-vn", "-ac", "1",
        "-ar", str(SR), "-t", str(MAX_SECONDS), str(dst),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if r.returncode != 0 or not dst.exists():
        raise ValueError("Could not read that file as audio. " + r.stderr.strip()[-200:])


def detect_key(chroma: np.ndarray) -> dict:
    prof = chroma.mean(axis=1)
    prof = prof / (np.linalg.norm(prof) + 1e-9)
    scores = []
    for tonic in range(12):
        for mode, kk in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            r = np.corrcoef(prof, np.roll(kk, tonic))[0, 1]
            scores.append((r, tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    # confidence = how clearly the best key beats the runner-up
    conf = float(np.clip((best[0] - second[0]) * 6 + 0.35, 0, 1))
    tonic, mode = best[1], best[2]
    major_tonic = tonic if mode == "major" else (tonic + 3) % 12
    flats = major_tonic in FLAT_KEYS_MAJOR
    return {
        "tonic": int(tonic), "mode": mode, "flats": bool(flats),
        "name": f"{note_name(tonic, flats)} {mode}",
        "confidence": round(conf, 2),
        "alternate": f"{note_name(second[1], flats)} {second[2]}",
    }


def detect_time_signature(onset_env: np.ndarray, beats: np.ndarray) -> dict:
    """Heuristic: compare accent regularity at groups of 3 vs 4 beats."""
    if len(beats) < 16:
        return {"value": "4/4", "confidence": 0.3}
    strength = onset_env[np.clip(beats, 0, len(onset_env) - 1)]
    strength = (strength - strength.mean()) / (strength.std() + 1e-9)
    scores = {}
    for n in (3, 4):
        best = 0.0
        for phase in range(n):
            pos = strength[phase::n]
            rest = np.delete(strength, np.arange(phase, len(strength), n))
            best = max(best, pos.mean() - rest.mean())
        scores[n] = best
    # 4/4 is by far the most common, so 3 must clearly win
    if scores[3] > scores[4] * 1.35 and scores[3] > 0.35:
        return {"value": "3/4", "confidence": round(float(min(0.8, scores[3] / 2)), 2)}
    return {"value": "4/4", "confidence": round(float(min(0.9, 0.4 + scores[4] / 3)), 2)}


def _diatonic_triads(tonic: int, mode: str) -> set[tuple[int, str]]:
    """(root, 'maj'|'min') triads that belong to the key (harmonic minor V included)."""
    if mode == "major":
        degs = [(0, "maj"), (2, "min"), (4, "min"), (5, "maj"), (7, "maj"), (9, "min")]
    else:
        degs = [(0, "min"), (3, "maj"), (5, "min"), (7, "min"), (7, "maj"), (8, "maj"), (10, "maj")]
    return {((tonic + d) % 12, q) for d, q in degs}


def _triad_family(q: str) -> str:
    return "min" if q in ("min", "min7") else "maj"


def _viterbi_chords(beat_chroma: np.ndarray, key: dict | None = None, stay: float = 0.82) -> tuple[np.ndarray, np.ndarray]:
    """Return (state index per beat, confidence per beat).

    Bass notes carry loud overtones (the 5th harmonic is a major third), which
    makes minor chords look major. A mild preference for chords that belong to
    the detected key corrects most of those without ruling out borrowed chords.
    """
    sim = TEMPL @ beat_chroma  # (n_states, n_beats)
    bonus = np.array([w for _, _, w in TEMPL_NAMES])
    if key is not None:
        dia = _diatonic_triads(key["tonic"], key["mode"])
        bonus = bonus * np.array([1.0 if (r, _triad_family(q)) in dia else 0.86 for r, q, _ in TEMPL_NAMES])
    sim = sim * bonus[:, None]
    prob = np.exp(sim * 9.0)
    prob /= prob.sum(axis=0, keepdims=True)
    n = len(TEMPL_NAMES)
    trans = librosa.sequence.transition_loop(n, stay)
    states = librosa.sequence.viterbi_discriminative(prob, trans)
    cols = np.arange(prob.shape[1])
    chosen = prob[states, cols]
    masked = prob.copy()
    masked[states, cols] = 0
    runner_up = masked.max(axis=0)
    conf = chosen / (chosen + runner_up + 1e-12)   # 0.5 = coin flip, ->1 = clear winner
    return states, conf


def fix_beat_levels(bt: np.ndarray) -> np.ndarray:
    """Repair half/double-speed slips inside a tracked beat grid.

    When a song slows down a lot the tracker can start ticking every half beat, or skip a beat.
    A sustained run of beats at about half the recent gap is merged back in pairs, and a single
    gap of about double the recent gap gets a midpoint beat. Steady songs pass through unchanged.
    """
    bt = np.asarray(bt, dtype=float)
    G = np.diff(bt)
    if len(G) < 8:
        return bt
    out = [bt[0]]
    hist: list[float] = []
    j = 0
    n = len(G)
    merging = False
    while j < n:
        ref = float(np.median(hist[-3:])) if len(hist) >= 3 else float(np.median(G[: min(8, n)]))
        g = G[j]
        need = 2 if merging else 4
        half_run = j + need <= n and all(0.38 < G[j + k] / ref < 0.64 for k in range(need))
        if half_run and j + 1 < n:
            gap = G[j] + G[j + 1]
            out.append(out[-1] + gap); hist.append(gap); j += 2; merging = True
            continue
        merging = False
        if len(hist) >= 3 and 1.75 < g / ref < 2.25:       # one missed beat
            half = g / 2
            out += [out[-1] + half, out[-1] + g]; hist += [half, half]
        else:
            out.append(out[-1] + g); hist.append(g)
        j += 1
    return np.array(out)


def tempo_map(beat_times: np.ndarray) -> dict:
    """Describe how the speed changes through the song, from the actual beat positions.

    Returns the average BPM, a list of steady stretches ("segments") and whether the tempo
    varies enough to matter (more than about 8% between slowest and fastest stretch).
    """
    from scipy.ndimage import median_filter

    gaps = np.diff(beat_times)
    gaps = np.clip(gaps, 60 / 260, 60 / 35)
    local = 60.0 / gaps                                  # BPM between each pair of beats
    smooth = median_filter(local, size=9, mode="nearest")
    n = len(smooth)
    average = 60.0 * (len(beat_times) - 1) / max(1e-6, beat_times[-1] - beat_times[0])

    # Split into stretches: a stretch ends when the (smoothed) tempo drifts more than 5% from where it began.
    bounds = [0]
    ref = float(np.median(smooth[: min(4, n)]))
    for i in range(1, n):
        if abs(smooth[i] - ref) / ref > 0.05:
            bounds.append(i)
            ref = float(np.median(smooth[i: i + 4]))
    bounds.append(n)
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 6 and segs:                            # too short to be real: absorb into the previous
            segs[-1]["b"] = b
            continue
        segs.append({"a": a, "b": b})
    out = []
    for sg in segs:
        a, b = sg["a"], sg["b"]
        out.append({
            "start": round(float(beat_times[a]), 2),
            "end": round(float(beat_times[min(b, len(beat_times) - 1)]), 2),
            "bpm": round(float(np.median(local[a:b])), 1),
        })
    out[0]["start"] = 0.0
    bpms = [s["bpm"] for s in out]
    varies = max(bpms) / min(bpms) > 1.08
    # confidence: how steady beats are around their own local tempo (not around one global tempo)
    resid = np.abs(local - smooth) / smooth
    conf = float(np.clip(1.0 - np.median(resid) * 6 - float(np.percentile(resid, 90)) * 1.5, 0.2, 0.95))
    return {"average_bpm": round(average, 1), "segments": out, "varies": bool(varies), "confidence": round(conf, 2),
            "min_bpm": round(min(bpms), 1), "max_bpm": round(max(bpms), 1)}


def analyse(audio_path: Path, work_dir: Path) -> dict:
    wav = work_dir / "source.wav"
    decode_to_wav(audio_path, wav)
    y, sr = librosa.load(wav, sr=SR, mono=True)
    duration = float(len(y) / sr)
    if duration < 15:
        raise ValueError("That recording is too short to analyse (need at least 15 seconds).")
    if np.max(np.abs(y)) < 1e-4:
        raise ValueError("That recording looks silent.")

    y_h, y_p = librosa.effects.hpss(y)

    # --- tempo and beats (percussive part is steadier) ---
    onset_env = librosa.onset.onset_strength(y=y_p + 0.3 * y_h, sr=sr, hop_length=HOP)
    # Start the beat search from a tempo measured in THIS recording. The library default (120 BPM)
    # often locks onto hi-hats at double speed. A loose "tightness" lets the beat grid follow a
    # band that pushes and pulls instead of forcing one steady tempo across the whole song.
    start_bpm = float(np.atleast_1d(librosa.feature.tempo(onset_envelope=onset_env, sr=sr, hop_length=HOP, start_bpm=95))[0])
    _, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=HOP, start_bpm=start_bpm, tightness=25)
    beat_frames = np.asarray(beat_frames)
    if len(beat_frames) < 8:
        raise ValueError("Could not find a steady beat in that recording.")
    beat_times = fix_beat_levels(librosa.frames_to_time(beat_frames, sr=sr, hop_length=HOP))
    beat_frames = np.round(beat_times * sr / HOP).astype(int)
    tmap = tempo_map(beat_times)
    bpm = tmap["average_bpm"]
    bpm_conf = tmap["confidence"]

    # --- chroma ---
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, hop_length=HOP, n_chroma=12, bins_per_octave=36)
    key = detect_key(chroma)
    flats = key["flats"]

    time_sig = detect_time_signature(onset_env, beat_frames)
    beats_per_bar = 3 if time_sig["value"] == "3/4" else 4

    # --- beat-synchronous chords ---
    # prepend frame 0 / append last frame so the whole song is covered
    # librosa.util.sync pads the end of the signal itself, so n_edges columns come back
    rms = librosa.feature.rms(y=y, hop_length=HOP)[0]
    n_frames = min(chroma.shape[1], rms.shape[0])
    chroma, rms = chroma[:, :n_frames], rms[:n_frames]
    edges = np.unique(np.concatenate([[0], beat_frames]))
    edges = edges[edges < n_frames]
    # Cut chords on a half-beat grid: a chord change still lands on a grid line when the beat
    # tracker is half a beat out of phase, or tracking at half speed.
    mids = (edges[:-1] + edges[1:]) // 2
    sub = np.unique(np.concatenate([edges, mids]))
    sub = sub[sub < n_frames]
    sub_chroma = librosa.util.sync(chroma, sub, aggregate=np.median)
    sub_chroma = sub_chroma / (np.linalg.norm(sub_chroma, axis=0, keepdims=True) + 1e-9)
    states, conf = _viterbi_chords(sub_chroma, key, stay=0.9)
    sub_times = librosa.frames_to_time(np.append(sub, n_frames), sr=sr, hop_length=HOP)
    sub_times[-1] = duration
    edge_times = librosa.frames_to_time(np.append(edges, n_frames), sr=sr, hop_length=HOP)
    edge_times[-1] = duration

    # --- loudness (dynamics): per beat for sections, per half-beat for chords ---
    beat_rms = librosa.util.sync(rms[None, :], edges, aggregate=np.mean)[0]
    lo, hi = np.percentile(beat_rms, 5), np.percentile(beat_rms, 95)
    energy = np.clip((beat_rms - lo) / (hi - lo + 1e-9), 0, 1)
    sub_rms = librosa.util.sync(rms[None, :], sub, aggregate=np.mean)[0]
    sub_energy = np.clip((sub_rms - lo) / (hi - lo + 1e-9), 0, 1)

    chords = []
    i = 0
    n_sub = len(states)
    while i < n_sub:
        j = i
        while j + 1 < n_sub and states[j + 1] == states[i]:
            j += 1
        root, q, _ = TEMPL_NAMES[states[i]]
        chords.append({
            "start": round(float(sub_times[i]), 3),
            "end": round(float(sub_times[j + 1]), 3),
            "chord": chord_label(root, q, flats),
            "root": int(root), "quality": q,
            "confidence": round(float(np.mean(conf[i:j + 1])), 2),
            "energy": round(float(np.mean(sub_energy[i:j + 1])), 2),
            "beats": max(1, int(round((j - i + 1) / 2))),
            "subbeats": int(j - i + 1),
            "source": "detected",
        })
        i = j + 1
    chords = _clean_chords(chords)

    sections = detect_sections(chroma, y, sr, edges, edge_times, energy, beats_per_bar, duration)

    return {
        "duration": round(duration, 2),
        "key": {**key, "kind": "detected"},
        "bpm": {"value": bpm, "confidence": round(bpm_conf, 2), "kind": "detected",
                "min": tmap["min_bpm"], "max": tmap["max_bpm"]},
        "tempo_map": tmap["segments"],
        "tempo_varies": tmap["varies"],
        "beats": [round(float(t), 3) for t in beat_times],
        "time_signature": {**time_sig, "kind": "estimated"},
        "chords": chords,
        "sections": sections,
        "beats_per_bar": beats_per_bar,
        "first_beat": round(float(beat_times[0]), 3),
        "notes": [
            "Chords are detected from the recording and may contain mistakes: please review and correct them.",
            "Section labels and time signature are estimates.",
            "The piano track is an approximation of the original arrangement.",
        ],
    }


def _clean_chords(chords: list[dict]) -> list[dict]:
    """Merge 1-beat blips into a neighbour and re-merge equal neighbours."""
    changed = True
    while changed and len(chords) > 2:
        changed = False
        for k in range(1, len(chords) - 1):
            c = chords[k]
            if c.get("subbeats", 2 * c["beats"]) <= 2 and c["confidence"] < 0.35:
                prev = chords[k - 1]
                prev["end"] = c["end"]
                prev["beats"] += c["beats"]
                prev["subbeats"] = prev.get("subbeats", 0) + c.get("subbeats", 0)
                del chords[k]
                changed = True
                break
        out = []
        for c in chords:
            if out and out[-1]["chord"] == c["chord"]:
                out[-1]["end"] = c["end"]
                out[-1]["beats"] += c["beats"]
                out[-1]["subbeats"] = out[-1].get("subbeats", 0) + c.get("subbeats", 0)
                out[-1]["energy"] = round((out[-1]["energy"] + c["energy"]) / 2, 2)
            else:
                out.append(c)
        if len(out) != len(chords):
            changed = True
        chords = out
    return chords


def detect_sections(chroma, y, sr, edges, edge_times, energy, beats_per_bar, duration) -> list[dict]:
    """Find section boundaries with a bar-level novelty curve, then label by repetition.

    Verses and choruses often share chords, so the features also include timbre
    and loudness (drums, strings and a fuller sound usually mark a chorus).
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=HOP, n_mfcc=13)
    rms = np.log1p(librosa.feature.rms(y=y, hop_length=HOP)[0] * 50)
    n = min(chroma.shape[1], mfcc.shape[1], rms.shape[0])
    edges = edges[edges < n]

    def z(m):
        return (m - m.mean(axis=1, keepdims=True)) / (m.std(axis=1, keepdims=True) + 1e-6)

    chroma_b = librosa.util.sync(chroma[:, :n], edges, aggregate=np.mean)
    mfcc_b = librosa.util.sync(mfcc[1:, :n], edges, aggregate=np.mean)
    rms_b = librosa.util.sync(rms[None, :n], edges, aggregate=np.mean)
    feat = np.vstack([0.6 * z(chroma_b), 0.8 * z(mfcc_b), 3.0 * z(rms_b)])
    n_beats = feat.shape[1]

    bpb = beats_per_bar
    n_bars = n_beats // bpb
    if n_bars < 8:
        return [{"label": "Song", "kind": "verse", "start": 0.0, "end": round(duration, 2), "energy": 0.5, "source": "estimated"}]
    bar_feat = feat[:, : n_bars * bpb].reshape(feat.shape[0], n_bars, bpb).mean(axis=2)
    nrm = bar_feat / (np.linalg.norm(bar_feat, axis=0, keepdims=True) + 1e-9)
    ssm = nrm.T @ nrm

    # Foote checkerboard novelty
    L = 4 if n_bars >= 24 else 2
    g = np.exp(-0.5 * (np.linspace(-1.5, 1.5, 2 * L) ** 2))
    kern = np.outer(g, g) * np.outer(np.sign(np.arange(2 * L) - L + 0.5), np.sign(np.arange(2 * L) - L + 0.5))
    pad = np.pad(ssm, L, mode="edge")
    nov = np.array([(kern * pad[i:i + 2 * L, i:i + 2 * L]).sum() for i in range(n_bars + 1)])[: n_bars]
    max_sections = int(np.clip(duration / 18, 3, 12))
    order = np.argsort(nov)[::-1]
    min_gap = max(3, L)
    picks: list[int] = []
    for i in order:
        if i < 2 or i > n_bars - 2 or nov[i] < 0.28 * nov.max():
            continue
        if all(abs(i - p) >= min_gap for p in picks):
            picks.append(int(i))
        if len(picks) >= max_sections - 1:
            break
    bar_bounds = [0] + sorted(picks) + [n_bars]

    segs = []
    for a, b in zip(bar_bounds[:-1], bar_bounds[1:]):
        v = nrm[:, a:b].mean(axis=1)
        e0, e1 = a * bpb, min(b * bpb, len(energy))
        segs.append({
            "a": a * bpb, "b": (n_beats if b == n_bars else b * bpb), "vec": v / (np.linalg.norm(v) + 1e-9),
            "energy": float(energy[e0:e1].mean()) if e0 < len(energy) else 0.0,
        })

    # group repeated sections by similarity of their mean feature vector
    groups: list[list[int]] = []
    for idx, s in enumerate(segs):
        for g_ in groups:
            if float(segs[g_[0]]["vec"] @ s["vec"]) > 0.88:
                g_.append(idx)
                break
        else:
            groups.append([idx])

    labels = ["verse"] * len(segs)
    repeated = [g_ for g_ in groups if len(g_) >= 2]
    if repeated:
        chorus_g = max(repeated, key=lambda g_: np.mean([segs[i]["energy"] for i in g_]))
        if len(groups) > 1:
            for i in chorus_g:
                labels[i] = "chorus"
    elif len(segs) > 2:
        loudest = max(range(len(segs)), key=lambda i: segs[i]["energy"])
        labels[loudest] = "chorus"
    for i in range(1,
