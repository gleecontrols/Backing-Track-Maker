"""Audio rendering: MIDI -> piano audio (SoundFont) -> reverb/mix/master -> export."""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf
from scipy import signal
from pedalboard import HighpassFilter, HighShelfFilter, LowShelfFilter, Pedalboard, Compressor

SR = 44100
ROOT = Path(__file__).resolve().parent.parent
SF_DIR = ROOT / "soundfonts"
PEAK_TARGET = 0.84  # about -1.5 dBFS: leaves headroom for MP3 encoding overshoot

REVERB_PRESETS = {
    "low": {"intensity": 0.35, "decay": 2.2, "mix": 0.25},
    "medium": {"intensity": 0.60, "decay": 4.0, "mix": 0.40},
    "high": {"intensity": 0.85, "decay": 7.0, "mix": 0.55},
}


@dataclass
class FxParams:
    intensity: float = 0.85    # 0..1  how strong the reverb send is
    decay: float = 7.0         # seconds (RT60)
    mix: float = 0.55          # 0..1  wet/dry
    volume: float = 0.9        # 0..1  final level (1.0 = -1.5 dBFS peak)
    width: float = 1.15        # 0..2  stereo width (1 = unchanged)
    tone: float = -0.15        # -1 (warm) .. +1 (bright)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FxParams":
        d = d or {}
        preset = d.get("preset")
        base = dict(REVERB_PRESETS.get(preset, {})) if preset else {}
        p = cls()
        for k in ("intensity", "decay", "mix", "volume", "width", "tone"):
            if k in base:
                setattr(p, k, float(base[k]))
            if d.get(k) is not None:
                setattr(p, k, float(d[k]))
        p.intensity = float(np.clip(p.intensity, 0, 1))
        p.decay = float(np.clip(p.decay, 0.3, 12))
        p.mix = float(np.clip(p.mix, 0, 1))
        p.volume = float(np.clip(p.volume, 0.05, 1))
        p.width = float(np.clip(p.width, 0, 2))
        p.tone = float(np.clip(p.tone, -1, 1))
        return p


def find_soundfont() -> Path:
    env = os.environ.get("PIANO_SF2")
    if env and Path(env).exists():
        return Path(env)
    cands = sorted(SF_DIR.glob("*.sf2"), key=lambda p: p.stat().st_size, reverse=True)
    if not cands:
        raise RuntimeError("No .sf2 SoundFont found in soundfonts/. Add a piano SoundFont there.")
    return cands[0]


def render_dry(pm: pretty_midi.PrettyMIDI, sf2: Path | None = None) -> np.ndarray:
    """Render MIDI with a SoundFont. Returns float32 array shaped (2, n)."""
    import tinysoundfont

    sf2 = sf2 or find_soundfont()
    synth = tinysoundfont.Synth(gain=-3.0, samplerate=SR)
    sfid = synth.sfload(str(sf2))
    synth.program_select(0, sfid, 0, 0)  # GM bank 0, preset 0 = Acoustic Grand Piano

    events = []  # (time, order, kind, a, b)
    inst = pm.instruments[0]
    for n in inst.notes:
        events.append((n.end, 0, "off", n.pitch, 0))
        events.append((n.start, 2, "on", n.pitch, n.velocity))
    for c in inst.control_changes:
        events.append((c.time, 1, "cc", c.number, c.value))
    events.sort(key=lambda e: (e[0], e[1]))

    end_t = max(e[0] for e in events) + 2.5  # let the release ring out
    total = int(end_t * SR)
    out = np.zeros((total, 2), dtype=np.float32)
    pos = 0
    for t, _, kind, a, b in events:
        target = min(int(round(t * SR)), total)
        if target > pos:
            buf = np.frombuffer(synth.generate(target - pos), dtype=np.float32).reshape(-1, 2)
            out[pos:target] = buf
            pos = target
        if kind == "on":
            synth.noteon(0, int(a), int(b))
        elif kind == "off":
            synth.noteoff(0, int(a))
        else:
            synth.control_change(0, int(a), int(b))
    if pos < total:
        buf = np.frombuffer(synth.generate(total - pos), dtype=np.float32).reshape(-1, 2)
        out[pos:total] = buf
    return out.T.copy()


@lru_cache(maxsize=16)
def _impulse_response(decay: float) -> np.ndarray:
    """Synthetic concert-hall impulse response: decorrelated stereo, darker as it decays."""
    rng = np.random.default_rng(42)
    n = int(SR * min(decay * 1.25, 14))
    t = np.arange(n) / SR
    out = np.zeros((2, n))
    bands = [((20, 400), 1.18), ((400, 4000), 1.0), ((4000, 14000), 0.55)]
    for ch in range(2):
        acc = np.zeros(n)
        for (lo, hi), rt_scale in bands:
            noise = rng.standard_normal(n)
            sos = signal.butter(2, [lo, hi], btype="band", fs=SR, output="sos")
            band = signal.sosfilt(sos, noise)
            env = np.exp(-6.91 * t / (decay * rt_scale))
            acc += band * env
        out[ch] = acc
    # smooth onset + pre-delay (hall feel) and a few early reflections
    onset = np.clip(t / 0.03, 0, 1) ** 2
    out *= onset
    pre = int(0.028 * SR)
    out = np.concatenate([np.zeros((2, pre)), out], axis=1)
    for dt, g, ch in [(0.041, 0.35, 0), (0.057, 0.30, 1), (0.083, 0.22, 0), (0.101, 0.2, 1)]:
        i = int(dt * SR)
        out[ch, i] += g * np.abs(out[ch]).max() * 0.6
    out /= np.sqrt((out ** 2).sum(axis=1, keepdims=True)) + 1e-12
    return out.astype(np.float32)


def apply_fx(dry: np.ndarray, fx: FxParams) -> np.ndarray:
    """Tone, width, reverb, level, fades. Takes (2, n); returns (2, m) with reverb tail."""
    x = dry.astype(np.float32)

    # stereo width (mid/side)
    mid, side = (x[0] + x[1]) / 2, (x[0] - x[1]) / 2
    side = side * fx.width
    x = np.stack([mid + side, mid - side])

    # tone: warm <-> bright
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=28),
        LowShelfFilter(cutoff_frequency_hz=220, gain_db=-fx.tone * 2.0),
        HighShelfFilter(cutoff_frequency_hz=3200, gain_db=fx.tone * 6.0),
        Compressor(threshold_db=-20, ratio=1.6, attack_ms=25, release_ms=250),
    ])
    x = board(x, SR)

    # reverb via convolution; pad so the tail is not cut off
    tail = int(min(fx.decay * 1.1, 13) * SR) if fx.mix > 0.001 else int(1.0 * SR)
    padded = np.concatenate([x, np.zeros((2, tail), dtype=np.float32)], axis=1)
    if fx.mix > 0.001:
        ir = _impulse_response(round(fx.decay, 1))
        wet = np.stack([signal.oaconvolve(padded[c], ir[c], mode="full")[: padded.shape[1]] for c in range(2)])
        # IR is energy-normalised, so scale wet back to roughly match the dry level
        wet *= 7.0
        wet_gain = fx.mix * (0.45 + 1.0 * fx.intensity)
        dry_gain = 1.0 - 0.55 * fx.mix
        y = padded * dry_gain + wet * wet_gain
    else:
        y = padded

    # trim silent tail, then fade out so it never ends abruptly
    env = np.abs(y).max(axis=0)
    thresh = max(env.max() * 10 ** (-65 / 20), 1e-6)
    idx = np.nonzero(env > thresh)[0]
    end = int(idx[-1]) + 1 if len(idx) else y.shape[1]
    y = y[:, :end]
    fade_len = int(min(2.0, max(0.6, y.shape[1] / SR * 0.05)) * SR)
    fade_len = min(fade_len, y.shape[1] // 2)
    y[:, -fade_len:] *= (0.5 * (1 + np.cos(np.linspace(0, np.pi, fade_len))))[None, :]
    fi = int(0.015 * SR)
    y[:, :fi] *= np.linspace(0, 1, fi)[None, :]

    # level: normalise to a safe peak, scaled by the volume control
    peak = float(np.abs(y).max()) + 1e-12
    y = y * (PEAK_TARGET * fx.volume / peak)
    y = np.clip(y, -0.999, 0.999)
    return y.astype(np.float32)


def write_wav(path: Path, audio: np.ndarray) -> None:
    sf.write(str(path), audio.T, SR, subtype="PCM_24")


def write_mp3(wav_path: Path, mp3_path: Path, bitrate: str = "256k") -> None:
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(wav_path), "-codec:a", "libmp3lame", "-b:a", bitrate, str(mp3_path)],
        capture_output=True, text=True, timeout=300,
    )
    if r.returncode != 0:
        raise RuntimeError("MP3 encoding failed: " + r.stderr[-200:])


def audio_report(audio: np.ndarray) -> dict:
    peak = float(np.abs(audio).max())
    rms = float(np.sqrt((audio ** 2).mean()))
    return {
        "peak_dbfs": round(20 * np.log10(peak + 1e-12), 2),
        "rms_dbfs": round(20 * np.log10(rms + 1e-12), 2),
        "clipped_samples": int((np.abs(audio) >= 0.999).sum()),
        "duration": round(audio.shape[1] / SR, 2),
        "last_samples_abs_max": float(np.abs(audio[:, -50:]).max()),
    }
