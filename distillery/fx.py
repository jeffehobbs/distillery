"""Effects, via pedalboard.

pedalboard takes (samples, channels) float32 arrays — the same layout used
everywhere else here — so buffers pass straight through with no transposing.

Two layers:

  * `event_filter` / `bus_glue` — per-loop and per-bus character, used on every run.
  * the **delights**: one or two effects rolled per run by the arranger and applied
    at specific bars (a filter sweep into a drop, a reverb throw on the bar before
    a transition, a dub echo out of a breakdown, crushed drums under the climax).
    They're recorded in the plan and printed in the report, so a run you like can
    be reproduced from its seed.

Everything degrades gracefully: if pedalboard is missing, `available()` is False
and the numpy/scipy paths in `audio.py` are used instead.
"""
import numpy as np

from . import audio, config

try:
    import pedalboard as pb
    _PB = True
except ImportError:                       # pragma: no cover - optional dependency
    pb = None
    _PB = False


def available():
    return _PB and config.FX_ENABLED


def version():
    return pb.__version__ if _PB else None


def apply(plugins, x, sr=config.SR, reset=True):
    """Run x through a list of pedalboard plugins."""
    if not available() or not plugins:
        return audio.to_stereo(x)
    board = pb.Pedalboard(list(plugins))
    out = board(np.ascontiguousarray(audio.to_stereo(x)), sr, reset=reset)
    return np.asarray(out, dtype=np.float32)


# ---------------------------------------------------------------- plugin builders

def ladder_lpf(cutoff_hz, resonance=0.15, drive=1.0):
    return pb.LadderFilter(mode=pb.LadderFilter.Mode.LPF12,
                           cutoff_hz=float(np.clip(cutoff_hz, 30.0, 18000.0)),
                           resonance=float(np.clip(resonance, 0.0, 0.85)),
                           drive=float(drive))


def ladder_hpf(cutoff_hz, resonance=0.1):
    return pb.LadderFilter(mode=pb.LadderFilter.Mode.HPF12,
                           cutoff_hz=float(np.clip(cutoff_hz, 20.0, 8000.0)),
                           resonance=float(np.clip(resonance, 0.0, 0.85)))


def reverb(room_size=0.6, damping=0.5, wet=0.3, width=1.0):
    return pb.Reverb(room_size=float(room_size), damping=float(damping),
                     wet_level=float(wet), dry_level=float(1.0 - wet),
                     width=float(width))


def delay(seconds, feedback=0.35, mix=0.3):
    return pb.Delay(delay_seconds=float(seconds), feedback=float(feedback),
                    mix=float(mix))


def chorus(rate_hz=0.6, depth=0.25, mix=0.3):
    return pb.Chorus(rate_hz=float(rate_hz), depth=float(depth), mix=float(mix))


def phaser(rate_hz=0.3, depth=0.5, mix=0.4):
    return pb.Phaser(rate_hz=float(rate_hz), depth=float(depth), mix=float(mix))


def drive(drive_db=8.0):
    return pb.Distortion(drive_db=float(drive_db))


def crush(bit_depth=8.0):
    return pb.Bitcrush(bit_depth=float(bit_depth))


def compressor(threshold_db=-18.0, ratio=2.0, attack_ms=8.0, release_ms=120.0):
    return pb.Compressor(threshold_db=float(threshold_db), ratio=float(ratio),
                         attack_ms=float(attack_ms), release_ms=float(release_ms))


# ---------------------------------------------------------------- per-event / bus

def event_filter(x, cutoff_hz, energy, sr=config.SR):
    """The per-loop tone shaping: a resonant ladder lowpass instead of a plain
    butterworth. Resonance eases off as energy rises, so early loops are filtered
    and vocal-sounding and the climax is open and flat."""
    if not available():
        return audio.lpf(x, cutoff_hz, sr, order=4)
    res = float(np.clip(0.45 * (1.0 - energy), 0.05, 0.45))
    return apply([ladder_lpf(cutoff_hz, resonance=res)], x, sr)


def ambient_smear(x, sr=config.SR, mix=0.35):
    """Chorus + reverb for the drenched, distant loops in the quiet sections."""
    if not available():
        return audio.reverb(x, wet=mix, seconds=2.6, seed=11)
    return apply([chorus(rate_hz=0.35, depth=0.35, mix=0.4),
                  reverb(room_size=0.85, damping=0.35, wet=mix, width=1.0)], x, sr)


def sweep(x, f_start, f_end, sr=config.SR, resonance=0.35, chunks=96):
    """Time-varying resonant lowpass. Plugin state is kept across chunks
    (reset=False), which is what keeps the sweep from clicking at chunk seams."""
    x = audio.to_stereo(x)
    if not available():
        return audio.lpf_sweep(x, f_start, f_end, sr)
    n = x.shape[0]
    if n < chunks * 8:
        return apply([ladder_lpf(min(f_start, f_end), resonance)], x, sr)
    freqs = np.geomspace(max(30.0, f_start), max(30.0, f_end), chunks)
    bounds = np.linspace(0, n, chunks + 1).astype(int)
    board = pb.Pedalboard([ladder_lpf(freqs[0], resonance)])
    out = np.empty_like(x)
    for i, f in enumerate(freqs):
        board[0].cutoff_hz = float(np.clip(f, 30.0, 18000.0))
        s, e = bounds[i], bounds[i + 1]
        if e > s:
            out[s:e] = board(np.ascontiguousarray(x[s:e]), sr, reset=False)
    return out.astype(np.float32)


# ---------------------------------------------------------------- delights

# name -> (needs_section, description)
DELIGHTS = {
    "filter-sweep": "resonant lowpass closing then opening into a big section",
    "reverb-throw": "wet-only reverb on the last bar before a transition",
    "dub-echo": "feedback delay throw coming out of a breakdown",
    "drum-crush": "bitcrush + drive on the loop bus under the climax",
    "phaser-drift": "slow phaser over the atmosphere bed",
    "tape-warp": "chorus + pitch drift on the loops in a quiet section",
}


def roll_delights(rng, sections, count_range=config.DELIGHTS_PER_RUN):
    """Choose this run's delights and place them at real bar positions."""
    if not available():
        return []
    by_name = {s["name"]: s for s in sections}
    names = list(DELIGHTS)
    n = int(rng.integers(count_range[0], count_range[1] + 1))
    picked = list(rng.permutation(names))[:n]
    out = []
    for name in picked:
        if name == "filter-sweep":
            s = by_name.get("climax") or by_name.get("build-2")
            if s and s["start_bar"] >= 8:
                out.append({"name": name, "bar": s["start_bar"] - 8, "bars": 8,
                            "f_start": 16000.0, "f_end": 600.0})
        elif name == "reverb-throw":
            s = by_name.get("build-2") or by_name.get("groove-2")
            if s and s["start_bar"] >= 1:
                out.append({"name": name, "bar": s["start_bar"] - 1, "bars": 1,
                            "wet": 0.85})
        elif name == "dub-echo":
            s = by_name.get("breakdown-1")
            if s:
                out.append({"name": name, "bar": s["start_bar"] + s["bars"] - 2,
                            "bars": 4, "beats": float(rng.choice([0.75, 1.0, 1.5])),
                            "feedback": 0.55})
        elif name == "drum-crush":
            s = by_name.get("climax")
            if s:
                out.append({"name": name, "bar": s["start_bar"] + s["bars"] // 2,
                            "bars": min(16, s["bars"] // 2), "bits": 7.0,
                            "drive_db": 9.0})
        elif name == "phaser-drift":
            s = by_name.get("groove-1")
            if s:
                out.append({"name": name, "bar": s["start_bar"], "bars": s["bars"],
                            "rate": 0.22})
        elif name == "tape-warp":
            s = by_name.get("breakdown-2") or by_name.get("breakdown-1")
            if s:
                out.append({"name": name, "bar": s["start_bar"], "bars": s["bars"],
                            "semitones": -0.25})
    return out


def _slice(bus, bar, bars, sr=config.SR):
    s = int(bar * config.SEC_PER_BAR * sr)
    e = min(bus.shape[0], int((bar + bars) * config.SEC_PER_BAR * sr))
    return s, e


def apply_delight(d, loop_bus, atmos_bus, sr=config.SR):
    """Apply one delight in place. Returns a short log line."""
    name = d["name"]
    if name == "filter-sweep":
        s, e = _slice(loop_bus, d["bar"], d["bars"], sr)
        loop_bus[s:e] = sweep(loop_bus[s:e], d["f_start"], d["f_end"], sr,
                              resonance=0.4)
        return f"filter-sweep bars {d['bar']}-{d['bar'] + d['bars'] - 1} " \
               f"({d['f_start']:.0f}->{d['f_end']:.0f} Hz)"
    if name == "reverb-throw":
        s, e = _slice(loop_bus, d["bar"], d["bars"], sr)
        wet = apply([reverb(room_size=0.9, damping=0.25, wet=1.0)], loop_bus[s:e], sr)
        tail = min(loop_bus.shape[0] - s, wet.shape[0])
        loop_bus[s:s + tail] += audio.gain_db(wet[:tail], -3.0)
        return f"reverb-throw bar {d['bar']}"
    if name == "dub-echo":
        s, e = _slice(loop_bus, d["bar"], d["bars"], sr)
        seg = apply([delay(d["beats"] * config.SEC_PER_BEAT, feedback=d["feedback"],
                           mix=0.5),
                     ladder_lpf(2600.0, resonance=0.2)], loop_bus[s:e], sr)
        loop_bus[s:e] = seg[:e - s]
        return f"dub-echo bars {d['bar']}-{d['bar'] + d['bars'] - 1} " \
               f"({d['beats']:g}-beat)"
    if name == "drum-crush":
        s, e = _slice(loop_bus, d["bar"], d["bars"], sr)
        seg = loop_bus[s:e]
        before = audio.peak_dbfs(seg)
        seg = apply([crush(d["bits"]), drive(d["drive_db"]),
                     ladder_lpf(14000.0, resonance=0.1)], seg, sr)
        # peak-match so a drive stage can't hijack the master's headroom
        loop_bus[s:e] = audio.gain_db(seg[:e - s], before - audio.peak_dbfs(seg))
        return f"drum-crush bars {d['bar']}-{d['bar'] + d['bars'] - 1} " \
               f"({d['bits']:g}-bit, {d['drive_db']:g}dB)"
    if name == "phaser-drift":
        s, e = _slice(atmos_bus, d["bar"], d["bars"], sr)
        atmos_bus[s:e] = apply([phaser(rate_hz=d["rate"], depth=0.6, mix=0.45)],
                               atmos_bus[s:e], sr)[:e - s]
        return f"phaser-drift bars {d['bar']}-{d['bar'] + d['bars'] - 1}"
    if name == "tape-warp":
        s, e = _slice(loop_bus, d["bar"], d["bars"], sr)
        loop_bus[s:e] = apply([pb.PitchShift(semitones=d["semitones"]),
                               chorus(rate_hz=0.25, depth=0.5, mix=0.5)],
                              loop_bus[s:e], sr)[:e - s]
        return f"tape-warp bars {d['bar']}-{d['bar'] + d['bars'] - 1}"
    return f"(unknown delight {name})"
