"""The bed: a synthesized ambient techno track at the target tempo (--bpm).
No samples anywhere.

Every voice is generated from oscillators and filtered noise:

  kick   sine with an exponential pitch drop (110 -> 42 Hz) plus a HP'd click
  sub    saturated sine an octave or two under the chord root, ducked by the kick
  hat    band-passed noise, short decay, offbeat 8ths (16ths when energy is up)
  clap   four noise bursts 9-13 ms apart through a band-pass, into reverb
  perc   brighter noise blip on syncopated 16ths
  pad    three detuned saws per chord note, slow attack, LFO'd lowpass, big reverb
  wash   lowpassed noise breathing under everything
  riser  noise band sweeping up + rising sine, to lift into a new section

The chord progression is derived from the album's own key, so the bed is in the
same tonality as the source material.
"""
import math

import numpy as np
from scipy import signal

from . import audio, config, parallel

SR = config.SR

MINOR_PROG = [(0, (0, 3, 7, 10)), (8, (0, 4, 7, 11)),
              (3, (0, 4, 7, 11)), (10, (0, 4, 7, 10))]
MAJOR_PROG = [(0, (0, 4, 7, 11)), (9, (0, 3, 7, 10)),
              (5, (0, 4, 7, 11)), (7, (0, 4, 7, 10))]


def midi_hz(m):
    return 440.0 * (2.0 ** ((m - 69) / 12.0))


def key_root_midi(key, scale, low=48):
    """Root MIDI note for the album key, in a low-ish register."""
    from .analyze import NOTE_SEMITONE
    semi = NOTE_SEMITONE.get(key or "A", 9)
    return low + semi


def drone_midi(key, low=None):
    """The album key's pitch class, placed in one fixed low octave.

    Anchoring the octave (rather than offsetting from the chord root) keeps the
    drone in the same 82-155 Hz register for every key.
    """
    from .analyze import NOTE_SEMITONE
    low = config.DRONE_LOW_MIDI if low is None else low
    semi = NOTE_SEMITONE.get(key or "A", 9)
    return low + ((semi - (low % 12)) % 12)


def progression(key, scale, chords=None):
    """Chord progression for the bed, or a single-note drone when chords are off.

    Chords are off by default: the bed holds one sustained root+octave drone
    (no third, no seventh — so there is no chord quality and nothing moves
    harmonically) and all the motion comes from the drums, the loops and the
    filters. `--chords` restores the four-chord progression.
    """
    if chords is None:
        chords = config.CHORDS
    if not chords:
        return [(0, tuple(config.DRONE_INTERVALS))]
    return MINOR_PROG if (scale or "minor").startswith("min") else MAJOR_PROG


# ---------------------------------------------------------------- one-shots

def kick(sr=SR, dur=0.62, f_start=118.0, f_end=42.0, pitch_tau=0.032,
         amp_tau=None, click=None, drive=1.6):
    amp_tau = config.KICK_AMP_TAU if amp_tau is None else amp_tau
    click = config.KICK_CLICK if click is None else click
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    f = f_end + (f_start - f_end) * np.exp(-t / pitch_tau)
    ph = 2 * np.pi * np.cumsum(f) / sr
    body = np.sin(ph).astype(np.float32) * np.exp(-t / amp_tau)
    rng = np.random.default_rng(11)
    tick = rng.normal(0, 1, n).astype(np.float32) * np.exp(-t / 0.0035) * click
    tick = audio.hpf(tick, 1800.0, sr, order=2)
    x = np.tanh((body + tick) * drive) / math.tanh(drive)
    x = audio.hpf(x, config.KICK_HPF_HZ, sr, order=2)   # drop inaudible rumble
    x = audio.fade(x, in_s=0.0005, out_s=0.02, sr=sr)
    return audio.to_stereo(x * 0.9)


def hat(sr=SR, dur=0.09, lo=6500.0, hi=12500.0, tau=0.022, seed=3):
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).astype(np.float32)
    x = audio.bpf(x, lo, hi, sr, order=4) * audio.exp_decay(n, sr, tau)
    return audio.to_stereo(audio.fade(x, out_s=0.005, sr=sr) * 0.5)


def clap(sr=SR, dur=0.5, seed=5):
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)
    for i, (d_ms, g) in enumerate(((0, 1.0), (9, 0.8), (19, 0.65), (30, 0.5))):
        d = int(d_ms * sr / 1000)
        burst = rng.normal(0, 1, n - d).astype(np.float32) * \
            audio.exp_decay(n - d, sr, 0.012 if i < 3 else 0.09) * g
        x[d:] += burst
    x = audio.bpf(x, 900.0, 3600.0, sr, order=4)
    x = audio.reverb(x, wet=0.34, seconds=1.1, damp_hz=5200.0, predelay_ms=8.0, seed=5)
    return audio.to_stereo(audio.fade(x, out_s=0.05, sr=sr) * 0.55)


def perc(sr=SR, dur=0.16, seed=9):
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, n).astype(np.float32)
    x = audio.bpf(x, 2600.0, 6800.0, sr, order=4) * audio.exp_decay(n, sr, 0.035)
    x = audio.reverb(x, wet=0.25, seconds=0.9, damp_hz=6000.0, seed=9)
    return audio.to_stereo(x * 0.4)


def rim(sr=SR, dur=0.12, seed=13):
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    x = (np.sin(2 * np.pi * 1750 * t) * 0.6 + np.sin(2 * np.pi * 410 * t) * 0.4)
    x = x.astype(np.float32) * audio.exp_decay(n, sr, 0.014)
    return audio.to_stereo(audio.hpf(x, 300.0, sr, order=2) * 0.35)


def sub_note(hz, dur, sr=SR, drive=1.25, lpf_hz=220.0, harm2=0.18, attack=0.012,
             glide_from=None, glide_ms=0.0):
    """One bass note. Timbre and portamento come from the run's bass design."""
    n = int(dur * sr)
    t = np.arange(n, dtype=np.float32) / sr
    if glide_from and glide_ms > 0 and abs(glide_from - hz) > 0.5:
        # portamento: slide into the note over the first few tens of ms
        gn = min(n, max(1, int(glide_ms * sr / 1000.0)))
        f = np.full(n, hz, dtype=np.float32)
        f[:gn] = np.geomspace(max(1.0, glide_from), hz, gn)
        ph = 2 * np.pi * np.cumsum(f) / sr
    else:
        ph = 2 * np.pi * hz * t
    x = np.sin(ph).astype(np.float32)
    x += harm2 * np.sin(2 * ph).astype(np.float32)
    env = audio.env_adsr(n, sr, a=attack, d=0.05, s=0.85, r=min(0.25, dur * 0.3))
    x = np.tanh(x * env * drive) / math.tanh(drive)
    return audio.to_stereo(audio.lpf(x, lpf_hz, sr, order=2) * 0.75)


def bass_design(rng, scale, chords=None):
    """Roll this run's bassline: rhythm, safe degrees, timbre.

    Every choice is seeded, so a seed still reproduces the piece exactly — but two
    seeds now differ in the one element that used to be identical in every render.
    """
    mode = "major" if not (scale or "minor").startswith("min") else "minor"
    pat_names = sorted(config.BASS_PATTERNS)
    name = str(rng.choice(pat_names))
    degrees = tuple(config.BASS_DEGREES[mode][
        int(rng.integers(0, len(config.BASS_DEGREES[mode])))])
    return {
        "pattern": name,
        "degrees": list(degrees),
        "drive": round(float(rng.uniform(1.05, 1.85)), 3),
        "lpf_hz": round(float(rng.uniform(170.0, 320.0)), 1),
        "harm2": round(float(rng.uniform(0.05, 0.32)), 3),
        "attack": round(float(rng.choice([0.004, 0.012, 0.03])), 4),
        "glide_ms": float(rng.choice([0.0, 0.0, 12.0, 35.0])),
        "octave": int(rng.choice([-12, -12, -12, 0])),   # mostly deep, sometimes up
    }


def _saw(hz, n, sr=SR, phase=0.0):
    t = np.arange(n, dtype=np.float32) / sr
    return signal.sawtooth(2 * np.pi * hz * t + phase).astype(np.float32)


def pad_voice(root_midi, intervals, dur, sr=SR, seed=17, brightness=1.0,
              detune_cents=7.0):
    """Detuned saw stack -> LFO'd lowpass -> long reverb -> widened.

    `intervals` is a chord when chords are enabled, or (0, 12) for the default
    drone. Same synthesis either way.
    """
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    left = np.zeros(n, dtype=np.float32)
    right = np.zeros(n, dtype=np.float32)
    for k, iv in enumerate(intervals):
        # the +12 spread only applies to 4-note chord voicings; a 1- or 2-note
        # drone must never get pushed up an octave
        m = root_midi + iv + (12 if (k >= 3 and len(intervals) >= 4) else 0)
        base = midi_hz(m)
        for d in (-detune_cents, 0.0, detune_cents):
            hz = base * (2 ** (d / 1200.0))
            ph = float(rng.uniform(0, 2 * np.pi))
            s = _saw(hz, n, sr, ph) * 0.16
            spread = 0.5 + 0.5 * ((k + (d > 0)) % 2)
            left += s * spread
            right += s * (1.5 - spread)
    x = np.stack([left, right], axis=1)
    # slow filter movement: two LFO cycles across the chord
    lfo = 0.5 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, 12, dtype=np.float32))
    # kept dark on purpose: a bright saw drone crowds the midrange and reads as a
    # lead line rather than atmosphere
    cut_lo, cut_hi = 240.0 * brightness, config.DRONE_TOP_HZ * brightness
    seg_bounds = np.linspace(0, n, 13).astype(int)
    zi = None
    out = np.empty_like(x)
    for i in range(12):
        sos = signal.butter(4, np.clip((cut_lo + (cut_hi - cut_lo) * lfo[i]) / (sr / 2),
                                       1e-4, 0.999), btype="lowpass", output="sos")
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, 2), dtype=np.float32)
        s, e = seg_bounds[i], seg_bounds[i + 1]
        seg, zi = signal.sosfilt(sos, x[s:e], axis=0, zi=zi)
        out[s:e] = seg
    out = audio.hpf(out, 90.0, sr, order=2)
    # swells rather than sits: slow attack, low sustain, long release
    env = audio.env_adsr(n, sr, a=min(4.0, dur * 0.45), d=dur * 0.2, s=0.5,
                         r=min(6.0, dur * 0.5))
    out *= env[:, None]
    # mostly wet — distance is what keeps it behind the drums
    out = audio.reverb(out, wet=0.8, seconds=4.2, damp_hz=3600.0, predelay_ms=35.0,
                       seed=17)[:n + int(1.5 * sr)]
    return audio.widen(out, 0.35) * 0.4


def wash(dur, sr=SR, seed=23, cutoff=900.0, depth=0.5):
    """Slowly breathing filtered-noise bed."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)).astype(np.float32)
    x = audio.lpf(x, cutoff, sr, order=4)
    x = audio.hpf(x, 120.0, sr, order=2)
    t = np.arange(n, dtype=np.float32) / sr
    lfo = (1.0 - depth) + depth * (0.5 + 0.5 * np.sin(2 * np.pi * t / 17.0 +
                                                      0.7 * np.sin(2 * np.pi * t / 41.0)))
    return (x * lfo[:, None] * 0.10).astype(np.float32)


def riser(dur, sr=SR, seed=29, f0=180.0, f1=5200.0, tone_midi=None):
    """Noise band sweeping up plus a rising sine — the lift into a new section."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)).astype(np.float32)
    chunks = 64
    bounds = np.linspace(0, n, chunks + 1).astype(int)
    freqs = np.geomspace(f0, f1, chunks)
    out = np.zeros_like(x)
    for i, f in enumerate(freqs):
        s, e = bounds[i], bounds[i + 1]
        out[s:e] = audio.bpf(x[s:e], f * 0.8, min(sr * 0.45, f * 1.6), sr, order=2)
    t = np.arange(n, dtype=np.float32) / sr
    if tone_midi is not None:
        f = midi_hz(tone_midi) * np.exp(np.linspace(0, math.log(4.0), n, dtype=np.float32))
        tone = np.sin(2 * np.pi * np.cumsum(f) / sr).astype(np.float32) * 0.25
        out += np.stack([tone, tone], axis=1)
    env = (t / max(1e-3, t[-1])) ** 2.0
    out *= env[:, None]
    return audio.fade(out * 0.5, in_s=0.05, out_s=0.02, sr=sr)


def impact(sr=SR, dur=2.8, seed=31):
    """Reverb-drenched downbeat hit for section starts."""
    n = int(dur * sr)
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, (n, 2)).astype(np.float32) * audio.exp_decay(n, sr, 0.05)[:, None]
    x = audio.lpf(x, 3200.0, sr, order=2)
    t = np.arange(n, dtype=np.float32) / sr
    boom = (np.sin(2 * np.pi * (58 * np.exp(-t / 0.25) + 34) * t) *
            np.exp(-t / 0.5)).astype(np.float32) * 0.7
    x += np.stack([boom, boom], axis=1)
    x = audio.reverb(x, wet=0.55, seconds=3.2, damp_hz=4000.0, seed=31)[:n]
    return x * 0.5


# ---------------------------------------------------------------- the bed

def _kick_times(plan, rng):
    """Four-on-the-floor with occasional 16th ghost notes at higher energy."""
    times, gains = [], []
    for b in plan.bars:
        if not b.kick:
            continue
        t0 = b.index * config.SEC_PER_BAR
        for beat in range(config.BEATS_PER_BAR):
            times.append(t0 + beat * config.SEC_PER_BEAT)
            gains.append(1.0)
        if b.energy > 0.55 and rng.random() < 0.18:      # ghost kick before a downbeat
            times.append(t0 + 3.5 * config.SEC_PER_BEAT)
            gains.append(0.55)
    return times, gains


def render_bed(plan, key, scale, rng, sr=SR, chords=None, bass=None):
    """Render the full synthesized bed.

    Returns (mix, kick_times, atmos) where `atmos` is the pad+wash bus kept
    separate so effects can be applied to the atmosphere without touching the
    drums. It is NOT included in `mix`.
    """
    if chords is None:
        chords = config.CHORDS
    n = int(round(plan.total_bars * config.SEC_PER_BAR * sr))
    tail = int(6.0 * sr)                  # room for pad/reverb tails to ring out
    bus = {name: np.zeros((n + tail, 2), np.float32)
           for name in ("kick", "sub", "hats", "clap", "perc", "pad", "wash", "fx")}

    root = key_root_midi(key, scale)
    prog = progression(key, scale, chords=chords)
    bass = bass or bass_design(rng, scale, chords=chords)

    # --- pad + sub. With chords off this is one sustained drone on the root, held
    # in long segments; with --chords it steps through the progression every 4 bars.
    chord_bars = 4 if chords else config.DRONE_SEGMENT_BARS
    dur = chord_bars * config.SEC_PER_BAR

    # Work out every segment first, then render the distinct voicings concurrently:
    # pad_voice is a pure function of its arguments, so this is a clean fan-out and
    # the results are identical whatever the worker count. Brightness is quantized
    # to 0.1 so segments share cache entries — at full precision nearly every
    # segment was a cache miss and got its own render for an inaudible difference.
    seg_specs = []
    for start in range(0, plan.total_bars, chord_bars):
        seg_bars = plan.bars[start:start + chord_bars]
        if not any(b.pad for b in seg_bars):
            continue
        e = float(np.mean([b.energy for b in seg_bars]))
        deg, ivs = prog[(start // chord_bars) % len(prog)]
        seg_specs.append((start, seg_bars, e, deg, ivs,
                          round(0.75 + 0.9 * e, 1)))

    keys = list(dict.fromkeys((deg, ivs, bright)
                              for _s, _sb, _e, deg, ivs, bright in seg_specs))
    if keys:
        workers = parallel.worker_count("cpu", n_items=len(keys), per_worker_gb=0.5)
        # chords sit an octave above the root; the drone is anchored low instead
        base = (root + 12) if chords else drone_midi(key)
        voices = parallel.run(
            lambda k: pad_voice(base + k[0], k[1], dur, sr, seed=17 + k[0],
                                brightness=k[2]), keys, workers)
        pad_cache = dict(zip(keys, voices))
    else:
        pad_cache = {}

    for start, seg_bars, e, deg, ivs, bright in seg_specs:
        ck = (deg, ivs, bright)
        # INVERSE of energy, across a wide range (about -1 dB at rest to -11 dB at
        # full tilt): the drone fills space when little else is playing — in the
        # intro it IS the arrangement — and buries itself once the drums and loops
        # arrive. The old curve rose with energy, so it fought the mix exactly when
        # the mix got busy. A flat cut instead of this curve empties out the intro.
        audio.mix_at(bus["pad"], pad_cache[ck], start * config.SEC_PER_BAR * sr,
                     gain=0.9 - 0.62 * e)

        # sub plays this run's bassline. Density follows the energy curve: the
        # downbeat only when things are sparse, the full pattern once they aren't.
        pattern = config.BASS_PATTERNS[bass["pattern"]]
        for b in seg_bars:
            if not b.sub:
                continue
            t0 = b.index * config.SEC_PER_BAR
            g = 0.55 + 0.45 * b.energy
            if b.energy < 0.30:
                hits = ((0.0, min(3.8, config.BEATS_PER_BAR * 0.95), 0),)
            elif b.energy < 0.55:
                on_grid = tuple(h for h in pattern if abs(h[0] - round(h[0])) < 1e-6)
                hits = on_grid or pattern[:1]
            else:
                hits = pattern
            prev_hz = None
            for beat, length, di in hits:
                semis = bass["degrees"][min(di, len(bass["degrees"]) - 1)]
                hz = midi_hz(root + deg + bass["octave"] + semis)
                note = sub_note(hz, max(0.08, length * config.SEC_PER_BEAT), sr,
                                drive=bass["drive"], lpf_hz=bass["lpf_hz"],
                                harm2=bass["harm2"], attack=bass["attack"],
                                glide_from=prev_hz, glide_ms=bass["glide_ms"])
                audio.mix_at(bus["sub"], note,
                             (t0 + beat * config.SEC_PER_BEAT) * sr, gain=0.82 * g)
                prev_hz = hz

    # --- drums. Every voice's level tracks bar energy: that (not just adding more
    # loops) is what makes the nine minutes read as one long build. The kick also
    # swaps to a muffled variant while energy is low, so it arrives from under a
    # blanket and opens up later.
    k = kick(sr)
    k_muffled = audio.lpf(k, 320.0, sr, order=4) * 1.15
    energy_at = {b.index: b.energy for b in plan.bars}
    ktimes, kgains = _kick_times(plan, rng)
    for t, g in zip(ktimes, kgains):
        e = energy_at.get(int(t // config.SEC_PER_BAR), 0.5)
        audio.mix_at(bus["kick"], k_muffled if e < 0.30 else k,
                     t * sr, gain=g * (0.45 + 0.55 * e))

    hats = [hat(sr, seed=s, tau=tau, lo=lo, hi=hi)
            for s, tau, lo, hi in ((3, 0.022, 6500, 12500), (4, 0.035, 5200, 10500),
                                   (6, 0.014, 8000, 14000))]
    cl, pc, rm = clap(sr), perc(sr), rim(sr)
    for b in plan.bars:
        t0 = b.index * config.SEC_PER_BAR
        if b.hats:
            # offbeat 8ths always; 16ths creep in as energy rises
            for beat in range(config.BEATS_PER_BAR):
                audio.mix_at(bus["hats"], hats[rng.integers(0, 2)],
                             (t0 + (beat + 0.5) * config.SEC_PER_BEAT) * sr,
                             gain=0.35 + 0.65 * b.energy)
                if b.energy > 0.5 and rng.random() < 0.35 * b.energy:
                    audio.mix_at(bus["hats"], hats[2],
                                 (t0 + (beat + 0.25) * config.SEC_PER_BEAT) * sr,
                                 gain=0.2 + 0.3 * b.energy)
        if b.clap:
            for beat in (1, 3):
                audio.mix_at(bus["clap"], cl, (t0 + beat * config.SEC_PER_BEAT) * sr,
                             gain=0.4 + 0.6 * b.energy)
        if b.perc:
            for slot in (0.75, 2.25, 3.5):
                if rng.random() < 0.35 + 0.4 * b.energy:
                    audio.mix_at(bus["perc"], pc, (t0 + slot * config.SEC_PER_BEAT) * sr,
                                 gain=0.3 + 0.55 * b.energy)
            if rng.random() < 0.25 * b.energy:
                audio.mix_at(bus["perc"], rm, (t0 + 2.75 * config.SEC_PER_BEAT) * sr,
                             gain=0.25 + 0.45 * b.energy)

    # --- wash across the whole piece, energy-shaped per bar
    #
    # The envelope is one value per bar, interpolated between bar centres. It used
    # to be built as a step function and then smoothed with a savgol filter over a
    # ~1 s window, which is a direct convolution: 24M samples x 44k taps, about a
    # trillion operations, and it was 98% of the entire render (268 s of 273 s).
    # Interpolating from the 270 bar values is O(n), gives a continuous curve with
    # no steps to smooth in the first place, and runs in a fraction of a second.
    w = wash(plan.total_bars * config.SEC_PER_BAR + 2.0, sr)
    bar_vals = np.array([0.35 + 0.65 * (1.0 - b.energy) * (1.0 if b.wash else 0.25)
                         for b in plan.bars], dtype=np.float64)
    centres = (np.arange(len(bar_vals)) + 0.5) * config.SEC_PER_BAR * sr
    env = np.interp(np.arange(w.shape[0]), centres, bar_vals).astype(np.float32)
    audio.mix_at(bus["wash"], w * env[:, None], 0)

    # --- risers and impacts at the section joins the plan asked for
    for ev in plan.fx_events:
        if ev["kind"] == "riser":
            r = riser(ev["bars"] * config.SEC_PER_BAR, sr, seed=29 + ev["bar"],
                      tone_midi=root + 12)
            audio.mix_at(bus["fx"], r, ev["bar"] * config.SEC_PER_BAR * sr, gain=ev.get("gain", 0.8))
        elif ev["kind"] == "impact":
            audio.mix_at(bus["fx"], impact(sr, seed=31 + ev["bar"]),
                         ev["bar"] * config.SEC_PER_BAR * sr, gain=ev.get("gain", 0.8))

    # --- bus levels: kick and sub own the low end, pad/wash sit back
    levels = dict(config.BED_LEVELS)
    levels.update({"pad": config.DRONE_LEVEL_DB, "wash": -20.0})
    # High-passed ABOVE the sub's fundamental range: the drone, the sub and the kick
    # were all inside one octave, which is textbook low-end mud. The drone keeps its
    # harmonics as low-mid warmth and stops competing for the bottom.
    bus["pad"] = audio.hpf(bus["pad"], config.DRONE_HPF_HZ, sr, order=2)
    bus["pad"] = audio.lpf(bus["pad"], config.DRONE_TOP_HZ * 1.6, sr, order=2)
    bus["wash"] = audio.hpf(bus["wash"], 180.0, sr, order=2)
    # duck hard on every kick — a sustained drone masks the low-mid otherwise
    bus["pad"] = audio.duck(bus["pad"], ktimes, depth_db=6.0, release_ms=220.0, sr=sr)
    # and duck the sub too, so the kick and the bass never occupy the same instant
    bus["sub"] = audio.duck(bus["sub"], ktimes, depth_db=config.SUB_DUCK_DB,
                            attack_ms=5.0, release_ms=120.0, sr=sr)

    # atmosphere (drone + wash) comes back separately so a delight can process it
    atmos = (audio.gain_db(bus["pad"], levels["pad"]) +
             audio.gain_db(bus["wash"], levels["wash"]))
    mix = np.zeros((n + tail, 2), np.float32)
    for name in ("kick", "sub", "hats", "clap", "perc", "fx"):
        mix += audio.gain_db(bus[name], levels[name])
    return mix, ktimes, atmos
