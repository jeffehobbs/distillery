"""Audio primitives: IO, resampling, filters, envelopes, reverb, mixing.

Everything is float32 stereo shaped (n, 2) at config.SR unless noted. No pydub /
pedalboard dependency — numpy + scipy + ffmpeg cover all of it, which keeps the
project runnable inside the existing essentia-explorer venv.
"""
import math
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from scipy import signal

from . import config

SR = config.SR


# ---------------------------------------------------------------- io

def read(path, sr=SR, stereo=True):
    """Read any wav/flac soundfile can open, resampling if needed."""
    a, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    if file_sr != sr:
        a = resample_to(a, file_sr, sr)
    if stereo:
        a = to_stereo(a)
    return a


def decode(path, sr=SR, start=None, dur=None):
    """ffmpeg-decode (a slice of) any media file to float32 stereo."""
    cmd = ["ffmpeg", "-y", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.4f}"]
    cmd += ["-i", str(path)]
    if dur is not None:
        cmd += ["-t", f"{dur:.4f}"]
    with tempfile.NamedTemporaryFile(suffix=".wav", dir=str(config.TMP_DIR),
                                     delete=True) as tmp:
        cmd += ["-ar", str(sr), "-ac", "2", "-c:a", "pcm_f32le", "-f", "wav", tmp.name]
        subprocess.run(cmd, check=True)
        a, _ = sf.read(tmp.name, dtype="float32", always_2d=True)
    return to_stereo(a)


def write(path, a, sr=SR, subtype="PCM_24"):
    sf.write(str(path), np.asarray(a, dtype=np.float32), sr, subtype=subtype)


def to_mp3(wav_path, mp3_path, bitrate="320k"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
                    "-b:a", bitrate, str(mp3_path)], check=True)


# ---------------------------------------------------------------- shape / level

def to_stereo(a):
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 1:
        return np.stack([a, a], axis=1)
    if a.shape[1] == 1:
        return np.repeat(a, 2, axis=1)
    return a[:, :2]


def mono(a):
    a = np.asarray(a, dtype=np.float32)
    return a if a.ndim == 1 else a.mean(axis=1)


def dbfs(a):
    m = mono(a)
    if not m.size:
        return -np.inf
    r = float(np.sqrt(np.mean(m.astype(np.float64) ** 2)))
    return 20 * math.log10(r) if r > 1e-12 else -np.inf


def peak_dbfs(a):
    p = float(np.max(np.abs(a))) if np.size(a) else 0.0
    return 20 * math.log10(p) if p > 1e-12 else -np.inf


def db2lin(db):
    return float(10 ** (db / 20.0))


def gain_db(a, db):
    return (np.asarray(a, dtype=np.float32) * db2lin(db)).astype(np.float32)


def normalize_rms(a, target_dbfs):
    cur = dbfs(a)
    if not np.isfinite(cur):
        return np.asarray(a, dtype=np.float32)
    return gain_db(a, target_dbfs - cur)


def peak_normalize(a, target_dbfs=config.MASTER_PEAK_DBFS):
    p = peak_dbfs(a)
    if not np.isfinite(p):
        return np.asarray(a, dtype=np.float32)
    return gain_db(a, target_dbfs - p)


def soft_clip(a, drive=1.0):
    return np.tanh(np.asarray(a, dtype=np.float32) * drive).astype(np.float32)


def pan(a, position):
    """Constant-power pan. position -1 = hard left, +1 = hard right."""
    a = to_stereo(a).copy()
    p = float(np.clip(position, -1.0, 1.0))
    ang = (p + 1.0) * (math.pi / 4.0)
    a[:, 0] *= math.cos(ang)
    a[:, 1] *= math.sin(ang)
    return a * math.sqrt(2.0)


def widen(a, amount=0.25):
    """Mid/side widening."""
    a = to_stereo(a)
    m = (a[:, 0] + a[:, 1]) * 0.5
    s = (a[:, 0] - a[:, 1]) * 0.5 * (1.0 + amount * 3.0)
    return np.stack([m + s, m - s], axis=1).astype(np.float32)


# ---------------------------------------------------------------- time

def resample_to(a, sr_in, sr_out):
    if sr_in == sr_out:
        return np.asarray(a, dtype=np.float32)
    a = np.asarray(a, dtype=np.float32)
    g = math.gcd(int(sr_in), int(sr_out))
    up, down = int(sr_out // g), int(sr_in // g)
    out = signal.resample_poly(a, up, down, axis=0)
    return out.astype(np.float32)


def speed(a, rate):
    """Change playback speed by `rate` (2.0 = twice as fast / half as long).

    Plain resampling — pitch moves with tempo. This is deliberate for drums:
    phase-vocoder stretching smears transients ("whooshy"), and the pitch shift
    of a modest tempo correction is inaudible on percussion.
    """
    a = np.asarray(a, dtype=np.float32)
    if abs(rate - 1.0) < 1e-9 or a.shape[0] < 2:
        return a
    n_out = max(2, int(round(a.shape[0] / rate)))
    src = np.linspace(0.0, a.shape[0] - 1.0, n_out, dtype=np.float64)
    if a.ndim == 1:
        return np.interp(src, np.arange(a.shape[0]), a).astype(np.float32)
    cols = [np.interp(src, np.arange(a.shape[0]), a[:, c]) for c in range(a.shape[1])]
    return np.stack(cols, axis=1).astype(np.float32)


def fit_length(a, n):
    """Resample so the clip is exactly n samples long (sub-percent nudge)."""
    a = np.asarray(a, dtype=np.float32)
    if a.shape[0] == n:
        return a
    return speed(a, a.shape[0] / float(n))[:n] if a.shape[0] > 1 else np.zeros((n, 2), np.float32)


def tile_to(a, n):
    """Repeat a loop to fill n samples (whole-loop repeats, exact tail cut)."""
    a = to_stereo(a)
    if a.shape[0] == 0:
        return np.zeros((n, 2), np.float32)
    reps = int(math.ceil(n / a.shape[0]))
    return np.tile(a, (reps, 1))[:n]


def pad_to(a, n):
    a = to_stereo(a)
    if a.shape[0] >= n:
        return a[:n]
    return np.vstack([a, np.zeros((n - a.shape[0], 2), np.float32)])


# ---------------------------------------------------------------- envelopes

def fade(a, in_s=0.0, out_s=0.0, sr=SR, shape="lin"):
    a = np.asarray(a, dtype=np.float32).copy()
    n = a.shape[0]
    ni = min(int(in_s * sr), n)
    no = min(int(out_s * sr), n - ni if n - ni > 0 else 0)
    if ni > 0:
        e = np.linspace(0.0, 1.0, ni, dtype=np.float32)
        if shape == "sqrt":
            e = np.sqrt(e)
        a[:ni] *= e[:, None] if a.ndim > 1 else e
    if no > 0:
        e = np.linspace(1.0, 0.0, no, dtype=np.float32)
        if shape == "sqrt":
            e = np.sqrt(e)
        a[n - no:] *= e[:, None] if a.ndim > 1 else e
    return a


def xfade_wrap(a, n_loop, n_x):
    """Equal-power seamless-wrap crossfade.

    `a` holds the loop plus n_x samples of post-roll; the post-roll is faded over
    the loop's head so the loop's end flows into its own start on repeat. This is
    the essentia-explorer fix — an MFCC seam score can be perfect and the splice
    still click.
    """
    a = to_stereo(a)
    n_x = int(min(n_x, n_loop // 4, max(0, a.shape[0] - n_loop)))
    if a.shape[0] < n_loop or n_x < 1:
        return pad_to(a, n_loop)
    body = a[:n_loop].copy()
    post = a[n_loop:n_loop + n_x]
    fin = np.sqrt(np.linspace(0.0, 1.0, n_x, dtype=np.float32))[:, None]
    fout = np.sqrt(np.linspace(1.0, 0.0, n_x, dtype=np.float32))[:, None]
    body[:n_x] = body[:n_x] * fin + post * fout
    return body


def env_adsr(n, sr=SR, a=0.01, d=0.1, s=0.7, r=0.3):
    """Simple ADSR gain envelope of length n."""
    na, nd = int(a * sr), int(d * sr)
    nr = int(r * sr)
    ns = max(0, n - na - nd - nr)
    parts = [np.linspace(0, 1, na, dtype=np.float32) if na else np.empty(0, np.float32),
             np.linspace(1, s, nd, dtype=np.float32) if nd else np.empty(0, np.float32),
             np.full(ns, s, dtype=np.float32),
             np.linspace(s, 0, nr, dtype=np.float32) if nr else np.empty(0, np.float32)]
    e = np.concatenate(parts)
    return e[:n] if e.size >= n else np.concatenate([e, np.zeros(n - e.size, np.float32)])


def exp_decay(n, sr=SR, tau=0.1):
    t = np.arange(n, dtype=np.float32) / sr
    return np.exp(-t / max(1e-4, tau)).astype(np.float32)


# ---------------------------------------------------------------- filters

def _sos(kind, cutoff, sr=SR, order=4):
    ny = sr * 0.5
    if kind in ("bp", "bandpass"):
        lo, hi = cutoff
        wn = [max(1e-4, lo / ny), min(0.999, hi / ny)]
        return signal.butter(order, wn, btype="bandpass", output="sos")
    wn = float(np.clip(cutoff / ny, 1e-4, 0.999))
    btype = "lowpass" if kind in ("lp", "lowpass") else "highpass"
    return signal.butter(order, wn, btype=btype, output="sos")


def lpf(a, cutoff, sr=SR, order=4):
    return signal.sosfilt(_sos("lp", cutoff, sr, order), np.asarray(a, np.float32),
                          axis=0).astype(np.float32)


def hpf(a, cutoff, sr=SR, order=4):
    return signal.sosfilt(_sos("hp", cutoff, sr, order), np.asarray(a, np.float32),
                          axis=0).astype(np.float32)


def bpf(a, lo, hi, sr=SR, order=4):
    return signal.sosfilt(_sos("bp", (lo, hi), sr, order), np.asarray(a, np.float32),
                          axis=0).astype(np.float32)


def lpf_sweep(a, f_start, f_end, sr=SR, order=4, chunks=96, curve="log"):
    """Time-varying lowpass. Filter state is carried across chunks so the sweep
    doesn't click at chunk boundaries."""
    a = to_stereo(a)
    n = a.shape[0]
    if n < chunks * 4:
        return lpf(a, min(f_start, f_end), sr, order)
    if curve == "log":
        freqs = np.geomspace(max(20.0, f_start), max(20.0, f_end), chunks)
    else:
        freqs = np.linspace(f_start, f_end, chunks)
    bounds = np.linspace(0, n, chunks + 1).astype(int)
    out = np.empty_like(a)
    zi = None
    for i, f in enumerate(freqs):
        s, e = bounds[i], bounds[i + 1]
        sos = _sos("lp", f, sr, order)
        if zi is None:
            zi = np.zeros((sos.shape[0], 2, a.shape[1]), dtype=np.float32)
        seg, zi = signal.sosfilt(sos, a[s:e], axis=0, zi=zi)
        out[s:e] = seg
    return out.astype(np.float32)


def transient_shape(a, amount=0.6, fast_ms=3.0, slow_ms=55.0, sr=SR, max_boost_db=6.0,
                    preserve_level=True):
    """Attack enhancement: boost where a fast envelope outruns a slow one.

    This is what "snappy" means mechanically — the leading edge of each hit gets
    gain, the sustain does not. Both followers are O(n) box filters, so it is cheap
    even on a nine-minute buffer.
    """
    from scipy.ndimage import uniform_filter1d
    a = to_stereo(a)
    if amount <= 0 or a.shape[0] < 16:
        return a
    env = np.abs(mono(a)).astype(np.float32)
    fast = uniform_filter1d(env, size=max(2, int(fast_ms * sr / 1000)), mode="nearest")
    slow = uniform_filter1d(env, size=max(4, int(slow_ms * sr / 1000)), mode="nearest")
    excess = np.maximum(0.0, fast - slow) / np.maximum(slow, 1e-5)
    g = 1.0 + amount * np.minimum(excess, db2lin(max_boost_db) - 1.0)
    g = np.minimum(g, db2lin(max_boost_db)).astype(np.float32)
    out = (a * g[:, None]).astype(np.float32)
    if preserve_level:
        # The shaper only ever boosts, so left alone it raises the whole bus and the
        # master's peak-normalization gives the level straight back. Matching RMS
        # keeps this a change of SHAPE rather than of loudness.
        before, after = dbfs(a), dbfs(out)
        if np.isfinite(before) and np.isfinite(after):
            out = gain_db(out, before - after)
    return out


def high_shelf(a, cutoff, gain_db, sr=SR, order=2):
    """Boost/cut everything above `cutoff` by gain_db (parallel-highpass shelf)."""
    a = to_stereo(a)
    if abs(gain_db) < 0.01:
        return a
    g = db2lin(gain_db) - 1.0
    return (a + g * hpf(a, cutoff, sr, order=order)).astype(np.float32)


def low_shelf(a, cutoff, gain_db, sr=SR, order=2):
    """Boost/cut everything below `cutoff` by gain_db (parallel-lowpass shelf)."""
    a = to_stereo(a)
    if abs(gain_db) < 0.01:
        return a
    g = db2lin(gain_db) - 1.0
    return (a + g * lpf(a, cutoff, sr, order=order)).astype(np.float32)


# ---------------------------------------------------------------- reverb

_IR_CACHE = {}


def reverb_ir(seconds=2.6, sr=SR, damp_hz=6500.0, predelay_ms=18.0, seed=7):
    """Decaying-noise impulse response with early reflections, cached."""
    key = (round(seconds, 3), sr, round(damp_hz, 1), round(predelay_ms, 2), seed)
    if key in _IR_CACHE:
        return _IR_CACHE[key]
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32) / sr
    decay = np.exp(-t * (6.9 / max(0.05, seconds)))
    ir = rng.normal(0.0, 1.0, size=(n, 2)).astype(np.float32) * decay[:, None]
    # early reflections give it a room rather than a noise cloud
    for d_ms, g in ((11.0, 0.5), (17.0, -0.4), (29.0, 0.32), (41.0, -0.24), (57.0, 0.18)):
        d = int(d_ms * sr / 1000.0)
        if d < n:
            ir[d:] += ir[:n - d] * g * 0.35
    ir = lpf(ir, damp_hz, sr, order=2)
    ir = hpf(ir, 90.0, sr, order=2)
    pre = int(predelay_ms * sr / 1000.0)
    if pre:
        ir = np.vstack([np.zeros((pre, 2), np.float32), ir])
    ir /= max(1e-9, float(np.max(np.abs(ir))))
    _IR_CACHE[key] = ir
    return ir


def reverb(a, wet=0.3, seconds=2.6, sr=SR, damp_hz=6500.0, predelay_ms=18.0,
           seed=7, wet_only=False):
    """Convolution reverb via overlap-add (cheap even on 9-minute buffers)."""
    a = to_stereo(a)
    ir = reverb_ir(seconds, sr, damp_hz, predelay_ms, seed)
    w = np.stack([signal.oaconvolve(a[:, c], ir[:, c], mode="full")
                  for c in range(2)], axis=1).astype(np.float32)
    w *= 0.35
    if wet_only:
        return w
    n = max(a.shape[0], w.shape[0])
    return (pad_to(a, n) * (1.0 - wet) + pad_to(w, n) * wet).astype(np.float32)


# ---------------------------------------------------------------- ducking / delay

def duck(a, onsets_s, depth_db=4.0, attack_ms=8.0, release_ms=180.0, sr=SR):
    """Sidechain-style gain envelope: dip at each onset time, ease back.

    Cheaper and more predictable than a real compressor, and it is what makes a
    loop bus sit under a four-on-the-floor kick.
    """
    a = to_stereo(a)
    n = a.shape[0]
    g = np.ones(n, dtype=np.float32)
    na, nr = max(1, int(attack_ms * sr / 1000)), max(1, int(release_ms * sr / 1000))
    floor = db2lin(-abs(depth_db))
    down = np.linspace(1.0, floor, na, dtype=np.float32)
    up = np.linspace(floor, 1.0, nr, dtype=np.float32)
    shape = np.concatenate([down, up])
    for t in onsets_s:
        i = int(t * sr)
        if i < 0 or i >= n:
            continue
        j = min(n, i + shape.size)
        g[i:j] = np.minimum(g[i:j], shape[:j - i])
    return (a * g[:, None]).astype(np.float32)


def echo(a, delay_s, repeats=6, decay_db=-4.5, lp_start=6000.0, lp_factor=0.86,
         pan_amount=0.35, sr=SR, dry=True):
    """Ping-ponged, progressively lowpassed echo tail (dub throw).

    `dry=False` returns the repeats only, which is what a throw wants: the source
    is already in the mix, so including the dry copy would double it.
    """
    a = to_stereo(a)
    d = int(delay_s * sr)
    out = np.zeros((a.shape[0] + d * (repeats + 1), 2), np.float32)
    if dry:
        out[:a.shape[0]] += a
    tap = a
    for k in range(1, repeats + 1):
        cut = max(1000.0, lp_start * (lp_factor ** k))
        tap = lpf(tap, cut, sr, order=2)
        seg = pan(gain_db(tap, decay_db * k), pan_amount * (1 if k % 2 else -1))
        s = d * k
        out[s:s + seg.shape[0]] += seg
    return out


def limiter(a, ceiling_db=-0.6, lookahead_ms=3.0, release_ms=60.0, sr=SR):
    """Lookahead peak limiter so the climax can be dense without clipping.

    Fully vectorized: a max-filter over |x| gives the lookahead/hold envelope,
    then the required gain reduction is smoothed with a moving average so it
    eases in and out instead of stepping.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    a = to_stereo(a)
    if a.shape[0] == 0:
        return a
    ceil = db2lin(ceiling_db)
    peak = np.max(np.abs(a), axis=1)
    la = max(1, int(lookahead_ms * sr / 1000))
    rel = max(la, int(release_ms * sr / 1000))
    # hold the peak across the release window (and look ahead by `la`)
    env = maximum_filter1d(peak, size=2 * rel + 1, mode="nearest")
    need = np.where(env > ceil, ceil / np.maximum(env, 1e-9), 1.0).astype(np.float32)
    g = uniform_filter1d(need, size=2 * la + 1, mode="nearest")
    g = np.minimum(g, need)          # never let smoothing undo the reduction
    return (a * g[:, None]).astype(np.float32)


def mix_at(bus, clip, start_sample, gain=1.0):
    """Add clip into bus at a sample offset, clipping to the bus length."""
    clip = to_stereo(clip)
    n = bus.shape[0]
    s = int(start_sample)
    if s >= n or clip.shape[0] == 0:
        return
    if s < 0:
        clip = clip[-s:]
        s = 0
    e = min(n, s + clip.shape[0])
    bus[s:e] += clip[:e - s] * gain
