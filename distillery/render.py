"""Render a Plan to audio: synthesized bed + sample-loop bus -> master.

Loop bus treatment, per event:
  reverse (only at low energy) -> tile to the event length -> energy-mapped
  lowpass -> highpass at 140 Hz so the synth kick keeps the low end -> reverb
  (wet when distant, dry when present) -> pan -> gain -> one/two-beat fades.

Then the whole loop bus is ducked against every kick onset, which is what makes
borrowed drums sit under a four-on-the-floor instead of fighting it.
"""
import math
import threading
import time

import numpy as np

from . import arrange, audio, config, fx, parallel, techno

SR = config.SR


_CACHE_LOCK = threading.Lock()


def _loop_audio(path, cache):
    """Read (and cache) a loop wav. Locked because several event workers can want
    the same loop at once — the arrangement reuses loops heavily."""
    with _CACHE_LOCK:
        hit = cache.get(path)
    if hit is None:
        hit = audio.read(path)
        with _CACHE_LOCK:
            cache[path] = hit
    return hit


def _render_texture_event(ev, cache):
    """Texture from the "other" stem: transposed into key, then pushed back.

    The pitch shift is what makes it harmonically safe (the arranger picked the
    interval); everything after it is about keeping this a texture rather than a
    tune — dark filtering, long reverb, slow fades, low level.
    """
    clip = _loop_audio(ev.wav, cache)
    if ev.reverse:
        clip = clip[::-1].copy()
    n = int(round(ev.bars * config.SEC_PER_BAR * SR))
    x = audio.tile_to(clip, n)
    if abs(ev.semitones) > 0.01 and fx.available():
        x = fx.apply([fx.pb.PitchShift(semitones=float(ev.semitones))], x, SR)[:n]
    # 300 Hz, not 200: the texture measured DARK (654 Hz centroid vs the mix's 1934),
    # so its energy was piling into the low-mids where the kick, sub and drone live —
    # which is what made it feel congested and present rather than merely loud.
    x = audio.hpf(x, 300.0, SR, order=2)
    x = audio.lpf(x, ev.lpf_hz, SR, order=4)
    x = audio.high_shelf(x, config.TEXTURE_AIR_HZ, config.TEXTURE_AIR_DB, SR)
    x = audio.reverb(x, wet=ev.reverb_wet, seconds=3.4, damp_hz=4200.0,
                     predelay_ms=30.0, seed=23)[:n + int(1.2 * SR)]
    x = audio.pan(x, ev.pan)
    x = audio.gain_db(x, ev.gain_db)
    fade_s = ev.fade_beats * config.SEC_PER_BEAT
    x = audio.fade(x, in_s=fade_s * 0.5, out_s=fade_s, sr=SR, shape="sqrt")

    if ev.delay_beats and ev.delay_repeats:
        # Throw the event's LAST BEAT, not the whole loop: echoing the entire clip
        # just smears it into mush. This is the tape-delay throw — one hit sent
        # ringing past the end of the event.
        beat_n = int(round(config.SEC_PER_BEAT * SR))
        grab = x[max(0, n - beat_n):n]
        if grab.shape[0] > 64:
            throw = audio.echo(audio.gain_db(grab, config.TEXTURE_DELAY_GAIN_DB),
                               ev.delay_beats * config.SEC_PER_BEAT,
                               repeats=ev.delay_repeats,
                               decay_db=config.TEXTURE_DELAY_DECAY_DB,
                               lp_start=5200.0, lp_factor=0.88, pan_amount=0.45,
                               sr=SR, dry=False)
            start = max(0, n - beat_n)
            total = max(x.shape[0], start + throw.shape[0])
            out = np.zeros((total, 2), np.float32)
            out[:x.shape[0]] = x
            audio.mix_at(out, throw, start)
            x = out
    return x


def _render_event(ev, cache):
    """Turn one LoopEvent into a stereo clip ready to be mixed in."""
    if getattr(ev, "kind", "drum") == "texture":
        return _render_texture_event(ev, cache)
    clip = _loop_audio(ev.wav, cache)
    if ev.reverse:
        clip = clip[::-1].copy()
    n = int(round(ev.bars * config.SEC_PER_BAR * SR))
    x = audio.tile_to(clip, n)
    if ev.lpf_hz < 17000:
        # resonant ladder filter (pedalboard) rather than a plain butterworth —
        # the resonance is most of the character in a filtered-build loop
        x = fx.event_filter(x, ev.lpf_hz, ev.energy, SR)
    x = audio.hpf(x, 140.0, SR, order=2)
    if ev.reverb_wet > 0.10:
        x = fx.ambient_smear(x, SR, mix=ev.reverb_wet)[:n + int(0.6 * SR)]
    x = audio.pan(x, ev.pan)
    x = audio.gain_db(x, ev.gain_db)
    # fast in, slower out: the attack is the snap, the release can breathe
    fade_s = ev.fade_beats * config.SEC_PER_BEAT
    return audio.fade(x, in_s=fade_s * 0.3, out_s=fade_s, sr=SR, shape="sqrt")


# Loop-bus level swing from energy 0 to energy 1. 17 dB buried the borrowed drums in
# the quieter sections; 12 dB still reads as a build (the bed's own voices scale with
# energy too) while keeping the album's drums audible throughout.
LOOP_MACRO_RANGE_DB = 12.0


def macro_envelope(plan, n, range_db=LOOP_MACRO_RANGE_DB, sr=SR):
    """Per-sample gain from the plan's energy curve, smoothed over ~a beat.

    One envelope, derived from the same curve the arranger used, carries the whole
    dynamic build. Smoothing matters: stepping the gain at bar lines would click.
    """
    from scipy.ndimage import uniform_filter1d
    g = np.ones(n, dtype=np.float32)
    for b in plan.bars:
        s = int(b.index * config.SEC_PER_BAR * sr)
        e = min(n, int((b.index + 1) * config.SEC_PER_BAR * sr))
        if s >= n:
            break
        g[s:e] = audio.db2lin(-range_db * (1.0 - b.energy))
    if n > int(config.SEC_PER_BAR * sr):
        g[int(plan.total_bars * config.SEC_PER_BAR * sr):] = g[
            min(n - 1, int(plan.total_bars * config.SEC_PER_BAR * sr) - 1)]
    return uniform_filter1d(g, size=max(3, int(0.5 * sr)), mode="nearest")


def render(plan, meta=None, verbose=True, return_stems=False):
    """Render the plan. Returns (master, info dict)."""
    rng = np.random.default_rng(plan.seed)
    total_n = int(round(plan.total_bars * config.SEC_PER_BAR * SR))
    tail_n = int(6.0 * SR)

    t0 = time.time()
    if verbose:
        print(f"  synthesizing ambient techno bed "
              f"({'chords' if plan.chords else 'drone, no chords'}) ...", flush=True)
    bed, kick_times, atmos = techno.render_bed(plan, plan.key, plan.key_scale, rng,
                                              chords=plan.chords, bass=plan.bass)
    bed = audio.pad_to(bed, total_n + tail_n)
    atmos = audio.pad_to(atmos, total_n + tail_n)
    t_bed = time.time() - t0

    # Each loop event is an independent chain (filter -> reverb -> pan -> fade), so
    # they render concurrently; only the mixing is serial. Results arrive in event
    # order, which keeps the output bit-identical to a single-worker run.
    t0 = time.time()
    workers = parallel.worker_count("cpu", n_items=len(plan.loop_events),
                                   per_worker_gb=0.4)
    if verbose:
        print(f"  placing {len(plan.loop_events)} loop events "
              f"({workers} worker{'s' if workers != 1 else ''}) ...", flush=True)
    loop_bus = np.zeros((total_n + tail_n, 2), np.float32)
    texture_bus = np.zeros((total_n + tail_n, 2), np.float32)
    cache = {}
    done = 0
    for ev, clip in parallel.imap_ordered(lambda e: _render_event(e, cache),
                                          plan.loop_events, workers):
        bus = texture_bus if getattr(ev, "kind", "drum") == "texture" else loop_bus
        audio.mix_at(bus, clip, ev.bar * config.SEC_PER_BAR * SR)
        done += 1
        if verbose and done % 40 == 0:
            print(f"    {done}/{len(plan.loop_events)}", flush=True)
    t_events = time.time() - t0

    # sidechain the borrowed drums against the synth kick
    loop_bus = audio.duck(loop_bus, kick_times, depth_db=4.0, attack_ms=8.0,
                          release_ms=config.LOOP_DUCK_RELEASE_MS, sr=SR)
    # attack enhancement before the soft clip, and a gentler clip: tanh at 1.15 was
    # rounding off the very transients we want to hear
    loop_bus = audio.transient_shape(loop_bus, amount=config.LOOP_TRANSIENT, sr=SR)
    loop_bus = audio.soft_clip(loop_bus, drive=1.05) * 0.90
    # the build: one smooth energy-derived envelope across the whole loop bus
    loop_bus *= macro_envelope(plan, loop_bus.shape[0])[:, None]

    n_tex = sum(1 for e in plan.loop_events if getattr(e, "kind", "drum") == "texture")
    if n_tex:
        texture_bus = audio.duck(texture_bus, kick_times, depth_db=3.0,
                                 release_ms=200.0, sr=SR)
        texture_bus = audio.gain_db(texture_bus, config.TEXTURE_LEVEL_DB)

    # this run's delights (pedalboard), placed at real bars by the arranger
    fx_log = []
    if plan.delights and fx.available():
        for d in plan.delights:
            try:
                fx_log.append(fx.apply_delight(d, loop_bus, atmos, SR))
            except Exception as e:        # noqa: BLE001 - never lose a mix to an fx
                fx_log.append(f"!! {d.get('name')}: {type(e).__name__}: {e}")
        if verbose:
            for line in fx_log:
                print(f"  fx: {line}", flush=True)

    t0 = time.time()
    if verbose:
        print("  mixing master ...", flush=True)
    # each bus's ACTUAL contribution, so what we measure is what is in the mix
    contrib = {"bed": audio.gain_db(bed, -3.0),
               "atmos": audio.gain_db(atmos, 0.0),
               "loops": audio.gain_db(loop_bus, config.LOOP_LEVEL_DB),
               "texture": texture_bus}
    pre = contrib["bed"] + contrib["atmos"] + contrib["loops"] + contrib["texture"]
    # tilt the whole mix brighter before limiting
    pre = audio.high_shelf(pre, config.MASTER_TILT_HIGH_HZ, config.MASTER_TILT_HIGH_DB)
    pre = audio.low_shelf(pre, config.MASTER_TILT_LOW_HZ, config.MASTER_TILT_LOW_DB)

    # Limit RELATIVE to the mix's own peak, shaving only the top couple of dB of
    # transients. A fixed ceiling pinned every section to it and flattened the
    # build — each section came out peaking at exactly the ceiling, which is the
    # opposite of what a nine-minute climb needs.
    ceiling = audio.peak_dbfs(pre) - config.MASTER_SHAVE_DB
    master = audio.limiter(pre, ceiling_db=ceiling, lookahead_ms=2.5, release_ms=45.0)
    gr_db = round(audio.peak_dbfs(pre) - audio.peak_dbfs(master), 2)
    master = master[:total_n]
    master = audio.fade(master, in_s=0.05, out_s=2.5, sr=SR, shape="sqrt")
    # the final normalization is a single scalar; stems must carry it too or any
    # stem-vs-master comparison is off by however much the mix was scaled
    pre_norm_peak = audio.peak_dbfs(master)
    master = audio.peak_normalize(master, config.MASTER_PEAK_DBFS)
    norm_db = config.MASTER_PEAK_DBFS - pre_norm_peak

    info = {
        "limiter_gr_db": gr_db,
        "duration_s": round(master.shape[0] / SR, 3),
        "bars": plan.total_bars, "bpm": config.TARGET_BPM,
        "loop_events": len(plan.loop_events),
        "unique_loops": len({e.loop_id for e in plan.loop_events}),
        "texture_events": n_tex,
        "songs_used": len({e.track_no for e in plan.loop_events}),
        "same_song_overlaps": len(arrange.check_overlaps(plan)),
        "same_song_rule": plan.same_song_rule,
        "self_overlaps": len(arrange.check_self_overlaps(plan)),
        "peak_dbfs": round(audio.peak_dbfs(master), 2),
        "rms_dbfs": round(audio.dbfs(master), 2),
        "seed": plan.seed, "key": plan.key, "key_scale": plan.key_scale,
        "chords": plan.chords,
        "pedalboard": fx.version() if fx.available() else None,
        "delights": fx_log or [d["name"] for d in plan.delights],
        "workers": workers,
        "timings_s": {"bed": round(t_bed, 1), "events": round(t_events, 1),
                      "master": round(time.time() - t0, 1)},
    }
    if verbose:
        print(f"  timings: bed {t_bed:.1f}s  events {t_events:.1f}s  "
              f"master {info['timings_s']['master']:.1f}s", flush=True)
    if return_stems:
        # post-gain and post-normalization: directly comparable to `master`
        info["stems"] = {k: audio.gain_db(v[:total_n], norm_db)
                         for k, v in contrib.items()}
    return master, info


def section_levels(master, plan):
    """Per-section RMS — a quick numeric read on whether the build actually builds."""
    out = []
    for s in plan.sections:
        s0 = int(s["start_bar"] * config.SEC_PER_BAR * SR)
        s1 = int((s["start_bar"] + s["bars"]) * config.SEC_PER_BAR * SR)
        seg = master[s0:min(s1, master.shape[0])]
        out.append((s["name"], round(audio.dbfs(seg), 2)))
    return out
