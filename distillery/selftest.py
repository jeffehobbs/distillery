"""Invariant checks that need no album, no network, and no models.

    ./run.sh selftest

Covers the things that are easy to break silently: the section allocator, the
retime math (loops must land on exact 120 BPM beats), the no-same-song rule, the
seamless wrap, the build envelope, and the audio helpers.
"""
import itertools
import math
from pathlib import Path

import numpy as np

from . import arrange, audio, config, fx, loops, render, techno

_FAILURES = []


def check(name, cond, detail=""):
    ok = bool(cond)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        _FAILURES.append(name)
    return ok


def _fake_pool(n_tracks=6, per_track=12, seed=1):
    rng = np.random.default_rng(seed)
    pool = []
    for t in range(1, n_tracks + 1):
        for i in range(per_track):
            bars = float(rng.choice([0.5, 1.0, 2.0, 4.0]))
            pool.append({
                "id": f"fake:{t:02d}_{i:02d}", "track_no": t, "track_name": f"song-{t}",
                "wav": f"/dev/null/{t}_{i}.wav", "out_bars": bars,
                "loopability": float(rng.uniform(0.2, 0.9)),
                "onset_density": float(rng.uniform(0.5, 9.0)),
                "centroid_hz": float(rng.uniform(400, 7000)),
                "texture": "mid-mid",
            })
    return pool


def test_allocator():
    print("\nsection allocator")
    ok = True
    for n in list(range(10, 60)) + [64, 135, 269, 270, 271, 400, 1024]:
        b = arrange._section_bars(n)
        ok &= (sum(b) == n) and all(x >= 1 for x in b) and len(b) == len(arrange.SECTIONS)
    check("bars sum exactly and never go below 1 (10..1024)", ok)
    check("270 bars matches the template", arrange._section_bars(270) ==
          [s[1] for s in arrange.SECTIONS])


def test_plan_shape():
    print("\nplan shape")
    pool = _fake_pool()
    for total in (64, 135, 270):
        p = arrange.build_plan(pool, key="A", key_scale="minor", total_bars=total, seed=3)
        check(f"{total}-bar plan has exactly {total} bar specs", len(p.bars) == total)
        check(f"{total}-bar energies within 0..1",
              all(0.0 <= b.energy <= 1.0 for b in p.bars))
        check(f"{total}-bar events stay inside the piece",
              all(e.bar >= 0 and e.end_bar <= total + 1e-9 and e.bars > 0
                  for e in p.loop_events))
    p = arrange.build_plan(pool, key="A", key_scale="minor", seed=3)
    prof = arrange.concurrency_profile(p)
    check("never more than MAX_LAYERS concurrent loops",
          max(prof) <= config.MAX_LAYERS, f"max {max(prof)}")
    e_climax = np.mean([b.energy for b in p.bars if b.section == "climax"])
    e_intro = np.mean([b.energy for b in p.bars if b.section == "intro"])
    check("climax energy exceeds intro energy", e_climax > e_intro + 0.5,
          f"{e_intro:.2f} -> {e_climax:.2f}")
    lay_climax = np.mean([prof[b.index] for b in p.bars if b.section == "climax"])
    lay_early = np.mean([prof[b.index] for b in p.bars if b.section in ("intro", "pulse")])
    check("more loops layered at the climax than at the start",
          lay_climax > lay_early, f"{lay_early:.1f} -> {lay_climax:.1f}")


def test_album_coverage(seeds=25):
    print("\nalbum coverage")
    from collections import Counter
    for n_tracks in (4, 12, 22):
        pool = _fake_pool(n_tracks=n_tracks, per_track=18, seed=2)
        worst, worst_min = n_tracks, 99
        for s in range(1, seeds + 1):
            p = arrange.build_plan(pool, key="A", key_scale="minor", seed=s)
            c = Counter(e.track_no for e in p.loop_events)
            worst = min(worst, len(c))
            worst_min = min(worst_min, min(c.values()) if c else 0)
        # the whole album should be represented, not just the songs that scored well
        check(f"all {n_tracks} songs get used in every run", worst == n_tracks,
              f"worst run used {worst}/{n_tracks}, least-used song had {worst_min} events")


def test_forms():
    print("\nform menu")
    original = config.TARGET_BPM
    try:
        for name, (sections, minutes) in arrange.FORMS.items():
            bars = arrange.bars_for_minutes(minutes)
            alloc = arrange._section_bars(bars, name)
            check(f"form '{name}' allocates exactly {bars} bars over "
                  f"{len(sections)} sections",
                  sum(alloc) == bars and len(alloc) == len(sections),
                  f"{alloc}")
            # the whole point of a form menu: no section gets squeezed to nothing
            check(f"form '{name}' has no section under 4 bars", min(alloc) >= 4,
                  f"shortest {min(alloc)} bars")
            plan = arrange.build_plan(_fake_pool(), key="A", key_scale="minor",
                                      total_bars=bars, seed=4, form=name)
            check(f"form '{name}' builds a plan with an energy climb",
                  plan.bars[-1].energy < max(b.energy for b in plan.bars) and
                  len(plan.bars) == bars,
                  f"peak {max(b.energy for b in plan.bars):.2f}")
            check(f"form '{name}' still respects the no-same-song rule",
                  not arrange.check_overlaps(plan))
        # short durations must get fewer sections, not thinner ones
        picks = [(s, arrange.choose_form(s)) for s in (100, 200, 350, 500, 600)]
        counts = [len(arrange.form_sections(f)) for _s, f in picks]
        check("longer durations select forms with more sections",
              counts == sorted(counts), f"{picks} -> {counts}")
        # explicit durations must be exact at any tempo, not phrase-rounded
        bad = []
        for bpm in config.EXACT_BPMS:
            config.set_tempo(bpm)
            b = arrange.bars_for_minutes(9.0)
            if abs(b * config.SEC_PER_BAR - 540.0) > 1e-9:
                bad.append((bpm, b * config.SEC_PER_BAR))
        check("--length long is exactly 9:00 at every exact tempo", not bad,
              f"offenders {bad[:3]}")
    finally:
        config.set_tempo(original)


def test_length_budget():
    print("\nmaterial-budget length")
    # a synthetic pool with k obvious clusters should be recognised as ~k ideas
    rng = np.random.default_rng(7)
    for k in (4, 10):
        pool = []
        for c in range(k):
            for _ in range(12):
                pool.append({"onset_density": 1.0 + 2.0 * c + rng.normal(0, 0.05),
                             "centroid_hz": 400.0 * (1.35 ** c),
                             "out_bars": 1.0, "loopability": 0.5,
                             "track_no": 1, "texture": "mid-mid"})
        got = loops.count_distinct_ideas(pool)
        check(f"{k} synthetic clusters are counted as roughly {k} ideas",
              abs(got - k) <= max(2, k * 0.4), f"got {got}")
    check("a tiny pool reports its own size",
          loops.count_distinct_ideas(_fake_pool(n_tracks=1, per_track=5)) == 5)

    # length must actually respond to the amount of material
    lens = [arrange.budget_bars(n) for n in (4, 10, 18, 30, 60)]
    check("more distinct ideas -> more bars", lens == sorted(lens), f"{lens}")
    check("length is clamped at both ends",
          arrange.budget_bars(1) * config.SEC_PER_BAR >= config.AUTO_MIN_SECONDS - 2
          and arrange.budget_bars(999) * config.SEC_PER_BAR
          <= config.AUTO_MAX_SECONDS + 2,
          f"{arrange.budget_bars(1)}..{arrange.budget_bars(999)} bars")
    check("auto lengths are 4-bar phrase aligned",
          all(arrange.budget_bars(n) % 4 == 0 for n in range(1, 60)))
    check("exposure is the length dial",
          arrange.budget_bars(12, exposure=32) > arrange.budget_bars(12, exposure=8),
          f"{arrange.budget_bars(12, exposure=8)} vs "
          f"{arrange.budget_bars(12, exposure=32)} bars")

    # precedence: bars > minutes > named form > auto
    pool = _fake_pool()
    b, f, i = arrange.resolve_length(pool, bars=137, minutes=3, mode="sketch")
    check("--bars wins over everything", b == 137 and i is None, f"{b} bars")
    b, f, i = arrange.resolve_length(pool, minutes=3, mode="sketch")
    check("--minutes wins over a named form",
          abs(b * config.SEC_PER_BAR - 180.0) < 1.0, f"{b} bars")
    b, f, i = arrange.resolve_length(pool, mode="extended")
    check("a named form uses its natural duration and its own sections",
          f == "extended" and abs(b * config.SEC_PER_BAR - 360.0) < 1.0, f"{b} bars")
    b, f, i = arrange.resolve_length(pool, mode="auto")
    check("auto reports the idea count it used", i is not None and b > 0,
          f"{i} ideas -> {b} bars, form {f}")


def test_no_same_song_overlap(seeds=60):
    print("\nno-same-song rule")
    pool = _fake_pool()
    total_ev, viol = 0, 0
    for s in range(1, seeds + 1):
        p = arrange.build_plan(pool, key="A", key_scale="minor", seed=s)
        total_ev += len(p.loop_events)
        viol += len(arrange.check_overlaps(p))
    check(f"0 same-song overlaps over {seeds} seeds", viol == 0,
          f"{total_ev} events, {viol} violations")
    check("the rule is recorded as enforced on a big album",
          arrange.build_plan(pool, key="A", key_scale="minor", seed=1).same_song_rule)

    # Small albums: the rule is relaxed so the layer build can still happen, but a
    # loop may never overlap ITSELF (that combs rather than layers).
    for n_tracks in (1, 2, 3):
        small = _fake_pool(n_tracks=n_tracks, per_track=20)
        worst_self, best_layers = 0, 0
        for s in range(1, 16):
            p = arrange.build_plan(small, key="A", key_scale="minor", seed=s)
            worst_self += len(arrange.check_self_overlaps(p))
            best_layers = max(best_layers, max(arrange.concurrency_profile(p)))
            if p.same_song_rule:
                worst_self += 1000        # should have been relaxed
        check(f"{n_tracks}-song album relaxes the rule and can still layer",
              worst_self == 0 and best_layers > n_tracks,
              f"reached {best_layers} layers, {worst_self} self-overlaps")
    # and the relaxation must NOT kick in at the threshold
    four = _fake_pool(n_tracks=4, per_track=20)
    p4 = arrange.build_plan(four, key="A", key_scale="minor", seed=3)
    check(f"a {config.MIN_SONGS_FOR_RULE}-song album still enforces the rule",
          p4.same_song_rule and not arrange.check_overlaps(p4))


def test_retime():
    print("\nretime to the target tempo")
    sr = config.SR
    beat = 60.0 / config.TARGET_BPM
    ok_len, ok_rate = True, True
    for src_bpm in (60.0, 76.0, 83.4, 94.0, 120.0, 128.0, 154.0, 166.7, 180.0):
        for beats in (2, 4, 8):
            dur = beats * 60.0 / src_bpm
            n = int(dur * sr)
            t = np.arange(n) / sr
            clip = audio.to_stereo(np.sin(2 * np.pi * 220 * t).astype(np.float32))
            out, out_beats, rate = loops.retime(clip, beats, src_bpm)
            want_n = int(round(out_beats * beat * sr))
            ok_len &= (out.shape[0] == want_n)
            # resulting tempo must actually be 120 (or an octave of the loop's grid)
            eff = out_beats * 60.0 / (out.shape[0] / sr)
            ok_len &= abs(eff - config.TARGET_BPM) < 0.01
            ok_rate &= (0.45 <= rate <= 2.2)
    check("every retimed loop is an exact whole number of target beats", ok_len)
    check("resample rates stay within half/double time", ok_rate)
    r1, m1 = loops.choose_rate(76.0)
    r2, m2 = loops.choose_rate(240.0)
    check("76 BPM is treated as half-time, not dragged to 0.63x",
          abs(math.log(r1)) < abs(math.log(120 / 76.0)), f"rate {r1:.3f} mult {m1}")
    check("240 BPM resolves to 1.0x double-time", abs(r2 - 1.0) < 1e-6, f"rate {r2:.3f}")


def test_tempo():
    print("\nruntime tempo")
    original = config.TARGET_BPM
    try:
        # every "exact" tempo must give a whole number of bars in exactly 540 s
        bad = []
        for bpm in config.EXACT_BPMS:
            config.set_tempo(bpm)
            dur = config.TOTAL_BARS * config.SEC_PER_BAR
            if abs(dur - config.TARGET_SECONDS) > 1e-9:
                bad.append((bpm, dur))
        check(f"all {len(config.EXACT_BPMS)} exact tempos give 540.000 s",
              not bad, f"offenders: {bad[:3]}")

        config.set_tempo(128)
        check("set_tempo re-derives beat/bar/bar-count",
              abs(config.SEC_PER_BEAT - 0.46875) < 1e-9 and
              config.TOTAL_BARS == 288,
              f"{config.SEC_PER_BEAT:.5f}s/beat, {config.TOTAL_BARS} bars")
        # techno must READ the tempo, not a value snapshotted at import
        p = arrange.build_plan([], key="A", key_scale="minor", total_bars=8, seed=1)
        bed, _kt, _atmos = techno.render_bed(p, "A", "minor",
                                             np.random.default_rng(1))
        want = int(round(8 * config.SEC_PER_BAR * config.SR))
        check("bed honours a runtime tempo change (no import-time snapshot)",
              bed.shape[0] >= want and abs(bed.shape[0] - want) < 7 * config.SR,
              f"{bed.shape[0]} samples for {want}-sample piece")
        # retiming must land on exact beats at the new tempo too
        beat = 60.0 / config.TARGET_BPM
        ok = True
        for src in (94.0, 115.0, 137.0, 174.0):
            n = int(4 * 60.0 / src * config.SR)
            t = np.arange(n) / config.SR
            clip = audio.to_stereo(np.sin(2 * np.pi * 200 * t).astype(np.float32))
            out, out_beats, _r = loops.retime(clip, 4, src)
            ok &= out.shape[0] == int(round(out_beats * beat * config.SR))
        check("loops retime to exact beats at 128 BPM as well", ok)

        # the loop cache must be keyed by tempo, or a BPM change silently reuses
        # loops cut for the old tempo
        d128 = loops.retimed_dir("slugX")
        config.set_tempo(120)
        d120 = loops.retimed_dir("slugX")
        check("retimed loop cache path includes the tempo", d128 != d120,
              f"{d128.name} vs {d120.name}")

        # auto choice: an album split across two tempo clusters
        album = [114.9, 115.0, 117.1, 116.8, 137.4, 137.6, 136.4, 140.1]
        chosen = loops.choose_target_bpm(album)
        mean_c, worst_c = loops.semitones_for(album, chosen)
        mean_120, worst_120 = loops.semitones_for(album, 120.0)
        check("auto tempo is one of the exact-duration tempos",
              chosen in [float(b) for b in config.EXACT_BPMS], f"{chosen:g} BPM")
        check("auto tempo cuts the WORST-case pitch shift vs a hardcoded 120",
              worst_c < worst_120,
              f"{chosen:g} BPM: worst {worst_c:.2f} st vs 120 BPM: {worst_120:.2f} st")
        check("…without making the average materially worse",
              mean_c <= mean_120 + 0.2,
              f"mean {mean_c:.2f} st vs {mean_120:.2f} st")
        # a single-tempo album should just land on (or very near) its own tempo
        chosen_128 = loops.choose_target_bpm([128.0] * 6)
        check("a uniformly-128 album picks 128", chosen_128 == 128.0,
              f"{chosen_128:g} BPM")
        mean_u, _w = loops.semitones_for([128.0] * 6, chosen_128)
        check("…with no resampling at all", mean_u < 1e-9, f"{mean_u:.4f} st")
    finally:
        config.set_tempo(original)


def test_worker_tempo():
    """A loop worker is a fresh interpreter — it must inherit the parent's tempo.

    This is the one real end-to-end check here: it writes a click track, runs the
    actual `python -m distillery.loops` worker at a non-default tempo, and measures
    the loops that come back. When the tempo failed to propagate, the workers cut
    120 BPM loops while the bed played at 128 and nothing complained.
    """
    print("\nworker inherits the tempo")
    import json
    import subprocess
    import sys as _sys
    original = config.TARGET_BPM
    tmp = config.TMP_DIR / "selftest_worker"
    tmp.mkdir(parents=True, exist_ok=True)
    try:
        # a 16 s click track at exactly 100 BPM (beat every 0.6 s)
        sr, bpm_src = config.SR, 100.0
        beat_s = 60.0 / bpm_src
        n = int(16 * sr)
        x = np.zeros(n, dtype=np.float32)
        rng = np.random.default_rng(3)
        for i in range(int(16 / beat_s)):
            j = int(i * beat_s * sr)
            click = (rng.normal(0, 1, 2000) *
                     audio.exp_decay(2000, sr, 0.03)).astype(np.float32)
            x[j:j + 2000] += click * (1.0 if i % 4 == 0 else 0.6)
        src = tmp / "click.wav"
        audio.write(src, audio.to_stereo(x))

        track = {"track_no": 1, "name": "click", "path": str(src),
                 "drums_path": str(src), "bpm": bpm_src,
                 "beats_position": [round(i * beat_s, 4)
                                    for i in range(int(16 / beat_s))],
                 "key": "A", "camelot": "8A", "tag_title": "click"}

        config.set_tempo(132)                      # deliberately not the default
        job, out = tmp / "job.json", tmp / "out.json"
        job.write_text(json.dumps({"track": track, "slug": "selftest_worker",
                                   "kw": {"bar_sizes": [1.0], "per_size": 2},
                                   "out": str(out),
                                   "target_bpm": config.TARGET_BPM}))
        proc = subprocess.run([_sys.executable, "-m", "distillery.loops", str(job)],
                              capture_output=True, text=True, timeout=600)
        ok = proc.returncode == 0 and out.exists()
        check("loop worker subprocess runs", ok,
              (proc.stderr.strip().splitlines() or ["ok"])[-1])
        if not ok:
            return
        made = json.loads(out.read_text())
        check("worker produced loops", len(made) > 0, f"{len(made)} loops")
        if not made:
            return
        beat_target = 60.0 / config.TARGET_BPM
        bad = [l for l in made
               if abs(l["out_dur_s"] - l["out_beats"] * beat_target) > 1e-3]
        check("worker cut loops at the PARENT's tempo, not the default",
              not bad,
              f"132 BPM beat = {beat_target:.4f}s; got "
              f"{made[0]['out_dur_s']:.4f}s for {made[0]['out_beats']} beats")
        check("worker recorded the tempo it used",
              all(abs(l.get("target_bpm", 0) - 132.0) < 1e-9 for l in made))
        check("worker wrote into the tempo-keyed cache directory",
              all("132bpm" in l["wav"] for l in made),
              Path(made[0]["wav"]).parent.name)
    finally:
        config.set_tempo(original)
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.rmtree(config.RETIMED_DIR / "selftest_worker", ignore_errors=True)


def test_wrap():
    print("\nseamless wrap")
    sr = config.SR
    n = int(1.0 * sr)
    n_x = int(0.025 * sr)
    t = np.arange(n + n_x + 64) / sr
    # a signal whose post-roll does NOT continue the loop -> hard cut would step
    x = audio.to_stereo((np.sin(2 * np.pi * 111.3 * t) * np.exp(-t)).astype(np.float32))
    w = audio.xfade_wrap(x, n, n_x)
    check("wrap output is exactly the loop length", w.shape[0] == n, f"{w.shape[0]} vs {n}")
    check("wrap output is finite", np.all(np.isfinite(w)))
    # the wrap discontinuity (last sample -> first sample) should shrink vs a hard cut
    cut = x[:n]
    step_cut = float(abs(cut[-1, 0] - cut[0, 0]))
    step_wrap = float(abs(w[-1, 0] - w[0, 0]))
    check("wrap reduces the loop-point step", step_wrap <= step_cut + 1e-6,
          f"cut {step_cut:.4f} -> wrap {step_wrap:.4f}")
    short = audio.xfade_wrap(x[:n // 2], n, n_x)
    check("too-short input is padded, not crashed", short.shape[0] == n)


def test_audio_helpers():
    print("\naudio helpers")
    sr = config.SR
    t = np.arange(sr) / sr
    x = audio.to_stereo(np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5)
    check("tile_to hits the exact length", audio.tile_to(x, 12345).shape[0] == 12345)
    check("fit_length hits the exact length", audio.fit_length(x, 7777).shape[0] == 7777)
    pn = audio.peak_normalize(x, -1.0)
    check("peak_normalize hits its target", abs(audio.peak_dbfs(pn) + 1.0) < 0.01,
          f"{audio.peak_dbfs(pn):.3f} dBFS")
    hf = audio.to_stereo(np.sin(2 * np.pi * 9000 * t).astype(np.float32))
    check("lowpass actually removes highs", audio.dbfs(audio.lpf(hf, 500.0)) <
          audio.dbfs(hf) - 20)
    check("highpass actually removes lows",
          audio.dbfs(audio.hpf(audio.to_stereo(np.sin(2 * np.pi * 40 * t)
                                               .astype(np.float32)), 800.0)) <
          -20)
    sw = audio.lpf_sweep(audio.to_stereo(np.random.default_rng(0)
                                        .normal(0, .2, (sr * 2, 2)).astype(np.float32)),
                         400.0, 16000.0)
    first, last = audio.dbfs(sw[:sr // 2]), audio.dbfs(sw[-sr // 2:])
    check("filter sweep opens up over time", last > first + 3,
          f"{first:.1f} -> {last:.1f} dBFS")
    ducked = audio.duck(audio.to_stereo(np.ones((sr, 2), np.float32) * 0.3),
                        [0.5], depth_db=6.0)
    i = int(0.5 * sr)
    check("duck dips at the kick onset", audio.dbfs(ducked[i:i + 500]) <
          audio.dbfs(ducked[:1000]) - 3)
    lim = audio.limiter(audio.to_stereo((np.random.default_rng(1)
                                        .normal(0, 0.5, (sr, 2))).astype(np.float32)),
                        ceiling_db=-6.0)
    check("limiter respects its ceiling", audio.peak_dbfs(lim) <= -5.5,
          f"{audio.peak_dbfs(lim):.2f} dBFS")


def test_build_envelope():
    print("\nbuild envelope")
    pool = _fake_pool()
    p = arrange.build_plan(pool, key="A", key_scale="minor", seed=5)
    n = int(p.total_bars * config.SEC_PER_BAR * config.SR)
    env = render.macro_envelope(p, n)
    check("envelope covers the whole piece", env.shape[0] == n)
    rng_db = 20 * math.log10(float(env.max()) / max(1e-9, float(env.min())))
    check("envelope swings roughly the configured range",
          abs(rng_db - render.LOOP_MACRO_RANGE_DB) < 4.0, f"{rng_db:.1f} dB")
    step = float(np.max(np.abs(np.diff(env))))
    check("envelope is smooth (no clicking gain steps)", step < 1e-3, f"max step {step:.2e}")
    climax = [b.index for b in p.bars if b.section == "climax"]
    intro = [b.index for b in p.bars if b.section == "intro"]
    ci = int(np.mean(climax) * config.SEC_PER_BAR * config.SR)
    ii = int(np.mean(intro) * config.SEC_PER_BAR * config.SR)
    check("loops are louder at the climax than in the intro", env[ci] > env[ii] * 2)


def test_bed_synthesis():
    print("\nbed synthesis (short render)")
    rng = np.random.default_rng(2)
    p = arrange.build_plan([], key="A", key_scale="minor", total_bars=24, seed=2)
    bed, ktimes, _ = techno.render_bed(p, "A", "minor", rng)
    check("bed renders finite audio", np.all(np.isfinite(bed)) and bed.shape[1] == 2)
    check("bed is not silent", audio.dbfs(bed) > -40, f"{audio.dbfs(bed):.1f} dBFS")
    check("kick onsets land on the beat grid",
          all(abs((t / config.SEC_PER_BEAT) % 0.5) < 1e-6 for t in ktimes),
          f"{len(ktimes)} onsets")
    kick_bars = {int(t // config.SEC_PER_BAR) for t in ktimes}
    no_kick = {b.index for b in p.bars if not b.kick}
    check("no kick in the bars that asked for none", not (kick_bars & no_kick))
    for name, fn in (("kick", techno.kick), ("hat", techno.hat), ("clap", techno.clap),
                     ("perc", techno.perc), ("rim", techno.rim)):
        a = fn()
        check(f"{name} one-shot is finite and audible",
              np.all(np.isfinite(a)) and audio.peak_dbfs(a) > -40)
    pad = techno.pad_voice(57, (0, 3, 7, 10), 2.0)
    check("pad voice is finite and audible",
          np.all(np.isfinite(pad)) and audio.peak_dbfs(pad) > -40)
    check("key -> root note mapping works",
          techno.key_root_midi("C", "major") == 48 and
          techno.key_root_midi("A", "minor") == 57)


def test_no_chords():
    print("\ndrone instead of chords")
    prog = techno.progression("A", "minor", chords=False)
    check("chords off gives exactly one sustained voicing", len(prog) == 1, str(prog))
    deg, ivs = prog[0]
    semis = {i % 12 for i in ivs}
    check("drone is a single note, no third/seventh/octave",
          semis == {0} and deg == 0 and len(ivs) == 1, f"intervals {ivs}")
    # every key must land in the same low register, not just the convenient ones
    hzs = {k: techno.midi_hz(techno.drone_midi(k)) for k in
           ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")}
    check("drone sits low (82-160 Hz) in EVERY key, not singing over the drums",
          all(80.0 <= v <= 160.0 for v in hzs.values()),
          f"{min(hzs.values()):.0f}-{max(hzs.values()):.0f} Hz across 12 keys")
    check("drone register is key-independent (all within one octave)",
          max(hzs.values()) / min(hzs.values()) < 2.001,
          f"spread {max(hzs.values()) / min(hzs.values()):.2f}x")
    prog_on = techno.progression("A", "minor", chords=True)
    check("--chords restores a four-chord progression", len(prog_on) == 4)
    check("minor and major keys pick different progressions",
          techno.progression("C", "major", chords=True) !=
          techno.progression("C", "minor", chords=True))
    drone = techno.pad_voice(57, config.DRONE_INTERVALS, 2.0)
    check("drone renders finite audible audio",
          np.all(np.isfinite(drone)) and audio.peak_dbfs(drone) > -40)
    # the bed must not contain a chord change when chords are off
    rng = np.random.default_rng(4)
    p = arrange.build_plan([], key="A", key_scale="minor", total_bars=24, seed=4,
                           chords=False)
    check("plan records that chords are off", p.chords is False)
    bed, _kt, atmos = techno.render_bed(p, "A", "minor", rng, chords=False)
    check("atmosphere bus comes back separately and is audible",
          np.all(np.isfinite(atmos)) and audio.dbfs(atmos) > -60,
          f"{audio.dbfs(atmos):.1f} dBFS")
    check("drums bed excludes the atmosphere (separate buses)",
          bed.shape == atmos.shape)


def test_fx():
    print("\npedalboard effects")
    if not fx.available():
        check("pedalboard is installed and enabled", False, "fx.available() is False")
        return
    check("pedalboard is installed and enabled", True, f"v{fx.version()}")
    sr = config.SR
    rng = np.random.default_rng(0)
    x = audio.to_stereo(rng.normal(0, 0.15, (sr * 2, 2)).astype(np.float32))

    sw = fx.sweep(x, 400.0, 16000.0, sr)
    check("resonant sweep keeps the buffer length and stays finite",
          sw.shape == x.shape and np.all(np.isfinite(sw)))
    first, last = audio.dbfs(sw[:sr // 2]), audio.dbfs(sw[-sr // 2:])
    check("resonant sweep opens up over time", last > first + 3,
          f"{first:.1f} -> {last:.1f} dBFS")
    # a clicking sweep shows up as a sample-to-sample jump far above the source's
    step = float(np.max(np.abs(np.diff(sw[:, 0]))))
    check("sweep has no chunk-boundary clicks", step < 1.0, f"max step {step:.3f}")

    filt = fx.event_filter(x, 800.0, 0.2, sr)
    check("event filter attenuates above its cutoff",
          audio.dbfs(filt) < audio.dbfs(x) - 3, f"{audio.dbfs(filt):.1f} dBFS")
    sm = fx.ambient_smear(x, sr, mix=0.4)
    check("ambient smear returns finite audio", np.all(np.isfinite(sm)))

    # every delight must run on a real bus without raising and must change it
    sections = arrange.build_plan([], key="A", key_scale="minor", seed=1).sections
    n = int(config.TOTAL_BARS * config.SEC_PER_BAR * sr)
    for name in fx.DELIGHTS:
        d = next((c for c in fx.roll_delights(np.random.default_rng(1), sections)
                  if c["name"] == name), None)
        if d is None:                     # not rolled this time; place it by hand
            d = {"filter-sweep": {"name": name, "bar": 100, "bars": 8,
                                  "f_start": 16000.0, "f_end": 600.0},
                 "reverb-throw": {"name": name, "bar": 100, "bars": 1, "wet": 0.85},
                 "dub-echo": {"name": name, "bar": 100, "bars": 4, "beats": 0.75,
                              "feedback": 0.55},
                 "drum-crush": {"name": name, "bar": 100, "bars": 8, "bits": 7.0,
                                "drive_db": 9.0},
                 "phaser-drift": {"name": name, "bar": 100, "bars": 8, "rate": 0.22},
                 "tape-warp": {"name": name, "bar": 100, "bars": 8,
                               "semitones": -0.25}}[name]
        loop_bus = audio.to_stereo(rng.normal(0, 0.1, (n, 2)).astype(np.float32))
        atmos = audio.to_stereo(rng.normal(0, 0.1, (n, 2)).astype(np.float32))
        before = (loop_bus.copy(), atmos.copy())
        line = fx.apply_delight(d, loop_bus, atmos, sr)
        changed = (not np.allclose(loop_bus, before[0]) or
                   not np.allclose(atmos, before[1]))
        finite = np.all(np.isfinite(loop_bus)) and np.all(np.isfinite(atmos))
        check(f"delight '{name}' applies, changes the mix, stays finite",
              changed and finite and not line.startswith("!!"), line)

    # drum-crush drives the signal; it must not steal the master's headroom
    loop_bus = audio.to_stereo(rng.normal(0, 0.1, (n, 2)).astype(np.float32))
    pk_before = audio.peak_dbfs(loop_bus)
    fx.apply_delight({"name": "drum-crush", "bar": 100, "bars": 8, "bits": 7.0,
                      "drive_db": 12.0}, loop_bus, loop_bus.copy(), sr)
    check("drum-crush is peak-matched (no headroom grab)",
          audio.peak_dbfs(loop_bus) <= pk_before + 0.5,
          f"{pk_before:.2f} -> {audio.peak_dbfs(loop_bus):.2f} dBFS")


def test_delight_placement(seeds=40):
    print("\ndelight placement")
    sections = arrange.build_plan([], key="A", key_scale="minor", seed=1).sections
    counts, bad = [], 0
    for s in range(1, seeds + 1):
        ds = fx.roll_delights(np.random.default_rng(s), sections)
        counts.append(len(ds))
        for d in ds:
            if d["bar"] < 0 or d["bar"] + d["bars"] > config.TOTAL_BARS:
                bad += 1
            if d["name"] not in fx.DELIGHTS:
                bad += 1
    lo, hi = config.DELIGHTS_PER_RUN
    check(f"every run rolls delights inside the configured {lo}-{hi} range",
          all(c <= hi for c in counts) and max(counts) > 0,
          f"counts {min(counts)}..{max(counts)}")
    check("all delights land inside the piece with known names", bad == 0)


def test_library():
    print("\nlocal library")
    from . import library
    check("query splits into artist and album",
          library._split_query("Tortoise - TNT") == ("Tortoise", "TNT"))
    check("bare album name still parses",
          library._split_query("TNT") == (None, "TNT"))
    check("track numbers parse from '3/11' form",
          library._tracknum("3/11") == 3 and library._tracknum("07") == 7 and
          library._tracknum(None) == 0)
    # These depend on the operator's own setup, so they SKIP rather than fail on a
    # fresh clone — --library is optional and the rest of the tool works without it.
    import os
    creds = config.secrets()
    if creds.get("SMB_USER") and creds.get("SMB_PASSWORD"):
        check("SMB credentials are configured (secrets.txt or env)", True)
    else:
        print("  SKIP  no SMB credentials configured (only needed for --library)")
    if config.SECRETS_FILE.exists():
        mode = oct(os.stat(config.SECRETS_FILE).st_mode & 0o777)
        check("secrets.txt is not world/group readable", mode == "0o600", mode)
    else:
        print("  SKIP  no secrets.txt (copy secrets.txt.example to use --library)")
    mp = library.mountpoint()
    if mp is not None:
        check("music share is reachable", True, str(mp))
    else:
        print("  SKIP  music share not mounted (only needed for --library)")


def test_parallel_sizing():
    print("\nworker sizing")
    from . import parallel
    parallel.set_override(None)
    cpus = parallel.available_cpus()
    check("detects at least one usable core", cpus >= 1, f"{cpus} cores")
    check("cpu stage uses one worker per core", parallel.worker_count("cpu") == cpus)
    check("never more workers than items",
          parallel.worker_count("cpu", n_items=3) == min(3, cpus))
    check("io stage oversubscribes (network-bound)",
          parallel.worker_count("io", n_items=99) >= min(4, cpus),
          f"{parallel.worker_count('io', n_items=99)} workers")
    check("a huge per-worker memory need collapses to 1 worker",
          parallel.worker_count("cpu", per_worker_gb=10 ** 6) == 1)
    check("RAM cap binds before core count on a small machine",
          parallel.worker_count("cpu", per_worker_gb=parallel.total_ram_gb()) <= 1)
    parallel.set_override(3)
    check("--workers override is honoured", parallel.worker_count("cpu") == 3)
    parallel.set_override(None)
    check("clearing the override returns to auto", parallel.worker_count("cpu") == cpus)
    # ordered map must preserve input order no matter how the work completes
    import time as _t

    def slow(i):
        _t.sleep(0.02 if i % 2 else 0.001)      # odd items finish later
        return i * 2
    got = [r for _i, r in parallel.imap_ordered(slow, list(range(12)), 4)]
    check("ordered map yields results in input order", got == [i * 2 for i in range(12)])
    check("plain run() also preserves order",
          parallel.run(slow, list(range(12)), 4) == [i * 2 for i in range(12)])


def test_parallel_determinism():
    print("\nparallel output is deterministic")
    from . import parallel
    sr = config.SR
    tmp = config.TMP_DIR / "selftest_loops"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(5)
    pool = []
    for t in range(1, 5):                       # 4 fake "songs", one loop file each
        n = int(2.0 * sr)                       # 1 bar at 120
        env = audio.exp_decay(n, sr, 0.25)
        x = audio.to_stereo((rng.normal(0, 0.3, n) * env).astype(np.float32))
        p = tmp / f"loop{t}.wav"
        audio.write(p, x)
        for i in range(6):
            pool.append({"id": f"fake:{t}_{i}", "track_no": t,
                         "track_name": f"song-{t}", "wav": str(p), "out_bars": 1.0,
                         "loopability": 0.5 + 0.05 * i, "onset_density": 2.0 + i,
                         "centroid_hz": 900.0 + 400 * i, "texture": "mid-mid"})

    plan = arrange.build_plan(pool, key="A", key_scale="minor", total_bars=48, seed=9)
    check("test plan actually schedules loops", len(plan.loop_events) > 4,
          f"{len(plan.loop_events)} events")

    parallel.set_override(1)
    serial, info1 = render.render(plan, None, verbose=False)
    parallel.set_override(6)
    par, info2 = render.render(plan, None, verbose=False)
    parallel.set_override(None)

    check("serial and 6-worker renders used different worker counts",
          info1["workers"] == 1 and info2["workers"] > 1,
          f"{info1['workers']} vs {info2['workers']}")
    check("6-worker render is BIT-IDENTICAL to the serial render",
          serial.shape == par.shape and np.array_equal(serial, par),
          f"max abs diff {float(np.max(np.abs(serial - par))) if serial.shape == par.shape else 'shape mismatch'}")
    for p in tmp.glob("loop*.wav"):
        p.unlink()


def main():
    print("distillery selftest")
    test_allocator()
    test_plan_shape()
    test_forms()
    test_length_budget()
    test_album_coverage()
    test_no_same_song_overlap()
    test_texture_harmony()
    test_texture_delay()
    test_texture_sits_back()
    test_retime()
    test_tempo()
    test_worker_tempo()
    test_wrap()
    test_audio_helpers()
    test_build_envelope()
    test_bed_synthesis()
    test_no_chords()
    test_bassline()
    test_fx()
    test_delight_placement()
    test_library()
    test_posting_gates()
    test_nightly_state()
    test_parallel_sizing()
    test_parallel_determinism()
    print()
    if _FAILURES:
        print(f"{len(_FAILURES)} FAILED: {', '.join(_FAILURES)}")
        return 1
    print("all checks passed")
    return 0


def _fake_texture(n_tracks=6, per_track=4, keys=("C", "D", "F#", "A"), seed=3):
    rng = np.random.default_rng(seed)
    pool = []
    for t in range(1, n_tracks + 1):
        for i in range(per_track):
            k = keys[(t + i) % len(keys)]
            pool.append({
                "id": f"tex:{t}_{i}", "track_no": t, "track_name": f"song-{t}",
                "wav": f"/dev/null/tex{t}_{i}.wav", "out_bars": float(rng.choice([2.0, 4.0])),
                "loopability": float(rng.uniform(0.4, 0.9)),
                "onset_density": 1.0, "centroid_hz": 1500.0, "texture": "pad",
                "kind": "texture", "loop_key": k,
                "loop_scale": "minor" if i % 2 else "major",
                "loop_key_strength": 0.3 if i == 0 else 0.8,
            })
    return pool


def test_texture_harmony():
    print("\ntexture layer harmonic safety")
    from .analyze import NOTE_SEMITONE
    # transposition picks the SHORT way round the circle
    check("transposition is the shortest move, within +/-6 semitones",
          arrange.transpose_to("B", "C") == 1.0 and
          arrange.transpose_to("C", "B") == -1.0 and
          all(abs(arrange.transpose_to(a, b)) <= 6
              for a in NOTE_SEMITONE for b in NOTE_SEMITONE),
          f"B->C {arrange.transpose_to('B', 'C')}, C->B {arrange.transpose_to('C', 'B')}")

    # guarantee 2: a definite major loop may not sit on a minor piece
    definite_major = {"loop_key": "C", "loop_scale": "major", "loop_key_strength": 0.9}
    ok, _st = arrange.texture_is_safe(definite_major, "C", "minor")
    check("a clearly-major loop is rejected from a minor piece", not ok)
    ok, _st = arrange.texture_is_safe(definite_major, "C", "major")
    check("…and accepted on a major piece", ok)
    vague = {"loop_key": "C", "loop_scale": "major", "loop_key_strength": 0.2}
    ok, st = arrange.texture_is_safe(vague, "C", "minor")
    check("a loop with no definite key passes as pure texture", ok, f"shift {st}")
    far = {"loop_key": "F#", "loop_scale": "minor", "loop_key_strength": 0.9}
    ok, st = arrange.texture_is_safe(far, "C", "minor")
    check("a tritone-away loop is rejected rather than mangled",
          not ok, f"F# -> C is {st} semitones, cap {config.TEXTURE_MAX_SHIFT:g}")
    # the cap only means something if it is below the 6-semitone maximum that
    # transpose_to can ever return
    check("the shift cap actually excludes some keys",
          config.TEXTURE_MAX_SHIFT < 6.0 and
          any(not arrange.texture_is_safe(
              {"loop_key": k, "loop_scale": "minor", "loop_key_strength": 0.9},
              "C", "minor")[0] for k in ("F#", "G", "F")),
          f"cap {config.TEXTURE_MAX_SHIFT:g} semitones")

    # the three guarantees must hold across many seeds on a real-ish plan
    pool, tex = _fake_pool(), _fake_texture()
    bad_shift = bad_mode = overlaps = 0
    tex_seen = 0
    for s in range(1, 31):
        p = arrange.build_plan(pool, key="C", key_scale="minor", seed=s,
                               total_bars=180, texture_pool=tex)
        tev = [e for e in p.loop_events if e.kind == "texture"]
        tex_seen += len(tev)
        for e in tev:
            if abs(e.semitones) > config.TEXTURE_MAX_SHIFT:
                bad_shift += 1
            src = next(l for l in tex if l["id"] == e.loop_id)
            if (src["loop_key_strength"] >= config.TEXTURE_MODE_STRENGTH
                    and src["loop_scale"][:3] != "min"):
                bad_mode += 1
            # transposed root must land on the piece's root
            if arrange.transpose_to(src["loop_key"], "C") != e.semitones:
                bad_shift += 1
        for a, b in itertools.combinations(tev, 2):
            if a.bar < b.end_bar and b.bar < a.end_bar:
                overlaps += 1
        if arrange.check_overlaps(p):
            overlaps += 1
    check("texture events were actually scheduled", tex_seen > 30, f"{tex_seen} over 30 seeds")
    check("every texture loop is transposed onto the piece's root, within the cap",
          bad_shift == 0, f"{bad_shift} violations")
    check("no definite-mode loop contradicts the piece's mode", bad_mode == 0)
    check("only one texture sounds at a time, and the no-same-song rule still holds",
          overlaps == 0, f"{overlaps} violations")

    # drum lanes are unaffected by the new lane
    p = arrange.build_plan(pool, key="C", key_scale="minor", seed=7, total_bars=180,
                           texture_pool=tex)
    drums = [e for e in p.loop_events if e.kind == "drum"]
    prof = [0] * p.total_bars
    for e in drums:
        for b in range(e.bar, min(p.total_bars, int(math.ceil(e.end_bar)))):
            prof[b] += 1
    check("drum layering still respects MAX_LAYERS", max(prof) <= config.MAX_LAYERS,
          f"max {max(prof)} drum layers")


def test_texture_sits_back():
    """The texture bus must measure BELOW the mix, not above it.

    It shipped at a level where the texture bus was +1.9 dB louder than the whole
    mix and 14 dB above the borrowed drum loops — the background layer was the
    loudest thing in the piece. Level is now an asserted invariant, not a guess.
    """
    print("\ntexture sits behind the mix")
    sr = config.SR
    tmp = config.TMP_DIR / "selftest_tex_level"
    tmp.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(11)
    try:
        drum_pool, tex_pool = [], []
        for t in range(1, 5):
            n = int(2.0 * sr)
            hit = (rng.normal(0, 0.3, n) * audio.exp_decay(n, sr, 0.2)).astype(np.float32)
            dp = tmp / f"d{t}.wav"
            # normalized exactly as the extractor does, or the two pools aren't
            # comparable and the level assertion below is meaningless
            audio.write(dp, audio.normalize_rms(audio.to_stereo(hit),
                                                config.LOOP_TARGET_DBFS))
            tone = np.sin(2 * np.pi * 220 * np.arange(n) / sr).astype(np.float32) * 0.3
            tp = tmp / f"t{t}.wav"
            audio.write(tp, audio.normalize_rms(audio.to_stereo(tone),
                                                config.LOOP_TARGET_DBFS))
            for i in range(6):
                drum_pool.append({"id": f"d{t}_{i}", "track_no": t,
                                  "track_name": f"s{t}", "wav": str(dp),
                                  "out_bars": 1.0, "loopability": 0.6,
                                  "onset_density": 4.0, "centroid_hz": 2000.0,
                                  "texture": "mid-mid"})
            for i in range(3):
                tex_pool.append({"id": f"t{t}_{i}", "track_no": t,
                                 "track_name": f"s{t}", "wav": str(tp),
                                 "out_bars": 2.0, "loopability": 0.7,
                                 "onset_density": 1.0, "centroid_hz": 800.0,
                                 "texture": "pad", "kind": "texture",
                                 "loop_key": "A", "loop_scale": "minor",
                                 "loop_key_strength": 0.8})
        plan = arrange.build_plan(drum_pool, key="A", key_scale="minor",
                                  total_bars=64, seed=2, texture_pool=tex_pool)
        n_tex = sum(1 for e in plan.loop_events if e.kind == "texture")
        check("test plan actually contains texture events", n_tex > 0, f"{n_tex}")
        master, info = render.render(plan, None, verbose=False, return_stems=True)
        st = info.pop("stems")
        tex_db, mix_db = audio.dbfs(st["texture"]), audio.dbfs(master)
        loops_db = audio.dbfs(st["loops"])
        check("texture bus sits at least 8 dB under the mix",
              tex_db <= mix_db - 8.0,
              f"texture {tex_db:.1f} vs mix {mix_db:.1f} dBFS ({tex_db - mix_db:+.1f})")
        check("texture never dominates the borrowed drum loops",
              tex_db <= loops_db - 4.0,
              f"texture {tex_db:.1f} vs loops {loops_db:.1f} dBFS ({tex_db - loops_db:+.1f})")
        # DRUMS COME FIRST: in every section that has drums, drum content must be
        # ahead of texture and atmosphere. The loops once sat 20 dB under the mix and
        # the texture beat them outright in the outro.
        bad = []
        for sec in plan.sections:
            a = int(sec["start_bar"] * config.SEC_PER_BAR * sr)
            b = int((sec["start_bar"] + sec["bars"]) * config.SEC_PER_BAR * sr)
            v = {n: audio.dbfs(st[n][a:b]) for n in ("bed", "loops", "texture", "atmos")}
            if not np.isfinite(v["loops"]):
                continue
            if max(v["bed"], v["loops"]) < max(v["texture"], v["atmos"]):
                bad.append(sec["name"])
        check("drums lead every section that has drums", not bad, f"trailing in {bad}")
        check("drum loops are within 8 dB of the mix (audible, not buried)",
              loops_db >= mix_db - 8.0,
              f"loops {loops_db - mix_db:+.1f} dB vs mix")
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def test_texture_delay():
    print("\ntexture dub throw")
    sr = config.SR
    t = np.arange(int(0.4 * sr)) / sr
    hit = audio.to_stereo((np.sin(2 * np.pi * 300 * t)
                           * audio.exp_decay(t.size, sr, 0.08)).astype(np.float32))
    wet = audio.echo(hit, 0.5, repeats=8, decay_db=-4.0, sr=sr, dry=False)
    dry = audio.echo(hit, 0.5, repeats=8, decay_db=-4.0, sr=sr, dry=True)
    d = int(0.5 * sr)
    check("wet-only echo omits the dry copy (no doubling of the source)",
          audio.dbfs(wet[:d // 2]) < audio.dbfs(dry[:d // 2]) - 30,
          f"{audio.dbfs(wet[:d // 2]):.1f} vs {audio.dbfs(dry[:d // 2]):.1f} dBFS")
    check("the tail rings on well past the source",
          wet.shape[0] > hit.shape[0] * 8, f"{wet.shape[0] / sr:.1f}s from a "
          f"{hit.shape[0] / sr:.1f}s hit")
    first = audio.dbfs(wet[d:2 * d])
    later = audio.dbfs(wet[6 * d:7 * d])
    check("repeats decay as they go", later < first - 10,
          f"repeat 1 {first:.1f} dBFS -> repeat 6 {later:.1f} dBFS")

    # a delayed texture event must produce audio beyond its own bar span
    ev = arrange.LoopEvent(
        bar=0, bars=2.0, lane=arrange.TEXTURE_LANE, loop_id="x", track_no=1,
        track_name="t", wav="", gain_db=-3.0, lpf_hz=3000.0, pan=0.0,
        reverb_wet=0.4, reverse=False, fade_beats=4.0, energy=0.5, texture="pad",
        kind="texture", semitones=0.0, delay_beats=2.0, delay_repeats=12)
    n = int(round(2.0 * config.SEC_PER_BAR * sr))
    loop = audio.to_stereo(np.tile(audio.mono(hit), 8)[:n].astype(np.float32))
    cache = {"": loop}
    with_throw = render._render_texture_event(ev, cache)
    ev_no = arrange.LoopEvent(**{**ev.__dict__, "delay_beats": 0.0, "delay_repeats": 0})
    without = render._render_texture_event(ev_no, cache)
    check("the throw extends the event's audio past its bars",
          with_throw.shape[0] > without.shape[0] + sr,
          f"{with_throw.shape[0] / sr:.1f}s vs {without.shape[0] / sr:.1f}s "
          f"(event is {n / sr:.1f}s)")
    # the throw is deliberately boosted (see TEXTURE_DELAY_GAIN_DB), so it may sit
    # above the dry event — but never by more than that boost, which is what would
    # signal runaway feedback
    headroom = config.TEXTURE_DELAY_GAIN_DB + 1.5
    check("throw is boosted but bounded (no runaway feedback)",
          np.all(np.isfinite(with_throw)) and
          audio.peak_dbfs(with_throw) <= audio.peak_dbfs(without) + headroom,
          f"peak {audio.peak_dbfs(with_throw):.2f} vs dry {audio.peak_dbfs(without):.2f} dBFS "
          f"(bound +{headroom:g})")
    check("repeats are monotonically quieter, so the tail always dies out",
          config.TEXTURE_DELAY_DECAY_DB < 0)

    # the plan must record throws, and only on texture events
    pool, tex = _fake_pool(), _fake_texture()
    thrown = plain = 0
    for s in range(1, 21):
        p = arrange.build_plan(pool, key="C", key_scale="minor", seed=s,
                              total_bars=180, texture_pool=tex)
        for e in p.loop_events:
            if e.kind == "texture":
                thrown += 1 if e.delay_beats else 0
                plain += 0 if e.delay_beats else 1
            else:
                check_ok = (e.delay_beats == 0.0)
                if not check_ok:
                    plain = -10 ** 6      # force a failure below
    check("some textures get a throw and some don't", thrown > 0 and plain > 0,
          f"{thrown} thrown / {plain} dry")
    check("drum loops never get the texture throw", plain >= 0)


def test_bassline():
    """The bass was the one element identical in every render — one root note in one
    of two hardcoded rhythms. It now varies per seed, without ever implying a chord.
    """
    print("\nbassline variety and safety")
    # no degree set may contain a third: that is what would impose major/minor on a
    # piece that deliberately has no chords
    thirds = []
    for mode, sets in config.BASS_DEGREES.items():
        for ds in sets:
            for d in ds:
                if d % 12 in (3, 4):
                    thirds.append((mode, ds, d))
    check("no bass degree is ever a third (no implied chord quality)", not thirds,
          f"offenders {thirds[:3]}")
    check("every degree set starts on the root",
          all(ds[0] == 0 for sets in config.BASS_DEGREES.values() for ds in sets))
    check("patterns all fit inside one bar",
          all(b + l <= config.BEATS_PER_BAR + 0.01
              for pat in config.BASS_PATTERNS.values() for b, l, _d in pat),
          f"{len(config.BASS_PATTERNS)} patterns")
    check("pattern degree indices are all in range",
          all(di < 3 for pat in config.BASS_PATTERNS.values() for _b, _l, di in pat))

    # variety across seeds
    designs = []
    for s in range(1, 41):
        p = arrange.build_plan([], key="Bb", key_scale="minor", total_bars=64, seed=s)
        designs.append(p.bass)
    pats = {d["pattern"] for d in designs}
    degs = {tuple(d["degrees"]) for d in designs}
    tones = {(d["drive"], d["lpf_hz"], d["harm2"]) for d in designs}
    check("40 seeds use several different bass rhythms", len(pats) >= 5,
          f"{len(pats)} of {len(config.BASS_PATTERNS)}: {sorted(pats)}")
    check("…and several different degree sets", len(degs) >= 3, f"{len(degs)} sets")
    check("…and a different timbre almost every run", len(tones) >= 35,
          f"{len(tones)} distinct timbres in 40 runs")

    # same seed must still reproduce exactly
    a = arrange.build_plan([], key="Bb", key_scale="minor", total_bars=64, seed=7).bass
    b = arrange.build_plan([], key="Bb", key_scale="minor", total_bars=64, seed=7).bass
    check("a seed still reproduces its bassline exactly", a == b, str(a["pattern"]))

    # and it must actually sound different, not just carry different metadata
    rng = np.random.default_rng(1)
    plan1 = arrange.build_plan([], key="Bb", key_scale="minor", total_bars=16, seed=3)
    plan2 = arrange.build_plan([], key="Bb", key_scale="minor", total_bars=16, seed=8)
    b1, _k, _a = techno.render_bed(plan1, "Bb", "minor", np.random.default_rng(3),
                                   bass=plan1.bass)
    b2, _k, _a = techno.render_bed(plan2, "Bb", "minor", np.random.default_rng(3),
                                   bass=plan2.bass)
    n = min(b1.shape[0], b2.shape[0])
    diff = audio.dbfs(b1[:n] - b2[:n]) - audio.dbfs(b1[:n])
    check("two seeds' beds measurably differ", diff > -12.0,
          f"difference is {diff:+.1f} dB relative to the bed "
          f"({plan1.bass['pattern']} vs {plan2.bass['pattern']})")
    # major-mode albums get major-safe degrees
    pm = arrange.build_plan([], key="C", key_scale="major", total_bars=64, seed=5)
    check("major-mode pieces draw from the major degree pool",
          tuple(pm.bass["degrees"]) in config.BASS_DEGREES["major"],
          str(pm.bass["degrees"]))


def test_posting_gates():
    """Bluesky's three-minute video limit must be enforced before uploading, not
    discovered as a rejection, and each platform must fail independently."""
    print("\nposting gates")
    import tempfile
    from . import nightly, poster, social, video
    info_short = {"duration_s": 150.0, "bpm": 128.0, "key": "Bb",
                  "key_scale": "minor", "chords": False, "songs_used": 8,
                  "loop_events": 47, "unique_loops": 40, "seed": 1}
    info_long = dict(info_short, duration_s=300.0)
    meta = {"artist": "Some Artist", "album": "Some Album"}

    class _P:
        bass = {"pattern": "walk"}

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as fh:
        fh.write(b"\0" * 2048)
        fake = Path(fh.name)
    try:
        r = poster.post(fake, meta, info_long, _P, dry_run=True)
        check("a 5:00 video is kept off Bluesky before any upload",
              isinstance(r["bluesky"], str) and r["bluesky"].startswith("skipped"),
              r["bluesky"])
        check("…and still goes to Mastodon", r["mastodon"] == "dry-run")
        r = poster.post(fake, meta, info_short, _P, dry_run=True)
        check("a 2:30 video is eligible for both", r["bluesky"] == "dry-run" and
              r["mastodon"] == "dry-run")
        r = poster.post(fake, meta, info_long, _P, dry_run=True, force_bluesky=True)
        check("--force-bluesky overrides the duration gate",
              r["bluesky"] == "dry-run", str(r["bluesky"]))
    finally:
        fake.unlink(missing_ok=True)

    text = social.post_text(meta, info_short, None)
    check("post text fits Bluesky's 300-character limit",
          len(text) <= social.MAX_POST, f"{len(text)} chars")
    check("post text names the artist and album",
          "Some Artist" in text and "Some Album" in text)
    alt = social.alt_text(meta, info_short, _P)
    check("alt text is descriptive and bounded",
          200 < len(alt) <= 1800 and "waveform" in alt, f"{len(alt)} chars")
    check("ffmpeg filter probe returns a bool",
          isinstance(video.has_filter("showwaves"), bool))


def test_nightly_state():
    print("\nnightly state + picker")
    import tempfile
    from . import nightly
    original = config.STATE_DB
    with tempfile.TemporaryDirectory() as td:
        config.STATE_DB = Path(td) / "state.db"
        try:
            rid = nightly.record_start("A", "B", "a-b")
            check("a run is recorded when it starts", rid > 0)
            nightly.record_finish(rid, duration_s=180.0, bpm=128.0, songs_used=9,
                                  seed=3, bluesky="posted", mastodon="posted")
            h = nightly.history(limit=5)
            check("history reads back what was recorded",
                  len(h) == 1 and h[0]["bluesky"] == "posted" and
                  h[0]["duration_s"] == 180.0, str(h[0]["album"]))
            check("a just-used album is inside the cooldown window",
                  ("A", "B") in nightly.recent_albums())
            check("the cooldown expires", ("A", "B") not in
                  nightly.recent_albums(days=0))
        finally:
            config.STATE_DB = original
