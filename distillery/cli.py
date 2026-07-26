"""distillery CLI — one album in, one nine-minute ambient techno track out.

    python -m distillery remix --album "Boards of Canada - Geogaddi"
    python -m distillery remix --url "https://…playlist…"
    python -m distillery remix --local ~/Music/SomeAlbum

Stages are individually runnable and every stage caches, so a re-run picks up
where it left off:

    download -> analyze (essentia) -> stems (demucs drums) -> loops (cut+retime)
             -> arrange -> render
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from . import (analyze, arrange, audio, config, download, fx, library,
               loops as loopmod, parallel, render, stems)


def _album_dir_for(args):
    """Resolve the album source into (dir, meta), fetching if needed."""
    if getattr(args, "library", None):
        return library.fetch_album(args.library, limit=args.limit,
                                   force=args.force_download)
    if args.local:
        return download.use_local(args.local, artist=args.artist, album=args.album_name)
    if args.url:
        return download.download_url(args.url, limit=args.limit)
    if args.album:
        return download.download_album(args.album, limit=args.limit, force=args.force_download)
    if args.slug:
        d = config.ALBUMS_DIR / args.slug
        meta_p = d / "album.json"
        if not meta_p.exists():
            raise SystemExit(f"no cached album at {d}")
        return d, json.loads(meta_p.read_text())
    raise SystemExit("give one of --library / --album / --url / --local / --slug")


def _album_key(tracks):
    """Album key = the most common key among tracks, ties broken by key strength."""
    votes = Counter()
    strength = {}
    for t in tracks:
        if not t.get("key"):
            continue
        k = (t["key"], t.get("key_scale") or "minor")
        votes[k] += 1
        strength[k] = max(strength.get(k, 0.0), t.get("key_strength") or 0.0)
    if not votes:
        return "A", "minor"
    best = max(votes, key=lambda k: (votes[k], strength.get(k, 0.0)))
    return best


def _apply_overrides(args):
    """CLI knobs that live in config/parallel so every module sees them."""
    if getattr(args, "loops_db", None) is not None:
        config.LOOP_LEVEL_DB = float(args.loops_db)
        print(f"  loop-bus level: {config.LOOP_LEVEL_DB:+.1f} dB")
    if getattr(args, "texture_db", None) is not None:
        config.TEXTURE_LEVEL_DB = float(args.texture_db)
        print(f"  texture level: {config.TEXTURE_LEVEL_DB:+.1f} dB")
    if getattr(args, "drone_db", None) is not None:
        config.DRONE_LEVEL_DB = float(args.drone_db)
        print(f"  drone level: {config.DRONE_LEVEL_DB:+.1f} dB")
    parallel.set_override(getattr(args, "workers", None))
    print(f"  hardware: {parallel.describe()}")
    bpm = getattr(args, "bpm", None)
    if bpm and str(bpm).lower() != "auto":
        config.set_tempo(float(bpm))
        print(f"  tempo: {config.TARGET_BPM:g} BPM ({config.TOTAL_BARS} bars)")


def _resolve_pool(slug, bpm_arg):
    """Load a cached loop pool and adopt the tempo it was cut for.

    Loop wavs are physically resampled, so the pool defines the tempo — arranging a
    120 BPM pool under a 128 BPM bed would put everything out of sync.
    """
    if bpm_arg:
        config.set_tempo(float(bpm_arg))
        idx = loopmod.pool_index(slug)
        if not idx.exists():
            raise SystemExit(f"no {config.TARGET_BPM:g} BPM loop pool for {slug!r} — "
                             f"run remix --bpm {config.TARGET_BPM:g} first")
        return json.loads(idx.read_text())

    base = config.RETIMED_DIR / slug
    cands = sorted(base.glob("*bpm/loops.json"))
    legacy = base / "loops.json"
    if legacy.exists():
        cands.append(legacy)
    if not cands:
        raise SystemExit(f"no loop pool for {slug!r} — run remix first")
    idx = max(cands, key=lambda q: q.stat().st_mtime)
    pool = json.loads(idx.read_text())
    bpm = (pool[0].get("target_bpm") if pool else None) or 120.0
    config.set_tempo(bpm)
    print(f"  pool: {idx.parent.name} -> {config.TARGET_BPM:g} BPM, "
          f"{config.TOTAL_BARS} bars")
    return pool


def default_remix_args(**over):
    """A full remix Namespace with defaults, for callers that aren't argparse."""
    base = dict(library=None, album=None, url=None, local=None, slug=None,
                artist=None, album_name=None, limit=None, bars=None, length=None,
                minutes=None, exposure=None, bpm="auto", seed=None,
                bar_sizes=list(config.LOOP_BAR_SIZES),
                per_size=config.LOOPS_PER_SIZE, mp3=False, chords=False,
                drone_db=None, texture_db=None, loops_db=None, workers=None,
                no_texture=False, keep_source_loops=False, force_download=False,
                force_analyze=False, force_stems=False, force_loops=False)
    base.update(over)
    return argparse.Namespace(**base)


def cmd_remix(args):
    run_remix(args)
    return 0


def run_remix(args):
    """The whole pipeline. Returns dict(base, wav, plan, meta, info)."""
    config.ensure_dirs()
    _apply_overrides(args)
    t_start = time.time()

    timings = {}
    _t = time.time()
    album_dir, meta = _album_dir_for(args)
    timings["fetch"] = round(time.time() - _t, 1)
    slug = meta["slug"]
    track_paths = download.local_tracks(album_dir)
    if args.limit:
        track_paths = track_paths[:args.limit]
    if not track_paths:
        raise SystemExit(f"no audio in {album_dir}")

    print(f"\n== 1/6 essentia analysis ({len(track_paths)} tracks)")
    _t = time.time()
    tracks = analyze.analyze_album(track_paths, album_dir / "analysis",
                                  force=args.force_analyze)
    timings["analyze"] = round(time.time() - _t, 1)
    print(f"  analysis: {timings['analyze']:.1f}s")
    if not tracks:
        raise SystemExit("no tracks analyzed successfully")

    if str(getattr(args, "bpm", "auto")).lower() == "auto":
        bpms = [t["bpm"] for t in tracks]
        chosen = loopmod.choose_target_bpm(bpms)
        config.set_tempo(chosen)
        mean_st, worst_st = loopmod.semitones_for(bpms, chosen)
        m120, w120 = loopmod.semitones_for(bpms, 120.0)
        print(f"  tempo (auto): {config.TARGET_BPM:g} BPM, {config.TOTAL_BARS} bars "
              f"-> {config.TOTAL_BARS * config.SEC_PER_BAR:.3f}s")
        print(f"    pitch shift from resampling: mean {mean_st:.2f}, worst "
              f"{worst_st:.2f} semitones   (at 120 BPM it would be "
              f"{m120:.2f} / {w120:.2f})")

    print(f"\n== 2/6 demucs drum stems ({len(tracks)} tracks)")
    _t = time.time()
    tracks = stems.drum_stems(tracks, slug, force=args.force_stems)
    timings["stems"] = round(time.time() - _t, 1)
    print(f"  stems: {timings['stems']:.1f}s")
    if not tracks:
        raise SystemExit("no drum stems produced")
    if len({t["track_no"] for t in tracks}) < 2:
        raise SystemExit("need at least 2 songs with drum stems (no-same-song rule)")

    print(f"\n== 3/6 loop extraction + retime to {config.TARGET_BPM:g} BPM")
    _t = time.time()
    pool = loopmod.extract_album_loops(
        tracks, slug, force=args.force_loops,
        bar_sizes=tuple(args.bar_sizes), per_size=args.per_size,
        keep_source_loops=args.keep_source_loops)
    timings["loops"] = round(time.time() - _t, 1)
    print(f"  loops: {timings['loops']:.1f}s")
    if len(pool) < 4:
        raise SystemExit(f"only {len(pool)} loops extracted — not enough to build with")
    by_track = Counter(l["track_no"] for l in pool)
    print(f"  pool: {len(pool)} loops from {len(by_track)} songs "
          f"(min {min(by_track.values())}, max {max(by_track.values())} per song)")

    texture_pool = []
    if not args.no_texture and config.TEXTURE_ENABLED:
        texture_pool = loopmod.extract_album_texture(tracks, slug,
                                                     force=args.force_loops)
        if texture_pool:
            print(f"  texture pool: {len(texture_pool)} loops from the 'other' stem "
                  f"({len({l['track_no'] for l in texture_pool})} songs)")

    print("\n== 4/6 arrangement")
    key, scale = _album_key(tracks)
    total_bars, form, ideas = arrange.resolve_length(
        pool, mode=args.length, bars=args.bars, minutes=args.minutes,
        exposure=args.exposure)
    if ideas:
        print(f"  material: {ideas} distinct ideas in {len(pool)} loops -> "
              f"{total_bars} bars ({total_bars * config.SEC_PER_BAR:.0f}s) "
              f"at {args.exposure or config.EXPOSURE_BARS:g} bars of exposure each")
    plan = arrange.build_plan(pool, key=key, key_scale=scale,
                              total_bars=total_bars, seed=args.seed,
                              chords=args.chords, form=form, ideas=ideas,
                              texture_pool=texture_pool)
    bad = arrange.check_overlaps(plan)
    selfbad = arrange.check_self_overlaps(plan)
    print(arrange.report(plan, meta))
    if bad or selfbad:
        print(f"\n!! {len(bad)} same-song / {len(selfbad)} self overlaps — "
              "this is a bug, refusing to render")
        for a, b, ba, bb in bad[:6]:
            print(f"   {a} @{ba}  vs  {b} @{bb}")
        for lid, ba, bb in selfbad[:6]:
            print(f"   {lid} overlaps itself @{ba} and @{bb}")
        raise SystemExit(2)
    print("  ✓ " + ("no two simultaneous loops come from the same song"
                    if plan.same_song_rule else
                    "same-song rule relaxed for a small album; no loop overlaps itself"))

    print("\n== 5/6 render")
    _t = time.time()
    master, info = render.render(plan, meta)
    timings["render"] = round(time.time() - _t, 1)

    print("\n== 6/6 write")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = config.OUT_DIR / f"distillery_{slug}_{stamp}"
    audio.write(f"{base}.wav", master)
    if args.mp3:
        audio.to_mp3(f"{base}.wav", f"{base}.mp3")
    (Path(f"{base}.plan.json")).write_text(arrange.plan_to_json(plan))
    levels = render.section_levels(master, plan)
    rep = (arrange.report(plan, meta) + "\n\nrendered\n" +
           json.dumps(info, indent=2) + "\n\nsection RMS (dBFS)\n" +
           "\n".join(f"  {n:<13}{v:>8.2f}" for n, v in levels) + "\n")
    Path(f"{base}.txt").write_text(rep)

    print(f"\n{json.dumps(info, indent=2)}")
    print("\nsection RMS (dBFS):")
    for n, v in levels:
        print(f"  {n:<13}{v:>8.2f}")
    print(f"\n-> {base}.wav" + (f"\n-> {base}.mp3" if args.mp3 else ""))
    print(f"   {info['duration_s']:.1f}s  ({info['duration_s'] / 60:.2f} min)  "
          f"in {time.time() - t_start:.0f}s total")
    print("   stage times: " + "  ".join(f"{k} {v:g}s" for k, v in timings.items())
          + f"   (workers: {info.get('workers')})")
    return {"base": base, "wav": Path(f"{base}.wav"), "plan": plan, "meta": meta,
            "info": info, "timings": timings}


def cmd_nightly(args):
    """Pick an album, distil it, render a video, post it. Built for cron."""
    from . import nightly, poster
    config.ensure_dirs()
    _apply_overrides(args)
    t0 = time.time()
    print(f"== distillery nightly  {time.strftime('%Y-%m-%d %H:%M:%S')}")

    chosen = nightly.pick(seed=args.pick_seed, min_tracks=args.min_tracks)
    query = f"{chosen['artist']} - {chosen['album']}"
    slug = download.slugify(f"{chosen['artist']}-{chosen['album']}")
    run_id = nightly.record_start(chosen["artist"], chosen["album"], slug)

    try:
        out = run_remix(default_remix_args(
            library=query, limit=args.limit, seed=args.seed, mp3=args.mp3,
            workers=args.workers, minutes=args.minutes, length=args.length))
    except BaseException as e:              # noqa: BLE001 - record then re-raise
        nightly.record_finish(run_id, error=f"{type(e).__name__}: {e}")
        raise

    info, plan, meta = out["info"], out["plan"], out["meta"]
    nightly.record_finish(run_id, duration_s=info["duration_s"], bpm=info["bpm"],
                          songs_used=info["songs_used"], seed=info["seed"],
                          wav=str(out["wav"]))

    print("\n== 7/7 video + post")
    mp4 = None
    try:
        mp4 = poster.make_video(out["wav"], meta, info, plan)
        nightly.record_finish(run_id, mp4=str(mp4))
    except Exception as e:                  # noqa: BLE001 - audio is already safe
        print(f"  !! video failed: {type(e).__name__}: {e}")
        nightly.record_finish(run_id, error=f"video: {type(e).__name__}: {e}")

    if mp4 and config.POST_ENABLED and not args.no_post:
        res = poster.post(mp4, meta, info, plan, dry_run=args.dry_run,
                          force_bluesky=args.force_bluesky)
        nightly.record_finish(run_id, bluesky=res["bluesky"],
                              mastodon=res["mastodon"])
    elif mp4:
        print("  posting disabled (--no-post or DISTILLERY_POST=0)")

    print(f"\ndone in {time.time() - t0:.0f}s")
    return 0


def cmd_post(args):
    """Render a video for an existing distillation and post it."""
    from . import poster
    config.ensure_dirs()
    wav = Path(args.wav) if args.wav else None
    if wav is None:
        wavs = sorted(config.OUT_DIR.glob("distillery_*.wav"),
                      key=lambda p: p.stat().st_mtime)
        if not wavs:
            raise SystemExit("no renders in data/output — run remix first")
        wav = wavs[-1]
    txt, planp = wav.with_suffix(".txt"), Path(str(wav)[:-4] + ".plan.json")
    if not planp.exists():
        raise SystemExit(f"no plan beside {wav.name} — can't caption it")
    plan_json = json.loads(planp.read_text())
    # rebuild just enough of `info` for the caption from the plan and the report
    info = {"duration_s": plan_json["total_bars"] * (60.0 / plan_json["bpm"]) * 4,
            "bpm": plan_json["bpm"], "key": plan_json["key"],
            "key_scale": plan_json["key_scale"], "chords": plan_json.get("chords"),
            "songs_used": len({e["track_no"] for e in plan_json["loop_events"]}),
            "loop_events": len(plan_json["loop_events"]),
            "unique_loops": len({e["loop_id"] for e in plan_json["loop_events"]}),
            "seed": plan_json["seed"]}
    if txt.exists():
        import re as _re
        m = _re.search(r'"duration_s": ([0-9.]+)', txt.read_text())
        if m:
            info["duration_s"] = float(m.group(1))
    slug = wav.stem.replace("distillery_", "").rsplit("_", 1)[0]
    metap = config.ALBUMS_DIR / slug / "album.json"
    meta = json.loads(metap.read_text()) if metap.exists() else {"slug": slug}

    class _P:            # poster only needs .bass off the plan
        bass = plan_json.get("bass")
    print(f"  source: {wav.name}")
    mp4 = Path(args.mp4) if args.mp4 else poster.make_video(wav, meta, info, _P)
    poster.post(mp4, meta, info, _P, dry_run=args.dry_run,
                force_bluesky=args.force_bluesky)
    return 0


def cmd_history(args):
    from . import nightly
    rows = nightly.history(limit=args.limit)
    if not rows:
        print("no nightly runs recorded yet")
        return 0
    print(f"{'when':<17}{'album':<44}{'len':>6}{'bpm':>5}  bluesky / mastodon")
    for r in rows:
        when = time.strftime("%Y-%m-%d %H:%M", time.localtime(r["started_at"]))
        name = f"{r['artist']} — {r['album']}"[:42]
        dur = f"{int((r['duration_s'] or 0) // 60)}:{int((r['duration_s'] or 0) % 60):02d}"
        print(f"{when:<17}{name:<44}{dur:>6}{(r['bpm'] or 0):>5.0f}  "
              f"{(r['bluesky'] or '-')[:28]} / {(r['mastodon'] or '-')[:28]}"
              + (f"   ERROR {r['error'][:40]}" if r["error"] else ""))
    return 0


def cmd_arrange_only(args):
    """Re-arrange + re-render from an existing loop pool (fast iteration)."""
    config.ensure_dirs()
    _apply_overrides(args)
    slug = args.slug
    pool = _resolve_pool(slug, getattr(args, "bpm", None))
    meta_p = config.ALBUMS_DIR / slug / "album.json"
    meta = json.loads(meta_p.read_text()) if meta_p.exists() else {"slug": slug}
    adir = config.ALBUMS_DIR / slug / "analysis"
    tracks = [json.loads(p.read_text()) for p in sorted(adir.glob("*.json"))] \
        if adir.exists() else []
    key, scale = _album_key([t for t in tracks if t.get("status") == "ok"])
    total_bars, form, ideas = arrange.resolve_length(
        pool, mode=args.length, bars=args.bars, minutes=args.minutes,
        exposure=args.exposure)
    tex_idx = loopmod.texture_index(slug)
    texture_pool = json.loads(tex_idx.read_text()) if tex_idx.exists() else []
    plan = arrange.build_plan(pool, key=key, key_scale=scale, total_bars=total_bars,
                              seed=args.seed, chords=args.chords, form=form,
                              ideas=ideas, texture_pool=texture_pool)
    print(arrange.report(plan, meta))
    if arrange.check_overlaps(plan) or arrange.check_self_overlaps(plan):
        raise SystemExit("!! overlap rule violated")
    print("  ✓ " + ("no two simultaneous loops come from the same song"
                    if plan.same_song_rule else
                    "same-song rule relaxed for a small album; no loop overlaps itself"))
    if args.dry_run:
        return 0
    master, info = render.render(plan, meta)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    base = config.OUT_DIR / f"distillery_{slug}_{stamp}"
    audio.write(f"{base}.wav", master)
    if args.mp3:
        audio.to_mp3(f"{base}.wav", f"{base}.mp3")
    Path(f"{base}.plan.json").write_text(arrange.plan_to_json(plan))
    levels = render.section_levels(master, plan)
    Path(f"{base}.txt").write_text(
        arrange.report(plan, meta) + "\n\nrendered\n" + json.dumps(info, indent=2) +
        "\n\nsection RMS (dBFS)\n" +
        "\n".join(f"  {n:<13}{v:>8.2f}" for n, v in levels) + "\n")
    print(json.dumps(info, indent=2))
    print("\nsection RMS (dBFS):")
    for n, v in levels:
        print(f"  {n:<13}{v:>8.2f}")
    print(f"-> {base}.wav")
    return 0


def cmd_bed(args):
    """Render just the synthesized bed (no samples) — for auditioning the techno."""
    config.ensure_dirs()
    _apply_overrides(args)
    total_bars, form, _ideas = arrange.resolve_length(
        [], mode=args.length or "long", bars=args.bars, minutes=args.minutes)
    plan = arrange.build_plan([], key=args.key, key_scale=args.scale,
                              total_bars=total_bars, seed=args.seed,
                              chords=args.chords, form=form)
    master, info = render.render(plan, None)
    out = config.OUT_DIR / f"bed_{args.key}{args.scale[:3]}_{plan.seed}.wav"
    audio.write(out, master)
    print(json.dumps(info, indent=2))
    print(f"-> {out}")
    return 0


def cmd_find(args):
    """Search the local collection so you can see what --library will match."""
    rows = library.search_index(args.query, limit=25)
    if rows:
        print(f"index matches for {args.query!r}:")
        for artist, album, n in rows:
            print(f"  {n:>3} tracks   {artist} — {album}")
        print(f'\nuse:  ./run.sh remix --library "{rows[0][0]} - {rows[0][1]}"')
        return 0
    print(f"nothing in the index for {args.query!r}; trying the share directly ...")
    mp = library.ensure_mounted()
    artist, album = library._split_query(args.query)
    tracks = library.album_from_walk(mp, artist, album)
    for t in tracks:
        print(f"  {t['n']:>3}  {t['title']}   {t['rel_path']}")
    if not tracks:
        print("  no match")
    return 0


def cmd_list(args):
    config.ensure_dirs()
    for d in sorted(config.ALBUMS_DIR.glob("*/album.json")):
        m = json.loads(d.read_text())
        pool = config.RETIMED_DIR / m["slug"] / "loops.json"
        nloops = len(json.loads(pool.read_text())) if pool.exists() else 0
        print(f"{m['slug']:<50} {len(m.get('tracks', [])):>3} tracks  "
              f"{nloops:>4} loops   {m.get('artist')} — {m.get('album')}")
    outs = sorted(config.OUT_DIR.glob("distillery_*.wav"))
    if outs:
        print("\nrenders:")
        for p in outs:
            print(f"  {p.name}  {p.stat().st_size / 1e6:.1f} MB")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="distillery",
                                description="Remix a whole album into one 9-minute "
                                            "ambient techno track.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("remix", help="full pipeline")
    src = r.add_argument_group("album source")
    src.add_argument("--library", help='"Artist - Album" from the local collection '
                                       f'on smb://{config.SMB_HOST}/{config.SMB_SHARE} '
                                       '(preferred — real mp3s, not a YouTube rip)')
    src.add_argument("--album", help='"Artist - Album" via MusicBrainz + yt-dlp '
                                     '(fallback; audio quality is worse)')
    src.add_argument("--url", help="playlist/album URL for yt-dlp")
    src.add_argument("--local", help="directory of audio files already on disk")
    src.add_argument("--slug", help="re-use a previously fetched album by slug")
    src.add_argument("--artist", help="artist name when using --local")
    src.add_argument("--album-name", dest="album_name", help="album name when using --local")
    r.add_argument("--limit", type=int, help="only use the first N tracks")
    r.add_argument("--bars", type=int, default=None,
                   help="length in bars (default: whatever makes exactly 9:00 at the "
                        "chosen BPM)")
    r.add_argument("--length", default=None,
                   help="auto (default: size the piece from the material) or a form "
                        "name: sketch (~2:15) / single (~4:00) / extended (~6:00) / "
                        "long (~9:00)")
    r.add_argument("--minutes", type=float, default=None,
                   help="explicit duration in minutes")
    r.add_argument("--exposure", type=float, default=None,
                   help=f"bars each distinct idea gets before something new arrives "
                        f"(default {config.EXPOSURE_BARS:g}); the main length dial")
    r.add_argument("--bpm", default="auto",
                   help="target tempo: 'auto' (default) picks the tempo needing the "
                        "least resampling for this album, or give a number e.g. 128")
    r.add_argument("--seed", type=int, help="arrangement seed (reproducible runs)")
    r.add_argument("--bar-sizes", type=float, nargs="+", default=list(config.LOOP_BAR_SIZES),
                   help="loop lengths in bars (default 0.5 1 2)")
    r.add_argument("--per-size", type=int, default=config.LOOPS_PER_SIZE,
                   help="loops kept per track per size")
    r.add_argument("--mp3", action="store_true", help="also write a 320k mp3")
    r.add_argument("--no-texture", action="store_true",
                   help="skip the melodic texture layer cut from the 'other' stem")
    r.add_argument("--loops-db", type=float, default=None,
                   help=f"drum-loop bus level in dB (default "
                        f"{config.LOOP_LEVEL_DB:g}); drums lead the mix")
    r.add_argument("--texture-db", type=float, default=None,
                   help=f"texture bus level in dB (default "
                        f"{config.TEXTURE_LEVEL_DB:g})")
    r.add_argument("--chords", action="store_true",
                   help="restore the four-chord pad progression (default: a single "
                        "root drone, no chords)")
    r.add_argument("--drone-db", type=float, default=None,
                   help=f"drone bus level in dB (default {config.DRONE_LEVEL_DB:g}; "
                        "lower = further back, e.g. -32)")
    r.add_argument("--workers", type=int, default=None,
                   help="parallel workers per stage (default: auto — one per usable "
                        "core, capped by RAM and any cgroup CPU quota)")
    r.add_argument("--keep-source-loops", action="store_true",
                   help="also keep the pre-retime loops")
    r.add_argument("--force-download", action="store_true")
    r.add_argument("--force-analyze", action="store_true")
    r.add_argument("--force-stems", action="store_true")
    r.add_argument("--force-loops", action="store_true")
    r.set_defaults(func=cmd_remix)

    a = sub.add_parser("rearrange", help="new arrangement from a cached loop pool")
    a.add_argument("slug")
    a.add_argument("--seed", type=int)
    a.add_argument("--bars", type=int, default=None)
    a.add_argument("--length", default=None)
    a.add_argument("--minutes", type=float, default=None)
    a.add_argument("--exposure", type=float, default=None)
    a.add_argument("--no-texture", action="store_true")
    a.add_argument("--texture-db", type=float, default=None)
    a.add_argument("--loops-db", type=float, default=None)
    a.add_argument("--bpm", default=None,
                   help="which tempo pool to arrange (default: the one on disk)")
    a.add_argument("--mp3", action="store_true")
    a.add_argument("--chords", action="store_true")
    a.add_argument("--drone-db", type=float, default=None)
    a.add_argument("--workers", type=int, default=None)
    a.add_argument("--dry-run", action="store_true", help="print the plan, render nothing")
    a.set_defaults(func=cmd_arrange_only)

    b = sub.add_parser("bed", help="render the synthesized bed alone")
    b.add_argument("--key", default="A")
    b.add_argument("--scale", default="minor")
    b.add_argument("--bars", type=int, default=None)
    b.add_argument("--length", default=None)
    b.add_argument("--minutes", type=float, default=None)
    b.add_argument("--bpm", default=None)
    b.add_argument("--seed", type=int, default=1)
    b.add_argument("--chords", action="store_true")
    b.add_argument("--drone-db", type=float, default=None)
    b.add_argument("--workers", type=int, default=None)
    b.set_defaults(func=cmd_bed)

    f = sub.add_parser("find", help="search the local collection for an album")
    f.add_argument("query")
    f.set_defaults(func=cmd_find)

    l = sub.add_parser("list", help="what's cached")
    l.set_defaults(func=cmd_list)

    n = sub.add_parser("nightly", help="pick an album, distil it, post the video")
    n.add_argument("--dry-run", action="store_true",
                   help="render everything, post nothing")
    n.add_argument("--no-post", action="store_true", help="skip posting entirely")
    n.add_argument("--force-bluesky", action="store_true",
                   help="attempt Bluesky even if the video exceeds its duration limit")
    n.add_argument("--pick-seed", type=int, default=None,
                   help="seed the album choice (reproducible picks)")
    n.add_argument("--min-tracks", type=int, default=None)
    n.add_argument("--limit", type=int, default=None)
    n.add_argument("--seed", type=int, default=None)
    n.add_argument("--minutes", type=float, default=None)
    n.add_argument("--length", default=None)
    n.add_argument("--mp3", action="store_true")
    n.add_argument("--workers", type=int, default=None)
    n.add_argument("--drone-db", type=float, default=None)
    n.add_argument("--texture-db", type=float, default=None)
    n.add_argument("--loops-db", type=float, default=None)
    n.set_defaults(func=cmd_nightly)

    po = sub.add_parser("post", help="render a video for a render and post it")
    po.add_argument("wav", nargs="?", help="a wav in data/output (default: newest)")
    po.add_argument("--mp4", help="use this mp4 instead of rendering one")
    po.add_argument("--dry-run", action="store_true")
    po.add_argument("--force-bluesky", action="store_true")
    po.set_defaults(func=cmd_post)

    h = sub.add_parser("history", help="recent nightly runs")
    h.add_argument("--limit", type=int, default=20)
    h.set_defaults(func=cmd_history)

    t = sub.add_parser("selftest", help="run invariant checks (no album needed)")
    t.set_defaults(func=lambda _a: __import__("distillery.selftest",
                                              fromlist=["main"]).main())

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
