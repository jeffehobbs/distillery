"""Cut many small loops out of each track's drum stem, then retime them to the
target tempo (config.TARGET_BPM, set at runtime by --bpm).

The selection method is the one mashup-app inherited from essentia-explorer's
`essexp/loops.py`:

  1. Use the beat grid Essentia already found (`rhythm.beats_position`) — no
     re-detection.
  2. Pick the downbeat phase whose bar-starts are loudest.
  3. Enumerate bar-aligned, non-overlapping windows at each requested bar size.
  4. Score each window for loopability from frame-wise MFCC/energy:
     seamlessness of the wrap (0.45) + internal timbral homogeneity (0.35) +
     tempo steadiness (0.20), each **min-max normalized per track per size** —
     without that, raw tempo cv dominates the ranking.
  5. Gate on level relative to the FULL MIX's loud percentile, so we only take
     loops from where the drums are actually prominent.
  6. Render with a 25 ms equal-power wrap crossfade (the user's A/B verdict: 25 ms
     beats 10 ms and beats a hard cut).

Then each loop is resampled to exactly the target tempo. Resampling, not
phase-vocoding:
transient smearing is worst on percussion, and the pitch shift that comes with a
modest tempo move is inaudible on drums. Half/double time is allowed so a 76 BPM
track doesn't get dragged to a crawl.
"""
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

from . import audio, config

BEATS_PER_BAR = config.BEATS_PER_BAR
W_SEAM, W_HOMO, W_TEMPO = 0.45, 0.35, 0.20


# ---------------------------------------------------------------- frame features

def _frame_features(path, sr=config.SR):
    """Frame-wise MFCC / centroid / energy for a whole file (mono)."""
    import essentia
    import essentia.standard as es
    essentia.log.warningActive = False

    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        mono = es.MonoLoader(filename=str(path), sampleRate=sr)()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

    fsize, hop = 2048, 1024
    w, spec = es.Windowing(type="hann"), es.Spectrum()
    mfcc = es.MFCC(numberCoefficients=13)
    centroid = es.Centroid(range=sr / 2)
    energy = es.Energy()

    times, mfccs, cents, energies = [], [], [], []
    for i, frame in enumerate(es.FrameGenerator(mono, frameSize=fsize, hopSize=hop,
                                               startFromZero=True)):
        s = spec(w(frame))
        _bands, coeffs = mfcc(s)
        mfccs.append(coeffs)
        cents.append(centroid(s))
        energies.append(energy(frame))
        times.append((i * hop + fsize / 2) / sr)
    return (np.array(times), np.array(mfccs), np.array(cents), np.array(energies))


def _frame_energy(path, sr=config.SR):
    """Cheap per-frame energy of the full mix — the stem-prominence reference."""
    import essentia
    import essentia.standard as es
    essentia.log.warningActive = False
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:
        os.dup2(devnull, 2)
        mono = es.MonoLoader(filename=str(path), sampleRate=sr)()
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)
    energy = es.Energy()
    return np.array([energy(f) for f in es.FrameGenerator(
        mono, frameSize=2048, hopSize=1024, startFromZero=True)])


def _pick_phase(beats, times, energies):
    """Downbeat phase (0..3) whose bar-starts are on average loudest."""
    if len(beats) < BEATS_PER_BAR * 2:
        return 0
    idx = np.clip(np.searchsorted(times, beats), 0, len(energies) - 1)
    be = energies[idx]
    best_p, best = 0, -1.0
    for p in range(BEATS_PER_BAR):
        downs = be[p::BEATS_PER_BAR]
        score = float(np.mean(downs)) if len(downs) else 0.0
        if score > best:
            best_p, best = p, score
    return best_p


def _window_metrics(t0, t1, beat_slice, times, mfccs, energies):
    iois = np.diff(beat_slice)
    cv = float(np.std(iois) / np.mean(iois)) if len(iois) and np.mean(iois) else 1.0
    fmask = (times >= t0) & (times < t1)
    fr, en = mfccs[fmask], energies[fmask]
    if len(fr) < 6:
        return None
    # timbral drift across the window (coeff 0 is overall level -> that's rms)
    disp = float(np.mean(np.std(fr[:, 1:], axis=0)))
    # the wrap: timbre at the end should match timbre at the start (~70 ms each side)
    k = max(2, min(4, len(fr) // 4))
    seam = float(np.linalg.norm(fr[:k, 1:].mean(axis=0) - fr[-k:, 1:].mean(axis=0)))
    rms = float(math.sqrt(max(0.0, np.mean(en))))
    return {"cv": cv, "disp": disp, "seam_dist": seam, "rms": rms}


def _normalize_scores(cands):
    """Min-max each raw metric within this candidate set (1 = best), then weight."""
    def inv_norm(vals):
        lo, hi = min(vals), max(vals)
        if hi - lo < 1e-9:
            return [1.0] * len(vals)
        return [(hi - v) / (hi - lo) for v in vals]

    seam_n = inv_norm([c["m"]["seam_dist"] for c in cands])
    homo_n = inv_norm([c["m"]["disp"] for c in cands])
    tempo_n = inv_norm([c["m"]["cv"] for c in cands])
    for c, s, h, t in zip(cands, seam_n, homo_n, tempo_n):
        c["seam"], c["homo"], c["tempo"] = s, h, t
        c["score"] = W_SEAM * s + W_HOMO * h + W_TEMPO * t


# ---------------------------------------------------------------- texture

def _texture(clip, centroid_hz, sr=config.SR):
    """Label a drum loop by onset density and brightness.

    The arranger leans on this: sparse/low loops belong in the ambient opening,
    busy/bright ones in the climax.
    """
    import essentia.standard as es
    m = audio.mono(clip).astype(np.float32)
    dur = max(1e-3, m.size / sr)
    density = 0.0
    if m.size > sr // 4:
        try:
            onsets, rate = es.OnsetRate()(m)
            density = float(len(onsets)) / dur
        except Exception:                 # noqa: BLE001
            density = 0.0
    busy = "busy" if density >= 5.0 else ("mid" if density >= 2.5 else "sparse")
    if centroid_hz < 1400:
        bright = "low"
    elif centroid_hz > 4200:
        bright = "bright"
    else:
        bright = "mid"
    return f"{busy}-{bright}", density


def detect_key(clip, sr=config.SR):
    """(key, scale, strength) for a rendered loop.

    Run AFTER retiming on purpose: retiming is plain resampling, so it shifts pitch
    along with tempo. Detecting the key on the source would describe a loop that no
    longer exists — this measures what will actually be heard.
    """
    import essentia.standard as es
    m = np.ascontiguousarray(audio.mono(clip).astype(np.float32))
    if m.size < sr // 4:
        return None, None, 0.0
    try:
        key, scale, strength = es.KeyExtractor()(m)
        return key, scale, float(strength)
    except Exception:                     # noqa: BLE001 - odd clips can throw
        return None, None, 0.0


# ---------------------------------------------------------------- retiming

def retimed_dir(slug):
    """Where loops retimed to the CURRENT target tempo live.

    The tempo is part of the path: a pool cut for 120 BPM is useless at 128, and
    keying only on slug meant a BPM change silently reused loops at the old tempo
    under a bed at the new one — everything out of sync, with no warning.
    """
    return config.RETIMED_DIR / slug / f"{config.TARGET_BPM:g}bpm"


def texture_index(slug):
    return retimed_dir(slug) / "texture.json"


def pool_index(slug):
    """Path to the loop index for the current tempo (legacy 120 BPM path honoured)."""
    new = retimed_dir(slug) / "loops.json"
    if not new.exists() and abs(config.TARGET_BPM - 120.0) < 1e-9:
        legacy = config.RETIMED_DIR / slug / "loops.json"
        if legacy.exists():
            return legacy
    return new


def count_distinct_ideas(pool, max_k=None, elbow=0.12, seed=1):
    """How many genuinely distinct drum ideas this pool holds.

    Pool *size* is the wrong measure of material: 394 Bumps loops are mostly
    variations on a dozen grooves, and length shouldn't scale with near-duplicates.
    So the loops are clustered on what actually distinguishes them — onset density,
    brightness, length, loopability — and k grows until another cluster stops paying
    for itself (marginal inertia drop under `elbow`).

    Measured on the three albums built so far, this lands around 12 for Dummy and
    Bumps, and higher for On The Corner, which really is the more varied record.
    """
    from scipy.cluster.vq import kmeans2, whiten
    n = len(pool)
    if n < 8:
        return max(1, n)
    X = np.array([[l["onset_density"],
                   math.log2(max(80.0, l.get("centroid_hz") or 80.0)),
                   math.log2(max(0.25, l["out_bars"])),
                   l["loopability"]] for l in pool], dtype=float)
    keep = X.std(axis=0) > 1e-9         # whiten() divides by std; drop flat columns
    X = X[:, keep]
    if X.shape[1] == 0:
        return 1
    Xw = whiten(X)
    cap = int(min(max_k or 28, max(4, n // 4)))
    inertia0, prev = None, None
    for k in range(2, cap + 1, 2):
        cent, lab = kmeans2(Xw, k, minit="++", seed=seed)
        inertia = float(sum(np.sum((Xw[lab == i] - cent[i]) ** 2)
                            for i in range(k)))
        if inertia0 is None:
            inertia0 = inertia
        # Two stopping rules are needed. The relative one (marginal improvement has
        # dried up) is what fires on real pools. The absolute one catches the case
        # where clusters are essentially pure: once inertia is ~0, every further
        # split still looks like a huge *relative* win (1e-4 -> 5e-5 reads as 50%
        # better) and a relative-only test runs away to the cap.
        if inertia0 > 0 and inertia <= 0.05 * inertia0:
            return k
        if prev is not None and prev > 0 and (prev - inertia) / prev < elbow:
            return max(2, k - 2)
        prev = inertia
    return cap


def choose_target_bpm(track_bpms, candidates=None):
    """Pick the target tempo that needs the least resampling for this album.

    Retiming is plain resampling, so every tempo move is also a pitch move. Cost is
    the total |log2(rate)| over the album using the very same half/double rate
    choice `retime()` will make, so this measures the actual pitch shift the album
    is about to get. Ties break toward 120.

    On an album with two tempo clusters this matters: On The Corner sits at ~115 and
    ~137, so forcing 120 drags the fast half down about 2.7 semitones while the slow
    half barely moves; 128 splits the difference.
    """
    bpms = [float(b) for b in track_bpms if b and float(b) > 0]
    if not bpms:
        return config.TARGET_BPM
    best, best_key = None, None
    for t in (candidates or config.EXACT_BPMS):
        rates = [choose_rate(b, target_bpm=t)[0] for b in bpms]
        shifts = [abs(math.log2(r)) for r in rates]
        # signed mean: positive means the album is being sped UP, which brightens it.
        # Resampling down darkens everything, so break near-ties upward.
        signed = sum(math.log2(r) for r in rates) / len(rates)
        # Minimize the WORST shift first, then the mean. Summed cost alone is nearly
        # flat for an album with two tempo clusters — moving toward one cluster moves
        # away from the other by the same amount — so it can't tell 120 from 128 and
        # falls back on the tie-break, leaving one cluster badly stretched. Minimax
        # is what stops any single track getting mangled.
        # rounded to 2 dp so genuinely close tempos count as tied and the
        # brighter-direction preference gets to decide
        key = (round(max(shifts), 2), round(sum(shifts) / len(shifts), 2),
               -signed, abs(t - 120.0))
        if best_key is None or key < best_key:
            best, best_key = float(t), key
    return best


def semitones_for(track_bpms, target_bpm):
    """Mean / worst pitch shift in semitones the album gets at this target."""
    st = [abs(12 * math.log2(choose_rate(b, target_bpm=target_bpm)[0]))
          for b in track_bpms if b and float(b) > 0]
    if not st:
        return 0.0, 0.0
    return sum(st) / len(st), max(st)


def choose_rate(loop_bpm, target_bpm=None):
    """Resample rate to bring `loop_bpm` to the target, allowing half/double time.

    Returns (rate, beat_multiplier): playing the loop at `rate` makes it
    `beats * beat_multiplier` beats long at the target tempo.
    """
    target_bpm = config.TARGET_BPM if target_bpm is None else float(target_bpm)
    if loop_bpm <= 0:
        return 1.0, 1.0
    options = [(target_bpm / loop_bpm, 1.0),            # straight
               (2 * target_bpm / loop_bpm, 0.5),        # double-time (half as long)
               (0.5 * target_bpm / loop_bpm, 2.0)]      # half-time (twice as long)
    return min(options, key=lambda o: abs(math.log(o[0])))


def retime(clip, beats, loop_bpm, target_bpm=None, sr=config.SR):
    """Resample a loop to the target tempo and snap it to an exact beat length."""
    target_bpm = config.TARGET_BPM if target_bpm is None else float(target_bpm)
    rate, mult = choose_rate(loop_bpm, target_bpm)
    out_beats = beats * mult
    if abs(out_beats - round(out_beats)) > 1e-6 or round(out_beats) < 1:
        rate, mult = target_bpm / loop_bpm, 1.0        # fall back to straight time
        out_beats = beats
    out_beats = int(round(out_beats))
    stretched = audio.speed(clip, rate)
    n_target = int(round(out_beats * (60.0 / target_bpm) * sr))
    return audio.fit_length(stretched, n_target), out_beats, rate


# ---------------------------------------------------------------- extraction

def extract_track_loops(track, slug, bar_sizes=config.LOOP_BAR_SIZES,
                        per_size=config.LOOPS_PER_SIZE, xfade_ms=config.XFADE_MS,
                        keep_source_loops=False, stem="drums", kind="drum",
                        detect_keys=False):
    """Extract + retime loops from one stem of one track. Returns loop dicts.

    `stem` selects which demucs stem to cut from ("drums", or "other" for the
    texture layer); `detect_keys` runs key detection on each retimed loop, which the
    texture layer needs in order to transpose loops into the piece's key.
    """
    beats = np.asarray(track.get("beats_position") or [], dtype=float)
    if beats.size < 4:
        print(f"  ! {track['name']}: no usable beat grid")
        return []

    stem_path = track.get(f"{stem}_path")
    if not stem_path:
        print(f"  ! {track['name']}: no {stem} stem")
        return []
    times, mfccs, cents, energies = _frame_features(stem_path)
    full_e = _frame_energy(track["path"])
    ref = float(np.sqrt(np.percentile(full_e, 90))) if full_e.size else 0.0
    gate = 0.10 * ref                     # stem must be prominent, not just present
    phase = _pick_phase(beats, times, energies)

    stem_audio = audio.read(stem_path)
    out_dir = retimed_dir(slug)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_dir = config.LOOPS_DIR / slug
    if keep_source_loops:
        src_dir.mkdir(parents=True, exist_ok=True)

    made = []
    for bars in bar_sizes:
        step = max(1, int(round(bars * BEATS_PER_BAR)))
        cands = []
        start = phase
        while start + step < beats.size:
            t0, t1 = float(beats[start]), float(beats[start + step])
            m = _window_metrics(t0, t1, beats[start:start + step + 1],
                                times, mfccs, energies)
            if m is not None and m["rms"] >= gate:
                cands.append({"t0": t0, "t1": t1, "m": m, "beats": step})
            start += step
        if not cands:
            continue
        _normalize_scores(cands)
        cands.sort(key=lambda c: c["score"], reverse=True)

        for c in cands[:per_size]:
            dur = c["t1"] - c["t0"]
            if dur <= 0.05:
                continue
            loop_bpm = c["beats"] * 60.0 / dur
            n_loop = int(round(dur * config.SR))
            n_x = int(round(xfade_ms * config.SR / 1000.0))
            i0 = int(round(c["t0"] * config.SR))
            raw = stem_audio[i0:i0 + n_loop + n_x + 64]
            if raw.shape[0] < n_loop // 2:
                continue
            clip = audio.xfade_wrap(raw, n_loop, n_x)

            cseg = cents[(times >= c["t0"]) & (times < c["t1"])]
            cmean = float(np.mean(cseg)) if cseg.size else 0.0
            tex, density = _texture(clip, cmean)

            retimed, out_beats, rate = retime(clip, c["beats"], loop_bpm)
            # re-wrap after resampling: fit_length nudges the tail, so refresh the seam
            retimed = audio.xfade_wrap(
                np.vstack([retimed, retimed[:n_x + 8]]), retimed.shape[0], n_x)
            lvl = audio.dbfs(retimed)
            if not np.isfinite(lvl) or lvl < -50:
                continue                  # empty/near-silent stem window

            tag = "" if stem == "drums" else f"{stem}_"
            name = (f"{track['track_no']:02d}_{tag}{bars:g}bar_{out_beats}b_"
                    f"{tex}_{c['t0']:06.1f}s_l{c['score']:.3f}")
            wav = out_dir / f"{name}.wav"
            audio.write(wav, audio.normalize_rms(retimed, config.LOOP_TARGET_DBFS))
            if keep_source_loops:
                audio.write(src_dir / f"{name}_src.wav", clip)

            lkey, lscale, lstrength = (detect_key(retimed) if detect_keys
                                       else (None, None, 0.0))
            made.append({
                "id": f"{slug}:{name}", "track_no": track["track_no"],
                "kind": kind, "stem": stem,
                "loop_key": lkey, "loop_scale": lscale,
                "loop_key_strength": round(lstrength, 4),
                "src_fp": config.fingerprint(track["path"]),
                "target_bpm": config.TARGET_BPM,
                "track_name": track["name"], "track_title": track.get("tag_title"),
                "wav": str(wav), "src_bars": bars, "src_beats": c["beats"],
                "src_start_s": round(c["t0"], 3), "src_dur_s": round(dur, 3),
                "src_bpm": round(loop_bpm, 2), "track_bpm": round(track["bpm"], 2),
                "rate": round(rate, 5), "out_beats": out_beats,
                "out_bars": out_beats / BEATS_PER_BAR,
                "out_dur_s": round(retimed.shape[0] / config.SR, 4),
                "loopability": round(c["score"], 4), "seam": round(c["seam"], 4),
                "homogeneity": round(c["homo"], 4), "tempo_stability": round(c["tempo"], 4),
                # level of the cut BEFORE the file was RMS-normalized to
                # LOOP_TARGET_DBFS -- i.e. how loud the drums were in the source
                "rms": round(c["m"]["rms"], 6), "dbfs_raw": round(lvl, 2),
                "centroid_hz": round(cmean, 1), "onset_density": round(density, 2),
                "texture": tex, "key": track.get("key"), "camelot": track.get("camelot"),
            })
    made.sort(key=lambda d: -d["loopability"])
    print(f"  {track['name']}: {len(made)} {kind} loops "
          f"(bpm {track['bpm']:.1f} -> {config.TARGET_BPM:g}, sizes {', '.join(f'{b:g}' for b in bar_sizes)})")
    return made


def _extract_parallel(tracks, slug, kw, label="loop"):
    """Cut loops for several tracks at once, one subprocess per track.

    Subprocesses rather than a process pool: Essentia can SIGSEGV on bad audio, and
    a dead worker takes a whole `ProcessPoolExecutor` down with it (the failure mode
    the essentia-explorer indexer had to be rewritten to avoid). Here a crashed
    child costs exactly that track. Threads supervise, blocked on the child.

    Each track writes loop wavs under its own `NN_` filename prefix, so parallel
    workers never contend for a path.
    """
    from . import parallel
    workers = parallel.worker_count("cpu", n_items=len(tracks), per_worker_gb=1.5)
    if workers > 1:
        print(f"  {workers} parallel workers ({parallel.describe()})")

    def one(track):
        with tempfile.TemporaryDirectory(dir=str(config.TMP_DIR)) as td:
            job = Path(td) / "job.json"
            out = Path(td) / "out.json"
            # The tempo MUST travel with the job: a worker is a fresh interpreter
            # that re-imports config at its default, so a parent-side set_tempo()
            # does not reach it. Without this the workers cut loops at 120 BPM while
            # the bed played at the chosen tempo — silently out of sync.
            job.write_text(json.dumps({"track": track, "slug": slug, "kw": kw,
                                       "out": str(out),
                                       "target_bpm": config.TARGET_BPM}))
            proc = parallel.run_subprocess(["-m", "distillery.loops", str(job)])
            if proc.returncode != 0 or not out.exists():
                why = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["no output"]
                kind = "segfault" if proc.returncode and proc.returncode < 0 else "failed"
                return {"error": f"{kind}: {why[0]}", "loops": []}
            return {"error": None, "loops": json.loads(out.read_text()),
                    "log": proc.stdout.strip()}

    made = []
    for track, res in zip(tracks, parallel.run(one, list(tracks), workers)):
        if res["error"]:
            print(f"  ! {track['name']}: {label} extraction {res['error']}")
            continue
        if res.get("log"):
            print(res["log"])
        made.extend(res["loops"])
    return made


def extract_album_texture(tracks, slug, force=False, bar_sizes=None, per_size=None):
    """Cut the texture pool from the "other" stem (horns/guitar/keys).

    Cached separately from the drum pool, and each loop carries the key detected on
    its retimed audio so the arranger can transpose it into the piece's key.
    """
    index = texture_index(slug)
    have = [t for t in tracks if t.get("other_path")]
    if not have:
        return []
    fps = {t["track_no"]: config.fingerprint(t["path"]) for t in have}
    cached = []
    if index.exists() and not force:
        try:
            cached = [l for l in json.loads(index.read_text())
                      if Path(l["wav"]).exists()
                      and l.get("src_fp") == fps.get(l["track_no"])]
        except (json.JSONDecodeError, KeyError, TypeError):
            cached = []
    done = {l["track_no"] for l in cached}
    todo = [t for t in have if t["track_no"] not in done]
    if cached:
        print(f"  cached: {len(cached)} texture loops from {len(done)} tracks")
    new = []
    if todo:
        new = _extract_parallel(todo, slug, {
            "bar_sizes": list(bar_sizes or config.TEXTURE_BAR_SIZES),
            "per_size": int(per_size or config.TEXTURE_PER_SIZE),
            "stem": "other", "kind": "texture", "detect_keys": True}, label="texture")
    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(cached + new, indent=2))
    want = {t["track_no"] for t in tracks}
    return [l for l in cached + new if l["track_no"] in want]


def extract_album_loops(tracks, slug, force=False, **kw):
    """Extract loops for every track, caching the loop index as JSON.

    The cache is per track, not per run: the index on disk holds every loop ever
    cut for this album, and only tracks missing from it get processed. That way
    adding songs to an album (or a first run with --limit) doesn't silently reuse a
    pool built from a different track set, and doesn't redo the work either.
    """
    index = pool_index(slug)
    cached = []
    if index.exists() and not force:
        try:
            cached = [l for l in json.loads(index.read_text())
                      if Path(l["wav"]).exists()]
        except (json.JSONDecodeError, KeyError, TypeError):
            cached = []
    # a track counts as "done" only if its cached loops came from this same audio
    fps = {t["track_no"]: config.fingerprint(t["path"]) for t in tracks}
    stale = {l["track_no"] for l in cached
             if l["track_no"] in fps and l.get("src_fp") != fps[l["track_no"]]}
    if stale:
        print(f"  source audio changed for {len(stale)} track(s) — recutting their loops")
        cached = [l for l in cached if l["track_no"] not in stale]
    have = {l["track_no"] for l in cached}
    todo = [t for t in tracks if t["track_no"] not in have]
    if cached:
        reused = len({l['track_no'] for l in cached} & {t['track_no'] for t in tracks})
        print(f"  cached: {len(cached)} loops from {len(have)} tracks "
              f"({reused} of this run's tracks already done)")

    new_loops = []
    if todo:
        new_loops = _extract_parallel(todo, slug, kw)

    index.parent.mkdir(parents=True, exist_ok=True)
    index.write_text(json.dumps(cached + new_loops, indent=2))
    want = {t["track_no"] for t in tracks}
    return [l for l in cached + new_loops if l["track_no"] in want]


if __name__ == "__main__":            # child-process entry: cut one track's loops
    _job = json.loads(Path(sys.argv[1]).read_text())
    if _job.get("target_bpm"):         # adopt the parent's tempo before cutting
        config.set_tempo(_job["target_bpm"])
    _made = extract_track_loops(_job["track"], _job["slug"], **_job["kw"])
    Path(_job["out"]).write_text(json.dumps(_made))
