"""Essentia analysis, one JSON per track.

Each track goes through Essentia's full `MusicExtractor` in a **child process**:
Essentia hard-crashes (SIGSEGV, via its bundled libav) on some malformed files,
and that is uncatchable in-process — the essentia-explorer indexer hit this
repeatedly. Isolating each track means one bad file costs us that file, not the
run.

Cached to data/albums/<slug>/analysis/<stem>.json, so re-runs are instant.
"""
import json
import subprocess
import sys
from pathlib import Path

CAMELOT_MAJOR = {'C': '8B', 'C#': '3B', 'Db': '3B', 'D': '10B', 'D#': '5B', 'Eb': '5B',
                 'E': '12B', 'F': '7B', 'F#': '2B', 'Gb': '2B', 'G': '9B', 'G#': '4B',
                 'Ab': '4B', 'A': '11B', 'A#': '6B', 'Bb': '6B', 'B': '1B'}
CAMELOT_MINOR = {'A': '8A', 'A#': '3A', 'Bb': '3A', 'B': '10A', 'C': '5A', 'C#': '12A',
                 'Db': '12A', 'D': '7A', 'D#': '2A', 'Eb': '2A', 'E': '9A', 'F': '4A',
                 'F#': '11A', 'Gb': '11A', 'G': '6A', 'G#': '1A', 'Ab': '1A'}

# Essentia spells keys with sharps C#/F# and flats Eb/Ab/Bb — don't assume all-sharps.
NOTE_SEMITONE = {'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4, 'F': 5,
                 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9, 'A#': 10,
                 'Bb': 10, 'B': 11}


def camelot(key, scale):
    if not key:
        return None
    table = CAMELOT_MINOR if (scale or "").startswith("min") else CAMELOT_MAJOR
    return table.get(key)


def _extract(path):
    """Run MusicExtractor and pull out the descriptors this project needs."""
    import os
    import essentia
    import essentia.standard as es

    essentia.log.warningActive = False
    devnull = os.open(os.devnull, os.O_WRONLY)
    saved = os.dup(2)
    try:                                  # Essentia is chatty on stderr
        os.dup2(devnull, 2)
        pool, _pool_frames = es.MusicExtractor(
            lowlevelStats=["mean", "stdev"],
            rhythmStats=["mean", "stdev"],
            tonalStats=["mean", "stdev"])(str(path))
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)

    def g(name, default=None):
        try:
            return pool[name]
        except Exception:                 # noqa: BLE001 - missing descriptor
            return default

    beats = g("rhythm.beats_position", [])
    key, scale = g("tonal.key_edma.key"), g("tonal.key_edma.scale")
    return {
        "path": str(path),
        "duration_s": float(g("metadata.audio_properties.length", 0.0) or 0.0),
        "bpm": float(g("rhythm.bpm", 0.0) or 0.0),
        "bpm_histogram_first_peak": float(g("rhythm.bpm_histogram_first_peak_bpm.mean", 0.0) or 0.0),
        "beats_count": int(len(beats)),
        "beats_position": [round(float(b), 4) for b in beats],
        "beats_loudness_mean": float(g("rhythm.beats_loudness.mean", 0.0) or 0.0),
        "onset_rate": float(g("rhythm.onset_rate", 0.0) or 0.0),
        "danceability": float(g("rhythm.danceability", 0.0) or 0.0),
        "key": key, "key_scale": scale,
        "key_strength": float(g("tonal.key_edma.strength", 0.0) or 0.0),
        "camelot": camelot(key, scale),
        "chords_key": g("tonal.chords_key"),
        "chords_scale": g("tonal.chords_scale"),
        "tuning_frequency": float(g("tonal.tuning_frequency", 440.0) or 440.0),
        "loudness_lufs": float(g("lowlevel.loudness_ebu128.integrated", 0.0) or 0.0),
        "dynamic_complexity": float(g("lowlevel.dynamic_complexity", 0.0) or 0.0),
        "spectral_centroid_mean": float(g("lowlevel.spectral_centroid.mean", 0.0) or 0.0),
        "spectral_energy_mean": float(g("lowlevel.spectral_energy.mean", 0.0) or 0.0),
        "zerocrossingrate_mean": float(g("lowlevel.zerocrossingrate.mean", 0.0) or 0.0),
        "tag_artist": g("metadata.tags.artist", [None])[0] if g("metadata.tags.artist") else None,
        "tag_title": g("metadata.tags.title", [None])[0] if g("metadata.tags.title") else None,
        "status": "ok",
    }


def analyze_track(path, cache_path, force=False):
    """Analyze one track in an isolated child process; returns the metadata dict."""
    from . import config
    cache_path = Path(cache_path)
    fp = config.fingerprint(path)
    if cache_path.exists() and not force:
        try:
            d = json.loads(cache_path.read_text())
            if d.get("status") == "ok" and d.get("src_fp") == fp:
                return d
        except json.JSONDecodeError:
            pass
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run([sys.executable, "-m", "distillery.analyze",
                           str(path), str(cache_path)],
                          capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not cache_path.exists():
        why = proc.stderr.strip().splitlines()[-1:] or ["no output"]
        status = "segfault" if proc.returncode and proc.returncode < 0 else "error"
        d = {"path": str(path), "status": status, "error": why[0], "src_fp": fp}
        cache_path.write_text(json.dumps(d, indent=2))
        return d
    d = json.loads(cache_path.read_text())
    d["src_fp"] = fp
    cache_path.write_text(json.dumps(d))
    return d


def analyze_album(track_paths, analysis_dir, force=False):
    """Analyze every track in parallel; returns the list of ok metadata dicts.

    Each track is already its own subprocess (for segfault isolation), so going
    parallel is just a matter of having more than one in flight. Threads do the
    supervising — they're blocked on `subprocess.run`, not holding the GIL — and
    results are collected in track order so the log stays readable.

    MusicExtractor on a long track is the memory-hungry part, hence per_worker_gb.
    """
    from . import parallel
    analysis_dir = Path(analysis_dir)
    workers = parallel.worker_count("cpu", n_items=len(track_paths), per_worker_gb=1.2)
    if workers > 1:
        print(f"  {workers} parallel workers ({parallel.describe()})")

    def one(p):
        return analyze_track(p, analysis_dir / f"{Path(p).stem}.json", force=force)

    results = parallel.run(one, list(track_paths), workers)

    out = []
    for i, (p, d) in enumerate(zip(track_paths, results), 1):
        if d.get("status") != "ok":
            print(f"  [{i:02d}] ! {Path(p).name}: {d.get('status')} {d.get('error','')}")
            continue
        d["track_no"] = i
        d["name"] = Path(p).stem
        out.append(d)
        print(f"  [{i:02d}] {Path(p).name}: {d['bpm']:.1f} BPM  "
              f"{d['key']} {d['key_scale']} ({d['camelot']})  "
              f"{d['beats_count']} beats  {d['loudness_lufs']:.1f} LUFS")
    return out


if __name__ == "__main__":                # child-process entry point
    src, dst = sys.argv[1], sys.argv[2]
    data = _extract(src)
    Path(dst).write_text(json.dumps(data))
