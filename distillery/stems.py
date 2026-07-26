"""Demucs stem separation — we only need the drums, but the model gives all four.

Whole tracks are separated once and cached under data/stems/<slug>/<stem_name>/,
because Demucs' analysis window means separating a 4-second clip costs nearly as
much as separating the whole song. Runs on MPS when available (~20-30 s/track on
Apple silicon), CPU otherwise.
"""
from pathlib import Path

from . import config

_SEPARATOR = None


def _get_separator(model=config.DEMUCS_MODEL):
    global _SEPARATOR
    if _SEPARATOR is None:
        import torch
        from demucs.api import Separator
        dev = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"  loading Demucs {model} on {dev} ...", flush=True)
        _SEPARATOR = Separator(model=model, device=dev)
    return _SEPARATOR


def separate(path, out_dir, model=config.DEMUCS_MODEL, want=None, force=False):
    """Separate one track. Returns {stem_name: wav_path} for the cached stems."""
    import json
    from demucs.api import save_audio
    want = tuple(want or config.WANT_STEMS)
    out_dir = Path(out_dir)
    fp = config.fingerprint(path)
    marker = out_dir / "source.json"
    have = {p.stem: p for p in out_dir.glob("*.wav")} if out_dir.exists() else {}
    cached_fp = None
    if marker.exists():
        try:
            cached_fp = json.loads(marker.read_text()).get("src_fp")
        except json.JSONDecodeError:
            cached_fp = None
    # only reuse stems that were separated from THIS audio (see config.fingerprint)
    if not force and all(w in have for w in want) and cached_fp == fp:
        return {w: have[w] for w in want}

    out_dir.mkdir(parents=True, exist_ok=True)
    sep = _get_separator(model)
    _origin, stems = sep.separate_audio_file(Path(path))
    paths = {}
    for name, tensor in stems.items():
        if name not in want:
            continue                      # skip writing ~40MB of unused bass/vocals
        p = out_dir / f"{name}.wav"
        save_audio(tensor, str(p), samplerate=sep.samplerate)
        paths[name] = p
    marker.write_text(json.dumps({"src": str(path), "src_fp": fp, "model": model}))
    return paths


def drum_stems(tracks, slug, force=False, want=None):
    """Separate every analyzed track into the stems we keep.

    Demucs computes all four sources whichever ones we ask for, so keeping "other"
    alongside "drums" costs nothing but disk — it was being computed and thrown away.
    "other" is everything that isn't drums/bass/vocals: horns, guitar, keys — the
    melodic meat, which is what the texture layer is cut from.

    `tracks` are analysis dicts; adds "drums_path" (and "other_path" when kept) to
    each and returns the list of tracks that produced a drums stem.
    """
    want = tuple(want or config.WANT_STEMS)
    base = config.STEMS_DIR / slug
    ok = []
    for t in tracks:
        d = base / t["name"]
        try:
            paths = separate(t["path"], d, want=want, force=force)
        except Exception as e:            # noqa: BLE001 - keep the album moving
            print(f"  ! {t['name']}: demucs failed: {type(e).__name__}: {e}")
            continue
        if "drums" not in paths:
            print(f"  ! {t['name']}: no drums stem produced")
            continue
        t["drums_path"] = str(paths["drums"])
        if "other" in paths:
            t["other_path"] = str(paths["other"])
        ok.append(t)
        print(f"  ✓ {t['name']}: " + ", ".join(sorted(paths)) + " stems")
    return ok
