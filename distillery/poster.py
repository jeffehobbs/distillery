"""Render a video for a finished distillation and post it.

Each platform is posted independently in its own try/except: one failing never stops
the other, and a failure never costs you the audio, which is already on disk.

**Bluesky only accepts videos under three minutes.** distillery's auto length runs to
6:40, so posting there is conditional on duration — checked before the upload rather
than discovered as a rejection. Mastodon has no duration limit but does have a size
cap, which `masto.py` asks the instance for instead of assuming.
"""
import logging
from pathlib import Path

from . import config, social, video

log = logging.getLogger("distillery")


def make_video(wav_path, meta, info, plan, out_dir=None, crf=None):
    """Render the waveform video next to the wav. Returns the mp4 Path."""
    wav_path = Path(wav_path)
    out_dir = Path(out_dir or config.VIDEO_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    artist = (meta or {}).get("artist") or "unknown"
    album = (meta or {}).get("album") or "unknown"
    key = f"{info.get('key')} {info.get('key_scale')}".strip()
    mins = int(info["duration_s"] // 60)
    secs = int(round(info["duration_s"] % 60))
    title_lines = [artist, album]
    # One line, space-separated: no bullet character, because a display font like
    # Kabel has no glyph for it and ffmpeg renders the missing glyph as a box.
    meta_lines = [f"{info['bpm']:g} BPM   {key}   {mins}:{secs:02d}   "
                  f"{info['songs_used']} songs   {info['loop_events']} loops"]
    mp4 = out_dir / (wav_path.stem + ".mp4")
    return video.build(wav_path, title_lines, meta_lines, mp4, crf=crf)


def post(mp4_path, meta, info, plan, dry_run=False, force_bluesky=False):
    """Post to both platforms. Returns a dict of what happened."""
    mp4_path = Path(mp4_path)
    text = social.post_text(meta, info, plan)
    alt = social.alt_text(meta, info, plan)
    dur = info.get("duration_s") or video.duration_s(mp4_path)
    size_mb = mp4_path.stat().st_size / 1e6
    result = {"text": text, "alt": alt, "duration_s": dur, "size_mb": size_mb,
              "bluesky": None, "mastodon": None}

    print(f"  video: {mp4_path.name}  {size_mb:.1f} MB  {dur:.0f}s")
    print("  post text:")
    for line in text.splitlines():
        print(f"    | {line}")

    # Bluesky: duration gate, checked up front
    too_long = dur >= config.BLUESKY_MAX_VIDEO_S and not force_bluesky
    if too_long:
        result["bluesky"] = (f"skipped: {dur:.0f}s exceeds Bluesky's "
                             f"{config.BLUESKY_MAX_VIDEO_S:.0f}s video limit")
        print(f"  bluesky: {result['bluesky']}")
    elif dry_run:
        result["bluesky"] = "dry-run"
    else:
        from . import bluesky
        try:
            bluesky.post_video(mp4_path, text, alt)
            result["bluesky"] = "posted"
        except Exception as e:              # noqa: BLE001 - never block Mastodon
            result["bluesky"] = f"failed: {type(e).__name__}: {e}"
            log.warning("Bluesky post failed: %s", e)
            print(f"  bluesky: {result['bluesky']}")

    if dry_run:
        result["mastodon"] = "dry-run"
    else:
        from . import masto
        try:
            masto.post_video(mp4_path, text, alt)
            result["mastodon"] = "posted"
        except Exception as e:              # noqa: BLE001
            result["mastodon"] = f"failed: {type(e).__name__}: {e}"
            log.warning("Mastodon post failed: %s", e)
            print(f"  mastodon: {result['mastodon']}")

    for k in ("bluesky", "mastodon"):
        if result[k] in ("posted", "dry-run"):
            print(f"  {k}: {result[k]}")
    return result
