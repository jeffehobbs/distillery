"""Compose the post text and alt text for a distillation."""
from . import config

MAX_POST = 300      # Bluesky's limit; Mastodon allows more but this reads fine


def _mmss(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def post_text(meta, info, plan):
    """The post body: what it is, what it was made from, and how."""
    artist = (meta or {}).get("artist") or "unknown"
    album = (meta or {}).get("album") or "unknown"
    key = f"{info.get('key')} {info.get('key_scale')}".strip()
    lines = [
        f"{artist} — {album}",
        f"distilled: {info['songs_used']} songs → one {_mmss(info['duration_s'])} "
        f"ambient techno track",
        f"{info['bpm']:g} BPM · {key} · {info['loop_events']} loops from "
        f"{info['unique_loops']} cuts",
    ]
    text = "\n".join(lines)
    if len(text) > MAX_POST:
        text = "\n".join(lines[:2])[:MAX_POST]
    return text


def alt_text(meta, info, plan):
    """Image/video description, for people using a screen reader."""
    artist = (meta or {}).get("artist") or "unknown artist"
    album = (meta or {}).get("album") or "unknown album"
    bass = (plan.bass or {}).get("pattern") if plan is not None else None
    bits = [
        f"An audio-visualiser video: a cyan waveform over a dark colour wash, "
        f"captioned with the album details.",
        f"The audio is a {_mmss(info['duration_s'])} ambient techno piece built "
        f"from the drum tracks of {info['songs_used']} songs on "
        f"{artist}'s album {album}.",
        f"It runs at {info['bpm']:g} BPM in {info.get('key')} "
        f"{info.get('key_scale')}, over a synthesized kick, "
        f"{'chord pad' if info.get('chords') else 'single-note drone'}"
        + (f" and a {bass} bassline" if bass else "") + ".",
    ]
    return " ".join(bits)[:1800]
