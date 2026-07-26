"""Get an album's worth of audio onto disk.

Three ways in, all landing in data/albums/<slug>/ as NN_Title.<ext>:

  1. `--album "Artist - Album"`  — look the release up on MusicBrainz to get the
     canonical tracklist, then find each track individually with yt-dlp. Going
     track-by-track (rather than grabbing one hour-long "full album" upload)
     means we get real track boundaries, correct titles, and a duration to match
     against, which is what keeps us from downloading a live cover by mistake.
  2. `--url <playlist|album url>` — hand the URL straight to yt-dlp.
  3. `--local <dir>` — use files that are already on disk (no network).
"""
import json
import re
import subprocess
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path

from . import config

MB_API = "https://musicbrainz.org/ws/2"
AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".opus", ".ogg", ".aiff", ".aif", ".wma"}


def slugify(s, maxlen=60):
    s = unicodedata.normalize("NFKD", s or "")
    s = re.sub(r"[^\w\s-]", "", s).strip()
    s = re.sub(r"[\s_-]+", "-", s)
    return s[:maxlen].strip("-") or "album"


def _sanitize_filename(s, maxlen=70):
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[/\\:*?\"<>|]+", "_", s).strip()
    return re.sub(r"\s+", " ", s)[:maxlen] or "track"


# ------------------------------------------------------------------ musicbrainz

def _mb_get(path, params):
    url = f"{MB_API}/{path}?" + urllib.parse.urlencode({**params, "fmt": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": config.USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def musicbrainz_tracklist(query):
    """Resolve "Artist - Album" (or just an album name) to a tracklist.

    Returns (artist, album, [{"position", "title", "seconds"}, ...]).
    """
    artist, album = None, query
    for sep in (" - ", " – ", " — ", ": "):
        if sep in query:
            artist, album = [p.strip() for p in query.split(sep, 1)]
            break

    lucene = f'release:"{album}"'
    if artist:
        lucene += f' AND artist:"{artist}"'
    data = _mb_get("release", {"query": lucene, "limit": 12})
    releases = data.get("releases") or []
    if not releases:
        raise RuntimeError(f"MusicBrainz found no release for {query!r}")

    # Prefer a release we can actually read a tracklist from; official albums
    # first, then whatever scored highest.
    def rank(r):
        official = 0 if (r.get("status") == "Official") else 1
        primary = 0 if ((r.get("release-group") or {}).get("primary-type") == "Album") else 1
        return (official, primary, -int(r.get("score") or 0))

    for rel in sorted(releases, key=rank):
        full = _mb_get(f"release/{rel['id']}", {"inc": "recordings+artist-credits"})
        media = full.get("media") or []
        tracks = []
        for m in media:
            for t in m.get("tracks") or []:
                length = t.get("length") or (t.get("recording") or {}).get("length")
                tracks.append({
                    "position": len(tracks) + 1,
                    "title": t.get("title") or (t.get("recording") or {}).get("title") or "",
                    "seconds": (length / 1000.0) if length else None,
                })
        if tracks:
            credit = full.get("artist-credit") or []
            got_artist = artist or (credit[0].get("name") if credit else None) or "Unknown"
            return got_artist, full.get("title") or album, tracks
    raise RuntimeError(f"MusicBrainz release found but no tracklist for {query!r}")


# ------------------------------------------------------------------ yt-dlp

def _ytdlp_search(query, n=5):
    """Return candidate entries (id/title/duration) for a search query."""
    out = subprocess.run(
        ["yt-dlp", "--no-warnings", "--flat-playlist", "-J", f"ytsearch{n}:{query}"],
        capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        return []
    try:
        data = json.loads(out.stdout)
    except json.JSONDecodeError:
        return []
    return [e for e in (data.get("entries") or []) if e.get("id")]


_BAD_WORDS = ("live", "cover", "remix", "reaction", "karaoke", "instrumental",
              "lyrics video", "tutorial", "sped up", "slowed", "8d", "nightcore")


def _pick_candidate(entries, want_seconds, title):
    """Score search hits by duration match, then by title sanity."""
    best, best_score = None, -1e9
    for e in entries:
        dur = e.get("duration") or 0
        if not dur or dur > 1800:
            continue
        score = 0.0
        if want_seconds:
            err = abs(dur - want_seconds) / max(1.0, want_seconds)
            if err > 0.35:
                continue
            score -= err * 100
        name = (e.get("title") or "").lower()
        for w in _BAD_WORDS:
            if w in name and w not in (title or "").lower():
                score -= 25
        if (title or "").lower().split("(")[0].strip() in name:
            score += 10
        if score > best_score:
            best, best_score = e, score
    return best


def _download_id(video_id, out_base, audio_format="mp3"):
    subprocess.run([
        "yt-dlp", "--no-warnings", "--no-playlist",
        "-x", "--audio-format", audio_format, "--audio-quality", "0",
        "-o", f"{out_base}.%(ext)s",
        f"https://www.youtube.com/watch?v={video_id}",
    ], check=True, timeout=900)
    hits = sorted(Path(out_base).parent.glob(Path(out_base).name + ".*"))
    return hits[0] if hits else None


def download_album(query, audio_format="mp3", limit=None, force=False):
    """MusicBrainz tracklist -> one yt-dlp search+download per track."""
    artist, album, tracks = musicbrainz_tracklist(query)
    if limit:
        tracks = tracks[:limit]
    slug = slugify(f"{artist}-{album}")
    dest = config.ALBUMS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Album: {artist} — {album}  ({len(tracks)} tracks)  -> {dest}")

    got = []
    for t in tracks:
        base = dest / f"{t['position']:02d}_{_sanitize_filename(t['title'])}"
        existing = sorted(p for p in dest.glob(base.name + ".*")
                          if p.suffix.lower() in AUDIO_EXTS)
        if existing and not force:
            print(f"  [{t['position']:02d}] cached: {existing[0].name}")
            got.append(existing[0])
            continue
        q = f"{artist} {t['title']}"
        cands = _ytdlp_search(f"{q} audio", n=6) or _ytdlp_search(q, n=6)
        pick = _pick_candidate(cands, t.get("seconds"), t["title"])
        if not pick:
            print(f"  [{t['position']:02d}] !! no usable match for {t['title']!r}")
            continue
        print(f"  [{t['position']:02d}] {t['title']} <- {pick.get('title')} "
              f"({pick.get('duration')}s)")
        try:
            path = _download_id(pick["id"], str(base), audio_format)
        except subprocess.CalledProcessError as e:
            print(f"       !! download failed: {e}")
            continue
        if path:
            got.append(path)

    meta = {"artist": artist, "album": album, "slug": slug, "source": "musicbrainz+yt-dlp",
            "query": query, "tracks": [p.name for p in got]}
    (dest / "album.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(got)}/{len(tracks)} tracks on disk.")
    return dest, meta


def download_url(url, audio_format="mp3", limit=None):
    """Download a playlist/album URL wholesale with yt-dlp."""
    info = subprocess.run(["yt-dlp", "--no-warnings", "--flat-playlist", "-J", url],
                          capture_output=True, text=True, timeout=300)
    title, uploader = "album", "Unknown"
    if info.returncode == 0:
        try:
            d = json.loads(info.stdout)
            title = d.get("title") or title
            uploader = d.get("uploader") or d.get("channel") or uploader
        except json.JSONDecodeError:
            pass
    slug = slugify(f"{uploader}-{title}")
    dest = config.ALBUMS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["yt-dlp", "--no-warnings", "-x", "--audio-format", audio_format,
           "--audio-quality", "0", "-o",
           str(dest / "%(playlist_index)02d_%(title)s.%(ext)s"), url]
    if limit:
        cmd += ["--playlist-end", str(limit)]
    subprocess.run(cmd, check=True)
    files = local_tracks(dest)
    meta = {"artist": uploader, "album": title, "slug": slug, "source": url,
            "query": url, "tracks": [p.name for p in files]}
    (dest / "album.json").write_text(json.dumps(meta, indent=2))
    return dest, meta


def local_tracks(d):
    return sorted(p for p in Path(d).iterdir()
                  if p.suffix.lower() in AUDIO_EXTS and not p.name.startswith("."))


def use_local(directory, artist=None, album=None):
    """Register an existing folder of audio as the album (no download)."""
    d = Path(directory).expanduser().resolve()
    files = local_tracks(d)
    if not files:
        raise RuntimeError(f"no audio files in {d}")
    album = album or d.name
    artist = artist or "Various"
    slug = slugify(f"{artist}-{album}")
    dest = config.ALBUMS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)
    linked = []
    for i, f in enumerate(files, 1):
        target = dest / f"{i:02d}_{_sanitize_filename(f.stem)}{f.suffix.lower()}"
        if not target.exists():
            try:
                target.symlink_to(f)
            except OSError:
                import shutil
                shutil.copy2(f, target)
        linked.append(target)
    meta = {"artist": artist, "album": album, "slug": slug, "source": str(d),
            "query": str(d), "tracks": [p.name for p in linked]}
    (dest / "album.json").write_text(json.dumps(meta, indent=2))
    print(f"Using local album: {artist} — {album} ({len(linked)} tracks) -> {dest}")
    return dest, meta
