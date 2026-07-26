"""Pull an album from a local music collection on an SMB share, rather than from
YouTube (which sounds worse and needs no involvement if you own the record).

Resolving "Artist - Album" to files:

  1. If an index database is configured (see config.ESSENTIA_DB), query it — the
     collection is already tagged there, so this is instant. Walking tens of
     thousands of files over SMB takes minutes.
  2. Otherwise walk the share for an `<Artist>/<Album>/` directory, reading track
     numbers with ffprobe.

Matched files are **copied** into data/albums/<slug>/ as NN_Title.ext rather than
symlinked, so Essentia and Demucs read from local disk (each reads the file more
than once) and the run survives the share unmounting mid-way.

Unicode gotcha, the same one that broke mashup-app: macOS records these paths
decomposed (NFD) and other layers hand back composed (NFC), so an accented path
can exist and still fail `os.path.exists`. Every lookup here tries the path
as-is, then NFC, then NFD.
"""
import os
import re
import shutil
import sqlite3
import subprocess
import unicodedata
from pathlib import Path

from . import config
from .download import AUDIO_EXTS, _sanitize_filename, slugify


# ---------------------------------------------------------------- mounting

def _readable(p):
    try:
        return Path(p).is_dir() and any(True for _ in os.scandir(p))
    except (OSError, PermissionError):
        return False


def mountpoint():
    """Return the mounted share, or None."""
    for c in config.MOUNT_CANDIDATES:
        if _readable(c):
            return Path(c)
    return None


def ensure_mounted(verbose=True):
    """Find the share, mounting it with mount_smbfs if it isn't already up."""
    mp = mountpoint()
    if mp:
        return mp

    s = config.secrets()
    host = s.get("SMB_HOST", config.SMB_HOST)
    share = s.get("SMB_SHARE", config.SMB_SHARE)
    user, pw = s.get("SMB_USER"), s.get("SMB_PASSWORD")
    if not user or not pw:
        raise RuntimeError(
            f"{config.SMB_SHARE} share is not mounted and no credentials found.\n"
            f"Either mount it in Finder (smb://{host}/{share}) or put\n"
            f"  SMB_USER=…\n  SMB_PASSWORD=…\n"
            f"in {config.SECRETS_FILE} (chmod 600).")

    target = config.FALLBACK_MOUNTPOINT
    target.mkdir(parents=True, exist_ok=True)
    # percent-encode: passwords routinely contain $ ! @ / which break the URL
    from urllib.parse import quote
    url = f"//{quote(user, safe='')}:{quote(pw, safe='')}@{host}/{share}"
    if verbose:
        print(f"  mounting //{user}@{host}/{share} -> {target}", flush=True)
    proc = subprocess.run(["/sbin/mount_smbfs", url, str(target)],
                          capture_output=True, text=True, timeout=120)
    if not _readable(target):
        raise RuntimeError(f"mount failed: {(proc.stderr or proc.stdout).strip()}")
    return target


# ---------------------------------------------------------------- path helpers

def resolve_path(p):
    """Return an existing path for p, trying as-is then NFC then NFD."""
    cands = [str(p), unicodedata.normalize("NFC", str(p)),
             unicodedata.normalize("NFD", str(p))]
    for c in cands:
        if os.path.exists(c):
            return Path(c)
    return None


def _tracknum(value):
    """'3/11' or '03' or None -> int (0 when unknown)."""
    if not value:
        return 0
    m = re.match(r"\s*(\d+)", str(value))
    return int(m.group(1)) if m else 0


def _ffprobe_tags(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries",
         "format_tags=title,artist,album,track", "-of", "default=nw=1", str(path)],
        capture_output=True, text=True, timeout=60)
    tags = {}
    for line in out.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            tags[k.replace("TAG:", "").replace("format_tags=", "").lower()] = v.strip()
    return tags


# ---------------------------------------------------------------- lookup

def _split_query(query):
    for sep in (" - ", " – ", " — ", ": "):
        if sep in query:
            a, b = query.split(sep, 1)
            return a.strip(), b.strip()
    return None, query.strip()


def search_index(query, limit=25):
    """Search the essentia-explorer index for matching (artist, album) pairs."""
    if not config.ESSENTIA_DB.exists():
        return []
    artist, album = _split_query(query)
    con = sqlite3.connect(f"file:{config.ESSENTIA_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    sql = ("SELECT tag_artist, tag_album, COUNT(*) n FROM tracks "
           "WHERE status='ok' AND tag_album IS NOT NULL AND tag_album LIKE ?")
    params = [f"%{album}%"]
    if artist:
        sql += " AND tag_artist LIKE ?"
        params.append(f"%{artist}%")
    sql += " GROUP BY tag_artist, tag_album ORDER BY n DESC LIMIT ?"
    params.append(limit)
    rows = [(r["tag_artist"], r["tag_album"], r["n"]) for r in con.execute(sql, params)]
    con.close()
    return rows


def album_from_index(artist, album):
    """Exact-match an album in the index; returns ordered track dicts."""
    con = sqlite3.connect(f"file:{config.ESSENTIA_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT path, rel_path, tag_title, tag_tracknumber FROM tracks "
        "WHERE status='ok' AND tag_artist=? AND tag_album=?", (artist, album)).fetchall()
    con.close()
    tracks = [{"path": r["path"], "rel_path": r["rel_path"],
               "title": r["tag_title"] or Path(r["path"]).stem,
               "n": _tracknum(r["tag_tracknumber"])} for r in rows]
    # unnumbered tags fall back to filename order rather than scrambling the album
    tracks.sort(key=lambda t: (t["n"] or 10 ** 6, t["path"]))
    return tracks


def album_from_walk(mp, artist, album, verbose=True):
    """Fallback: find <Artist>/<Album>/ on the share and read tags with ffprobe."""
    def norm(s):
        return unicodedata.normalize("NFC", (s or "").lower())

    want_album, want_artist = norm(album), norm(artist) if artist else None
    hits = []
    for entry in os.scandir(mp):
        if not entry.is_dir():
            continue
        if want_artist and want_artist not in norm(entry.name):
            continue
        try:
            for sub in os.scandir(entry.path):
                if sub.is_dir() and want_album in norm(sub.name):
                    hits.append(Path(sub.path))
        except (OSError, PermissionError):
            continue
    if not hits:
        return []
    d = hits[0]
    if verbose:
        print(f"  found on share: {d}")
    files = sorted(p for p in d.iterdir() if p.suffix.lower() in AUDIO_EXTS)
    tracks = []
    for f in files:
        tags = _ffprobe_tags(f)
        tracks.append({"path": str(f), "rel_path": str(f.relative_to(mp)),
                       "title": tags.get("title") or f.stem,
                       "n": _tracknum(tags.get("track"))})
    tracks.sort(key=lambda t: (t["n"] or 10 ** 6, t["path"]))
    return tracks


# ---------------------------------------------------------------- fetch

def fetch_album(query, limit=None, force=False, verbose=True):
    """Copy an album out of the local collection into data/albums/<slug>/."""
    mp = ensure_mounted(verbose=verbose)
    artist, album = _split_query(query)

    tracks, src = [], None
    if config.ESSENTIA_DB.exists():
        matches = search_index(query, limit=8)
        if matches:
            if verbose and len(matches) > 1:
                print("  index matches:")
                for a, al, n in matches:
                    print(f"    {a} — {al}  ({n} tracks)")
            artist, album = matches[0][0], matches[0][1]
            tracks = album_from_index(artist, album)
            src = "essentia-explorer index"
    if not tracks:
        tracks = album_from_walk(mp, artist, album, verbose=verbose)
        src = "share walk"
    if not tracks:
        raise RuntimeError(f"no album matching {query!r} in the collection")
    if limit:
        tracks = tracks[:limit]

    slug = slugify(f"{artist}-{album}")
    dest = config.ALBUMS_DIR / slug
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Album: {artist} — {album}  ({len(tracks)} tracks, via {src})  -> {dest}")

    # Copies run concurrently: these workers sit blocked on SMB round-trips, so
    # overlapping them hides the latency. Oversubscribed relative to core count.
    from . import parallel
    workers = parallel.worker_count("io", n_items=len(tracks))

    def fetch_one(item):
        i, t = item
        srcp = resolve_path(t["path"])
        if srcp is None:                      # relative to the share as a last resort
            srcp = resolve_path(mp / t["rel_path"])
        if srcp is None:
            return None, f"  [{i:02d}] !! missing on share: {t['rel_path']}"
        target = dest / f"{i:02d}_{_sanitize_filename(t['title'])}{srcp.suffix.lower()}"
        if target.exists() and not force and target.stat().st_size == srcp.stat().st_size:
            return target, f"  [{i:02d}] cached: {target.name}"
        shutil.copy2(srcp, target)
        return target, (f"  [{i:02d}] {target.name}  "
                        f"({target.stat().st_size / 1e6:.1f} MB)")

    got = []
    for target, line in parallel.run(fetch_one, list(enumerate(tracks, 1)), workers):
        print(line)
        if target is not None:
            got.append(target)

    meta = {"artist": artist, "album": album, "slug": slug,
            "source": f"smb://{config.SMB_HOST}/{config.SMB_SHARE}",
            "query": query, "tracks": [p.name for p in got]}
    import json
    (dest / "album.json").write_text(json.dumps(meta, indent=2))
    print(f"{len(got)}/{len(tracks)} tracks copied locally.")
    return dest, meta
