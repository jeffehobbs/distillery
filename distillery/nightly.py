"""Pick an album from the collection, distil it, and post the result.

Designed to run unattended from cron. Everything that can fail soft does: a failed
post still leaves the audio and video on disk, and the album is only marked as used
once it has actually been distilled, so a crash doesn't burn it.

Album choice reads the same index `--library` uses. Candidates need at least
NIGHTLY_MIN_TRACKS tracks, must not be in an excluded genre or named in
excluded.txt, and must not have been used within NIGHTLY_COOLDOWN_DAYS.
"""
import errno
import os
import random
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from . import config


@contextmanager
def lock():
    """Refuse to start if another run is going.

    A distillation on a CPU-only box can outlast the gap between two cron firings,
    and two Demucs runs on the same 4 cores would take longer than either alone.
    The pid is written so a stale lock is obvious, and cleared if the process is gone.
    """
    path = config.DATA_DIR / "nightly.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            pid = int(path.read_text().split()[0])
        except (ValueError, IndexError, OSError):
            pid = None
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except OSError as e:
                alive = e.errno == errno.EPERM
        if alive:
            raise RuntimeError(
                f"another distillery run is active (pid {pid}, lock {path}) — "
                f"skipping this one")
        print(f"  clearing stale lock from pid {pid}")
        path.unlink(missing_ok=True)
    path.write_text(f"{os.getpid()} {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


# ---------------------------------------------------------------- state

def _state():
    config.STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(config.STATE_DB)
    con.execute("""CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        slug TEXT, artist TEXT, album TEXT,
        started_at REAL, finished_at REAL,
        duration_s REAL, bpm REAL, songs_used INTEGER, seed INTEGER,
        wav TEXT, mp4 TEXT, bluesky TEXT, mastodon TEXT, error TEXT)""")
    con.commit()
    return con


def recent_albums(days=None):
    """(artist, album) pairs used inside the cooldown window."""
    days = config.NIGHTLY_COOLDOWN_DAYS if days is None else days
    cutoff = time.time() - days * 86400
    con = _state()
    rows = con.execute(
        "SELECT artist, album FROM runs WHERE started_at >= ?", (cutoff,)).fetchall()
    con.close()
    return {(a, b) for a, b in rows}


def record_start(artist, album, slug):
    con = _state()
    cur = con.execute(
        "INSERT INTO runs (slug, artist, album, started_at) VALUES (?,?,?,?)",
        (slug, artist, album, time.time()))
    con.commit()
    run_id = cur.lastrowid
    con.close()
    return run_id


def record_finish(run_id, **fields):
    if not fields:
        return
    con = _state()
    cols = ", ".join(f"{k}=?" for k in fields)
    con.execute(f"UPDATE runs SET {cols}, finished_at=? WHERE id=?",
                (*fields.values(), time.time(), run_id))
    con.commit()
    con.close()


def history(limit=20):
    con = _state()
    con.row_factory = sqlite3.Row
    rows = con.execute("SELECT * FROM runs ORDER BY id DESC LIMIT ?",
                       (limit,)).fetchall()
    con.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------- picking

def _excluded_names():
    if not config.EXCLUDED_FILE.exists():
        return []
    return [l.strip().lower() for l in
            config.EXCLUDED_FILE.read_text().splitlines()
            if l.strip() and not l.startswith("#")]


def candidates(min_tracks=None, verbose=False):
    """Albums in the index that are eligible tonight."""
    if not config.ESSENTIA_DB.exists():
        raise RuntimeError(
            f"no collection index at {config.ESSENTIA_DB} — set DISTILLERY_INDEX_DB "
            f"to a SQLite index of your library (see the README).")
    min_tracks = min_tracks or config.NIGHTLY_MIN_TRACKS
    con = sqlite3.connect(f"file:{config.ESSENTIA_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    # genre lives on the track rows; an album counts as excluded if ANY of its tracks
    # carries an excluded genre, which is what catches a podcast feed tagged unevenly
    rows = con.execute("""
        SELECT tag_artist AS artist, tag_album AS album, COUNT(*) AS n,
               GROUP_CONCAT(DISTINCT LOWER(COALESCE(tag_genre,''))) AS genres
        FROM tracks
        WHERE status='ok' AND tag_album IS NOT NULL AND TRIM(tag_album) <> ''
          AND tag_artist IS NOT NULL AND TRIM(tag_artist) <> ''
        GROUP BY tag_artist, tag_album
        HAVING n >= ?""", (min_tracks,)).fetchall()
    con.close()

    excluded_names = _excluded_names()
    recent = recent_albums()
    out, skipped = [], {"genre": 0, "excluded": 0, "cooldown": 0}
    for r in rows:
        artist, album = r["artist"], r["album"]
        genres = (r["genres"] or "").lower()
        if any(g and g in genres for g in config.EXCLUDED_GENRES):
            skipped["genre"] += 1
            continue
        hay = f"{artist} {album}".lower()
        if any(x in hay for x in excluded_names):
            skipped["excluded"] += 1
            continue
        if (artist, album) in recent:
            skipped["cooldown"] += 1
            continue
        out.append({"artist": artist, "album": album, "n": r["n"]})
    if verbose:
        print(f"  {len(out)} eligible albums "
              f"(skipped {skipped['genre']} by genre, {skipped['excluded']} by name, "
              f"{skipped['cooldown']} in cooldown)")
    return out


def pick(seed=None, min_tracks=None, verbose=True):
    """Choose tonight's album."""
    pool = candidates(min_tracks=min_tracks, verbose=verbose)
    if not pool:
        raise RuntimeError("no eligible albums — the cooldown may have consumed the "
                           "whole collection, or the index is empty")
    rng = random.Random(seed if seed is not None else time.time())
    chosen = rng.choice(pool)
    if verbose:
        print(f"  tonight: {chosen['artist']} — {chosen['album']} "
              f"({chosen['n']} tracks)")
    return chosen
