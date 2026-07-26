"""Paths and global constants for distillery."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path=None):
    """Fold ROOT/.env into the environment, without overriding what's already set.

    Done here rather than only in run.sh so that configuration holds however the
    package is entered — a cron line, a direct `python -m distillery`, or an import
    from another script all see the same settings.
    """
    path = Path(path or ROOT / ".env")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()

DATA_DIR = Path(os.environ.get("DISTILLERY_DATA", ROOT / "data"))

ALBUMS_DIR = DATA_DIR / "albums"      # downloaded source tracks, per album slug
STEMS_DIR = DATA_DIR / "stems"        # demucs output, per album slug / track
LOOPS_DIR = DATA_DIR / "loops"        # rendered drum loops (source tempo)
RETIMED_DIR = DATA_DIR / "retimed"    # loops resampled to TARGET_BPM
OUT_DIR = DATA_DIR / "output"         # finished remixes + reports
TMP_DIR = DATA_DIR / "tmp"

SR = 44100
CHANNELS = 2

# The remix itself.
TARGET_SECONDS = 9 * 60          # 540 s
BEATS_PER_BAR = 4

# Tempo is runtime-settable (--bpm / set_tempo), so nothing may snapshot these into
# module constants at import — read config.SEC_PER_BAR etc. at the point of use.
TARGET_BPM = float(os.environ.get("DISTILLERY_BPM") or 120.0)
SEC_PER_BEAT = 60.0 / TARGET_BPM
SEC_PER_BAR = SEC_PER_BEAT * BEATS_PER_BAR
TOTAL_BARS = int(round(TARGET_SECONDS / SEC_PER_BAR))

# Only tempos where TARGET_SECONDS is a whole number of bars land on exactly
# 540.000 s (4/4 -> any multiple of 4). Anything else is up to a bar short.
EXACT_BPMS = tuple(range(104, 145, 4))

# --- length. "auto" sizes the piece from the material rather than the clock:
#     bars = distinct_ideas x EXPOSURE_BARS / MEAN_LAYERS
# EXPOSURE_BARS is the real aesthetic dial — how long one idea gets to breathe
# before something else arrives. MEAN_LAYERS is measured, not assumed: finished
# pieces average about 1.8 loops sounding at once.
LENGTH_MODE = os.environ.get("DISTILLERY_LENGTH", "auto")
# Bars of loop-time each distinct idea gets — about two 8-bar statements. Measured
# pieces gave ~8 bars per unique LOOP, but a distinct idea (a cluster) covers
# several loops, so budgeting 8 here made every album collapse onto the 2:30 floor
# and the length stopped responding to the material at all.
EXPOSURE_BARS = float(os.environ.get("DISTILLERY_EXPOSURE", 16.0))
MEAN_LAYERS = 1.8
AUTO_MIN_SECONDS = 150.0     # 2:30 — below this the form can't breathe
AUTO_MAX_SECONDS = 400.0     # 6:40 — past this we're back to under-sampling


def set_tempo(bpm):
    """Set the target tempo and re-derive everything that depends on it."""
    global TARGET_BPM, SEC_PER_BEAT, SEC_PER_BAR, TOTAL_BARS
    TARGET_BPM = float(bpm)
    SEC_PER_BEAT = 60.0 / TARGET_BPM
    SEC_PER_BAR = SEC_PER_BEAT * BEATS_PER_BAR
    TOTAL_BARS = int(round(TARGET_SECONDS / SEC_PER_BAR))
    return TARGET_BPM

# Loop extraction: small loops, lots of them.
LOOP_BAR_SIZES = (0.5, 1.0, 2.0)   # half-bar (2 beats), bar, two bars
LOOPS_PER_SIZE = 6                 # top-N per track per size  -> up to 18/track
XFADE_MS = 25.0                    # user-verified seamless wrap (see memory)

# Mix. DRUMS COME FIRST: the borrowed drum loops are the point of the whole thing,
# so they lead the mix. They were measured at -19.7 dB under the master and never
# led a single section — the synth bed led everywhere, and in the outro even the
# texture layer beat them.
# Calibrated across the well-recorded albums (On The Corner, Bumps, Ron Miles) —
# deliberately NOT on the Lee Perry, whose Black Ark mixes bury the drums under
# effects and would drag the default off. At +9 the loops sit level with the synth
# bed and 8-10 dB ahead of the texture layer.
LOOP_LEVEL_DB = float(os.environ.get("DISTILLERY_LOOPS_DB", 9.0))
LOOP_TARGET_DBFS = -18.0   # each loop normalized here before arrangement gain
# The loop lowpass floor. 700 Hz meant the borrowed drums only got bright at the very
# top of the energy curve — at mid energy they sat at ~3.5 kHz.
LOOP_LPF_MIN_HZ = 1500.0
LOOP_LPF_MAX_HZ = 19000.0
LOOP_TRANSIENT = 0.6       # attack enhancement on the loop bus
LOOP_DUCK_RELEASE_MS = 90.0   # was 170: slow recovery reads as pumping, not groove
MASTER_PEAK_DBFS = -1.0
MAX_LAYERS = 4
# The no-two-loops-from-one-song rule caps concurrency at the number of songs, so a
# 2-track album could never stack more than 2 layers and the 0->4 build was
# impossible. Below this many songs the rule is relaxed (user directive 2026-07-25);
# a loop may still never overlap ITSELF, which would only comb-filter.
MIN_SONGS_FOR_RULE = 4

# --- bed voice levels. Measured: at the old balance the bed was 98.4% sub-200 Hz
# with a 99 Hz centroid — its own kick and sub swamped its hats, clap and perc, and
# since the bed leads the mix the whole master inherited that (73% sub-200 Hz, where
# techno sits nearer 35-50%). The sub was doing the kick's job an octave up; the
# percussion that carries brightness and snap was 10-12 dB under it.
BED_LEVELS = {"kick": -4.0, "sub": -13.0, "hats": -8.0, "clap": -6.5,
              "perc": -10.0, "fx": -12.0}
# The three low voices used to crowd one octave: kick fundamental ~42 Hz, sub
# 65-123 Hz, drone 82-155 Hz. The drone is now high-passed above the sub's
# fundamental so it contributes low-mid warmth instead of competing down there, and
# the sub is ducked against the kick so the two never sound in the same instant.
DRONE_HPF_HZ = 150.0
SUB_DUCK_DB = 4.5
KICK_HPF_HZ = 30.0        # below this is headroom-eating rumble, not audible weight
KICK_AMP_TAU = 0.12       # was 0.18 — lengthened once for headroom, at the cost of snap
KICK_CLICK = 0.45         # was 0.32 — the click IS the attack

# --- master tilt. Cheapest global correction for the low-heavy balance.
MASTER_TILT_HIGH_HZ = 4000.0
MASTER_TILT_HIGH_DB = 3.0
MASTER_TILT_LOW_HZ = 150.0
MASTER_TILT_LOW_DB = -2.0
# How much transient peak the master limiter shaves, relative to the mix's own peak.
# A snappier kick has a higher crest, so the mix got quieter; swept 3.5/5/6.5 dB and
# 5.0 bought 1.2 dB of level for 0.55 dB of build arc; after ducking the sub and
# high-passing the drone the mix lost sustained low energy (which is what carries RMS),
# so 6.5 recovers another 1.4 dB. Raising the kick instead was measured and rejected:
# it cost 5 points of low-share and 90 Hz of centroid for 0.4 dB.
MASTER_SHAVE_DB = 6.5

# Demucs
DEMUCS_MODEL = "htdemucs"
# Stems written to disk. Demucs computes all four regardless, so keeping
# "other" (horns/guitar/keys — everything that is not drums/bass/vocals) is
# free compute; it feeds the texture layer.
WANT_STEMS = ("drums", "other")

# --- video + posting (see video.py, poster.py). Both optional.
VIDEO_DIR = DATA_DIR / "videos"
VIDEO_FONT = Path(os.environ.get("DISTILLERY_VIDEO_FONT", ROOT / "data" / "kabel.ttf"))
VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_FPS = 1280, 720, 25
VIDEO_CRF = 23
# Target file size. The visuals are noisy (a moving waveform plus a glow), so
# quality-based encoding produced ~3.6 Mbps — 68 MB for 2:31, and a 6:40 piece would
# have been ~135 MB, past what Mastodon accepts. Encoding to a size budget instead
# means the file fits whatever the length; masto.py still asks the instance for its
# real cap before uploading.
# 90 MB fits mastodon.social's advertised 104 MB cap with margin and is far inside
# Bluesky's, while giving a six-minute piece a usable bitrate. masto.py still asks the
# instance for its real limit before uploading, so a stricter server is respected.
VIDEO_MAX_MB = float(os.environ.get("DISTILLERY_VIDEO_MAX_MB", 90.0))
POST_ENABLED = os.environ.get("DISTILLERY_POST", "1") not in ("0", "no", "false")
# Bluesky rejects videos of 3 minutes or longer, and distillery's auto length runs to
# 6:40 — so posting there is conditional on duration, with a margin under the limit.
BLUESKY_MAX_VIDEO_S = float(os.environ.get("DISTILLERY_BSKY_MAX_S", 175.0))

# --- nightly picker
STATE_DB = DATA_DIR / "state.db"
NIGHTLY_MIN_TRACKS = 3          # fewer than this makes a thin, repetitive pool
NIGHTLY_COOLDOWN_DAYS = 90      # don't revisit an album for this long
EXCLUDED_GENRES = tuple(g.strip().lower() for g in os.environ.get(
    "DISTILLERY_EXCLUDED_GENRES",
    "podcast,books & spoken,comedy,spoken word,audiobook").split(",") if g.strip())
EXCLUDED_FILE = ROOT / "excluded.txt"   # one artist/album substring per line

USER_AGENT = "distillery/0.1 ( +https://github.com/jeffehobbs/distillery )"

# Harmony. Chords are OFF: the bed holds a single tonal drone (root + octave) on
# the album's key instead of running a four-chord progression. Set
# DISTILLERY_CHORDS=1 (or --chords) to bring the progression back.
CHORDS = os.environ.get("DISTILLERY_CHORDS", "") not in ("", "0", "no", "false")
DRONE_INTERVALS = (0,)         # ONE note. No octave above it: that upper octave
                               # landed around 260-520 Hz and read as a sustained
                               # lead line sitting in the middle of the mix.
# The note is placed in ONE fixed low octave starting here (E2, 82 Hz), so the
# drone lands at 82-155 Hz whatever the album's key. A plain octave offset instead
# leaves the register key-dependent — keys A and B came out at 220-247 Hz, right
# back in the middle of the mix where the complaint started.
DRONE_LOW_MIDI = 40
DRONE_SEGMENT_BARS = 8         # long sustains; the chord path uses 4
# The drone is atmosphere, not a part — it belongs well under the drums and loops.
# Override per run with --drone-db (or DISTILLERY_DRONE_DB).
DRONE_LEVEL_DB = float(os.environ.get("DISTILLERY_DRONE_DB", -22.0))
DRONE_TOP_HZ = 600.0           # dark: harmonics above this are gone, so the drone
                               # can't sing over the drums

# --- texture layer, cut from the demucs "other" stem (horns/guitar/keys).
# Harmonic safety rests on three guarantees, not on luck:
#   1. every texture loop is PITCH-SHIFTED so its root matches the piece's root, so
#      it is consonant with the drone by construction;
#   2. a loop whose key is clearly stated (strength >= TEXTURE_MODE_STRENGTH) must
#      also match the piece's mode, or its third fights the piece's;
#   3. only ONE texture loop sounds at a time, so no texture can clash with another.
TEXTURE_ENABLED = os.environ.get("DISTILLERY_TEXTURE", "1") not in ("0", "no", "false")
TEXTURE_BAR_SIZES = (2.0, 4.0)      # longer than drum loops: texture needs room
TEXTURE_PER_SIZE = 3
# Measured, not guessed: at -9 dB the texture bus came out +1.9 dB LOUDER than the
# whole mix (+4.9 in the outro), with the borrowed drum loops sitting 15 dB below it.
# A background layer belongs ~12-15 dB UNDER the mix.
TEXTURE_LEVEL_DB = float(os.environ.get("DISTILLERY_TEXTURE_DB", -22.0))
# Max semitones a texture loop may be transposed. Must be < 6: the shortest move
# round the circle is always <= 6, so a cap of 6 rejects nothing at all. 4 keeps the
# pitch shifter in territory where it sounds like the instrument, and still leaves
# 3/4 of all keys usable.
TEXTURE_MAX_SHIFT = 4.0
TEXTURE_MODE_STRENGTH = 0.5         # above this, the key is definite enough to clash
# Texture measured 75% low-mid with 0.0% above 3 kHz — pure mud contribution. Raising
# its ceiling and adding a little shelf gives it sheen; at -22 dB, highs read as air
# rather than presence, so this does not undo the level work.
TEXTURE_TOP_HZ = 5000.0
TEXTURE_AIR_DB = 2.5
TEXTURE_AIR_HZ = 4000.0
# Dub-style trailing delay on some texture events: the event's last beat is thrown
# into a long, progressively-darkening, ping-ponged echo that rings on past the end
# of the event. Tails DO spill over the following texture, which is the point — and
# it stays consonant because every texture is transposed onto the piece's root.
TEXTURE_DELAY_BEATS = (1.0, 1.5, 2.0, 3.0)
TEXTURE_DELAY_BARS = (4, 8)         # how far the tail rings on, in bars
# A throw has to be pushed UP to read at all. mashup-app learned this the hard way:
# its first delay implementation was inaudible because the grabbed beat sat ~20 dB
# under the mix it was ringing into. Boost the throw and decay it gently.
# The throw no longer needs its own boost: it was added when the texture bus was far
# too hot, and it made the ring-outs the most prominent thing in the piece.
TEXTURE_DELAY_GAIN_DB = 0.0
TEXTURE_DELAY_DECAY_DB = -3.0
TEXTURE_DELAY_PROB = 0.5

# --- bass. It used to be ONE note (the root) in one of two hardcoded rhythms with
# fixed timbre — identical on every album and every seed, which is what made runs
# sound like each other. Now a bassline is designed per run from the seed: a rhythm,
# a set of safe scale degrees, and timbre.
#
# Degrees are semitones above the bass root and NEVER include a third (3 or 4), since
# the third is exactly what would impose major/minor on a piece that deliberately has
# no chords. Root, fourth, fifth, sixth, minor seventh and octave are all consonant
# over a root drone.
BASS_DEGREES = {
    "minor": ((0, 7, 12), (0, 10, 12), (0, 5, 12), (0, 7, 10), (0, 12, 7)),
    "major": ((0, 7, 12), (0, 5, 12), (0, 7, 9), (0, 12, 7), (0, 9, 12)),
}
# (beat, length_in_beats, degree_index) — degree_index picks from the run's degrees
BASS_PATTERNS = {
    "long":        ((0.0, 3.80, 0),),
    "half":        ((0.0, 1.60, 0), (2.0, 1.60, 0)),
    "root-pulse":  ((0.0, 0.90, 0), (1.0, 0.90, 0), (2.0, 0.90, 0), (3.0, 0.90, 0)),
    "offbeat":     ((0.5, 0.45, 0), (1.5, 0.45, 0), (2.5, 0.45, 0), (3.5, 0.45, 1)),
    "walk":        ((0.0, 1.80, 0), (2.0, 1.80, 1)),
    "octave-push": ((0.0, 0.90, 0), (1.5, 0.45, 2), (2.5, 0.90, 0), (3.5, 0.45, 1)),
    "three-two":   ((0.0, 0.70, 0), (0.75, 0.70, 0), (1.5, 0.70, 0), (2.5, 1.40, 1)),
    "dub-drop":    ((0.0, 2.40, 0), (3.5, 0.45, 1)),
    "sixteenth":   ((0.0, 0.22, 0), (0.25, 0.22, 0), (0.5, 0.22, 0), (0.75, 0.22, 0),
                    (2.0, 0.90, 1)),
}

# Effects (pedalboard). Number of per-run "delights" rolled by the arranger.
FX_ENABLED = os.environ.get("DISTILLERY_FX", "1") not in ("0", "no", "false")
DELIGHTS_PER_RUN = (2, 3)

# Local music library over SMB. Credentials come from secrets.txt (mode 600) or
# the environment — never hardcoded in the source.
SECRETS_FILE = ROOT / "secrets.txt"
# Point these at your own NAS via secrets.txt or the environment; see
# secrets.txt.example. Nothing here is specific to one machine.
SMB_HOST = os.environ.get("SMB_HOST", "nas.local")
SMB_SHARE = os.environ.get("SMB_SHARE", "music")
# Mountpoints checked in order; the first that exists and is readable wins.
MOUNT_CANDIDATES = (f"/Volumes/{SMB_SHARE}",
                    str(Path.home() / f".distillery/mnt/{SMB_SHARE}"))
FALLBACK_MOUNTPOINT = Path.home() / f".distillery/mnt/{SMB_SHARE}"
# OPTIONAL: a pre-built SQLite index of your collection (the schema used by
# essentia-explorer: a `tracks` table with path/rel_path/tag_artist/tag_album/
# tag_title/tag_tracknumber/status). If present, "Artist - Album" resolves to exact
# file paths instantly instead of walking the whole share. Without it, distillery
# just walks <Artist>/<Album>/ directories, which works fine but is slower.
ESSENTIA_DB = Path(os.environ.get(
    "DISTILLERY_INDEX_DB",
    Path.home() / "Scripts/essentia-explorer/data/essentia.db"))


# Extra places to look for credentials, so an existing setup can be reused rather
# than duplicated. Checked after secrets.txt, never overriding it.
EXTRA_SECRETS_FILES = tuple(
    Path(p) for p in os.environ.get("DISTILLERY_EXTRA_SECRETS", "").split(":") if p)
# A two-line username=/password= file, the shape mount.cifs and smbclient use.
SMB_CREDENTIALS_FILE = Path(os.environ.get(
    "DISTILLERY_SMB_CREDENTIALS", Path.home() / ".smb_credentials"))


def _read_kv(path):
    out = {}
    try:
        text = Path(path).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return out
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def secrets():
    """All KEY=VALUE credentials, from secrets.txt then any extra files then env.

    Holds SMB credentials for --library and the Bluesky/Mastodon tokens for
    posting. Environment variables win, so a cron entry can override a file.
    """
    out = {}
    for extra in EXTRA_SECRETS_FILES:
        out.update(_read_kv(extra))
    out.update(_read_kv(SECRETS_FILE))
    # a username=/password= credentials file maps onto SMB_USER/SMB_PASSWORD
    if SMB_CREDENTIALS_FILE.exists():
        kv = _read_kv(SMB_CREDENTIALS_FILE)
        out.setdefault("SMB_USER", kv.get("username", ""))
        out.setdefault("SMB_PASSWORD", kv.get("password", ""))
        out = {k: v for k, v in out.items() if v}
    for k in ("SMB_HOST", "SMB_SHARE", "SMB_USER", "SMB_PASSWORD",
              "BLUESKY_HANDLE", "BLUESKY_PASSWORD",
              "MASTODON_INSTANCE_URL", "MASTODON_ACCESS_TOKEN"):
        if os.environ.get(k):
            out[k] = os.environ[k]
    return out


def fingerprint(path):
    """Cheap identity for a source file: size + mtime.

    Every cache stage keys on this, not just the filename. Re-fetching the same
    album from a different source (bad YouTube rip -> the good local mp3) produces
    identical filenames, and stale analysis/stems/loops would otherwise be reused
    silently against completely different audio.
    """
    try:
        st = os.stat(path)
    except OSError:
        return None
    return f"{st.st_size}:{int(st.st_mtime)}"


def ensure_dirs():
    for d in (ALBUMS_DIR, STEMS_DIR, LOOPS_DIR, RETIMED_DIR, OUT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
