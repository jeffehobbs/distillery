"""Render an MTV-style waveform video from a finished distillation.

A bright cyan waveform filling the whole frame, with a blurred neon copy behind it,
over a colour wash derived from the audio's own frequency content. The album's
details sit bottom-left. Pure ffmpeg — no extra Python dependency.

The CQT wash is computed at half resolution and upscaled, because the heavy blur
hides the difference and the full-resolution version is several times slower. That
matters on a small always-on box rendering a five-minute piece nightly.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from . import config

WAVE_COLOR = "0x8ef0ff"     # bright cyan trace
BG_COLOR = "0x0a0a14"       # near-black indigo
TITLE_SIZE = 56
META_SIZE = 32
MARGIN_X = 56
MARGIN_BOTTOM = 52
LINE_SPACING = 8
GROUP_GAP = 16
META_COLOR = "0xffe64d"     # hot yellow
TITLE_COLOR = "white"


FALLBACK_FONTS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                  "/System/Library/Fonts/Supplemental/Futura.ttc",
                  "/System/Library/Fonts/Helvetica.ttc")


def _covers(font, text):
    """Does `font` have a glyph for every character in `text`?

    A display font often carries only basic Latin. ffmpeg draws anything missing as
    an empty box rather than complaining, so an accented artist name silently comes
    out mangled — worth checking rather than discovering it in a published video.
    """
    try:
        from fontTools.ttLib import TTFont
    except ImportError:
        return True                        # can't check; assume the operator knows
    wanted = {ord(c) for c in text if c not in "\n\r\t "}
    try:
        f = TTFont(str(font), fontNumber=0, lazy=True)
        have = set()
        for table in f["cmap"].tables:
            have.update(table.cmap.keys())
        f.close()
    except Exception:                       # noqa: BLE001 - unreadable font
        return True
    return not (wanted - have)


def font_path(text=""):
    """The caption font: the configured one if it can render `text`, else a fallback."""
    candidates = []
    p = Path(config.VIDEO_FONT)
    if p.exists():
        candidates.append(p)
    candidates += [Path(c) for c in FALLBACK_FONTS if Path(c).exists()]
    if not candidates:
        return None
    for cand in candidates:
        if not text or _covers(cand, text):
            if cand != candidates[0]:
                print(f"  note: {candidates[0].name} lacks glyphs for this caption — "
                      f"using {cand.name}")
            return cand
    return candidates[0]


def has_filter(name):
    """Is this ffmpeg build carrying `name`? Cached per process."""
    if name not in _FILTERS:
        try:
            out = subprocess.run(["ffmpeg", "-hide_banner", "-filters"],
                                 capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            out = ""
        _FILTERS[name] = f" {name} " in out
    return _FILTERS[name]


_FILTERS = {}


def _textfile(tmp_dir, name, lines):
    p = Path(tmp_dir) / name
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _drawtext(font, textfile, size, color, x, y):
    parts = [f"textfile={textfile}", f"fontcolor={color}", f"fontsize={size}",
             f"x={x}", f"y={y}", f"line_spacing={LINE_SPACING}",
             "shadowcolor=0x000000C0", "shadowx=3", "shadowy=3"]
    if font:
        parts.insert(0, f"fontfile={font}")
    return "drawtext=" + ":".join(parts)


def build(wav_path, title_lines, meta_lines, out_path, crf=None,
          target_mb=None):
    """Render `wav_path` to an mp4 at `out_path`. Returns the Path."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not on PATH — needed to render video")
    wav_path, out_path = Path(wav_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    W, H, FPS = config.VIDEO_WIDTH, config.VIDEO_HEIGHT, config.VIDEO_FPS
    # Encode to a size budget rather than a quality target, so a six-minute piece
    # lands under the platform caps just as a two-minute one does.
    audio_kbps = 160
    dur = duration_s(wav_path)
    target_mb = config.VIDEO_MAX_MB if target_mb is None else target_mb
    if target_mb and dur > 1:
        total_kbps = target_mb * 8192.0 / dur
        v_kbps = int(max(350, total_kbps - audio_kbps - 32))
        rate_args = ["-b:v", f"{v_kbps}k", "-maxrate", f"{int(v_kbps * 1.3)}k",
                     "-bufsize", f"{int(v_kbps * 2)}k"]
    else:
        rate_args = ["-crf", str(crf or config.VIDEO_CRF)]
    title_lines = [s for s in title_lines if s]
    meta_lines = [s for s in meta_lines if s]
    # Chosen per block, not for the whole caption: an accented artist name should not
    # cost the display font on the metadata line, which is plain ASCII.
    title_font = font_path("".join(title_lines))
    meta_font = font_path("".join(meta_lines))

    # stack both groups up from the bottom-left corner
    title_lh, meta_lh = TITLE_SIZE + LINE_SPACING, META_SIZE + LINE_SPACING
    meta_y = H - MARGIN_BOTTOM - len(meta_lines) * meta_lh
    title_y = meta_y - GROUP_GAP - len(title_lines) * title_lh

    with tempfile.TemporaryDirectory(dir=str(config.TMP_DIR)) as td:
        tf = _textfile(td, "title.txt", title_lines)
        mf = _textfile(td, "meta.txt", meta_lines)
        cqt = (f"[0:a]showcqt=s={W // 2}x{H // 2}:fps={FPS}:sono_h=0:axis_h=0:"
               f"bar_h={H // 2}:count=6:gamma=3:bar_g=2,scale={W}x{H},format=rgba,"
               f"gblur=sigma=18:steps=2,eq=saturation=2.4:brightness=0.03,"
               f"colorchannelmixer=aa=0.5[cqt]")
        wave = (f"[0:a]showwaves=s={W}x{H}:rate={FPS}:mode=cline:"
                f"colors={WAVE_COLOR}:draw=full,format=rgba,split=2[wsharp][wpre];"
                f"[wpre]gblur=sigma=9:steps=2,eq=saturation=1.6[wglow]")
        # drawtext needs an ffmpeg built with libfreetype. Where it's missing, render
        # the visualiser without the caption rather than failing the whole job.
        if has_filter("drawtext"):
            caption = (
                f"[b3]{_drawtext(title_font, tf, TITLE_SIZE, TITLE_COLOR, MARGIN_X, title_y)},"
                f"{_drawtext(meta_font, mf, META_SIZE, META_COLOR, MARGIN_X, meta_y)}[v]")
        else:
            print("  note: this ffmpeg has no drawtext filter — rendering without "
                  "the caption (rebuild ffmpeg with libfreetype for captions)")
            caption = "[b3]null[v]"
        graph = (
            f"color=c={BG_COLOR}:s={W}x{H}:r={FPS}[bg];{cqt};{wave};"
            f"[bg][cqt]overlay=0:0:shortest=1[b1];"
            f"[b1][wglow]overlay=0:0[b2];"
            f"[b2][wsharp]overlay=0:0[b3];" + caption
        )
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav_path),
               "-filter_complex", graph, "-map", "[v]", "-map", "0:a",
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-profile:v", "high",
               *rate_args, "-preset", "veryfast",
               "-c:a", "aac", "-b:a", f"{audio_kbps}k", "-movflags", "+faststart",
               str(out_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{proc.stderr[-1500:]}")
    return out_path


def duration_s(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=nw=1:nk=1", str(path)],
        capture_output=True, text=True)
    try:
        return float(out.stdout.strip())
    except ValueError:
        return 0.0
