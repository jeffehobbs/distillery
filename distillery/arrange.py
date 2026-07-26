"""The arrangement: what plays where, over 270 bars (9:00 at 120 BPM).

Shape is a long build with two breakdowns:

    intro  pulse  groove-1  build-1  bd-1  groove-2  build-2  bd-2  climax  outro
     16     16      32        32      16     40        32      12     48      26

Excitement is built with five knobs moving together, not just "more loops":

  * layer count      0 -> 4 concurrent sample loops
  * lowpass on loops ~700 Hz -> 18 kHz (the classic filter-opening build)
  * loop gain        quiet under the bed -> level with it
  * reverb           drenched and distant -> dry and present
  * bed voices       pad/wash -> +kick -> +hats -> +clap/perc -> ghost kicks

**Hard rule: two loops playing at the same time never come from the same song on
the album.** Enforced at selection time by checking the candidate's track number
against every event already scheduled that overlaps the new event's bar range,
and re-verified afterwards by `check_overlaps`.
"""
import json
import math
from dataclasses import dataclass, field, asdict

from . import config

BEATS_PER_BAR = config.BEATS_PER_BAR


@dataclass
class BarSpec:
    index: int
    section: str
    energy: float
    kick: bool = True
    hats: bool = True
    clap: bool = False
    perc: bool = False
    pad: bool = True
    sub: bool = True
    wash: bool = True


@dataclass
class LoopEvent:
    bar: int
    bars: int
    lane: int
    loop_id: str
    track_no: int
    track_name: str
    wav: str
    gain_db: float
    lpf_hz: float
    pan: float
    reverb_wet: float
    reverse: bool
    fade_beats: float
    energy: float
    texture: str
    kind: str = "drum"
    semitones: float = 0.0
    delay_beats: float = 0.0      # dub throw on the event's tail (0 = none)
    delay_repeats: int = 0

    @property
    def end_bar(self):
        return self.bar + self.bars


@dataclass
class Plan:
    total_bars: int
    bars: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    loop_events: list = field(default_factory=list)
    fx_events: list = field(default_factory=list)
    delights: list = field(default_factory=list)
    key: str = None
    key_scale: str = None
    seed: int = 0
    chords: bool = False
    form: str = "long"
    ideas: int = None
    bass: dict = None
    same_song_rule: bool = True   # False on albums too small to sustain it


# Forms. Each is a list of (name, bars, e_start, e_end, lanes, flags); the bar
# numbers are proportions — the whole thing is scaled to the target length.
#
# Short pieces need FEWER sections, not thinner ones. Scaling the 10-section form
# down to three minutes gave a 4-bar second breakdown, which isn't a breakdown, it's
# a stumble. So the section count follows the duration.
_LONG = [
    ("intro",       16, 0.02, 0.10, 0, dict(kick=False, hats=False, sub=False, clap=False, perc=False)),
    ("pulse",       16, 0.12, 0.24, 1, dict(kick=True,  hats=False, clap=False, perc=False)),
    ("groove-1",    32, 0.26, 0.42, 2, dict(clap=False, perc=True)),
    ("build-1",     32, 0.42, 0.60, 2, dict(clap=True,  perc=True)),
    ("breakdown-1", 16, 0.52, 0.14, 1, dict(kick=False, hats=False, clap=False, perc=False)),
    ("groove-2",    40, 0.46, 0.66, 3, dict(clap=True,  perc=True)),
    ("build-2",     32, 0.66, 0.86, 3, dict(clap=True,  perc=True)),
    ("breakdown-2", 12, 0.80, 0.24, 1, dict(kick=False, hats=False, clap=False, perc=False)),
    ("climax",      48, 0.86, 1.00, 4, dict(clap=True,  perc=True)),
    ("outro",       26, 0.60, 0.04, 2, dict(clap=False, perc=True)),
]

_EXTENDED = [   # one breakdown instead of two
    ("intro",       12, 0.02, 0.10, 0, dict(kick=False, hats=False, sub=False, clap=False, perc=False)),
    ("pulse",       12, 0.12, 0.26, 1, dict(kick=True,  hats=False, clap=False, perc=False)),
    ("groove-1",    24, 0.28, 0.44, 2, dict(clap=False, perc=True)),
    ("build-1",     24, 0.44, 0.62, 2, dict(clap=True,  perc=True)),
    ("breakdown-1", 12, 0.56, 0.16, 1, dict(kick=False, hats=False, clap=False, perc=False)),
    ("groove-2",    28, 0.50, 0.70, 3, dict(clap=True,  perc=True)),
    ("build-2",     24, 0.70, 0.88, 3, dict(clap=True,  perc=True)),
    ("climax",      32, 0.88, 1.00, 4, dict(clap=True,  perc=True)),
    ("outro",       12, 0.60, 0.04, 2, dict(clap=False, perc=True)),
]

_SINGLE = [     # one groove, one build, one breakdown, one climax
    ("intro",        8, 0.03, 0.12, 0, dict(kick=False, hats=False, sub=False, clap=False, perc=False)),
    ("pulse",        8, 0.14, 0.28, 1, dict(kick=True,  hats=False, clap=False, perc=False)),
    ("groove-1",    20, 0.30, 0.48, 2, dict(clap=False, perc=True)),
    ("build-1",     24, 0.48, 0.70, 3, dict(clap=True,  perc=True)),
    ("breakdown-1", 10, 0.62, 0.20, 1, dict(kick=False, hats=False, clap=False, perc=False)),
    ("climax",      34, 0.80, 1.00, 4, dict(clap=True,  perc=True)),
    ("outro",       16, 0.55, 0.04, 2, dict(clap=False, perc=True)),
]

_SKETCH = [     # no pulse, no breakdown: state it, build it, land it
    ("intro",        6, 0.05, 0.16, 0, dict(kick=False, hats=False, sub=False, clap=False, perc=False)),
    ("groove-1",    16, 0.30, 0.50, 2, dict(clap=False, perc=True)),
    ("build-1",     16, 0.52, 0.76, 3, dict(clap=True,  perc=True)),
    ("climax",      20, 0.86, 1.00, 4, dict(clap=True,  perc=True)),
    ("outro",        6, 0.50, 0.04, 1, dict(clap=False, perc=True)),
]

# name -> (sections, natural duration in minutes when the form is asked for by name)
FORMS = {
    "sketch":   (_SKETCH, 2.25),
    "single":   (_SINGLE, 4.0),
    "extended": (_EXTENDED, 6.0),
    "long":     (_LONG, 9.0),
}
# Which form suits a given duration. Chosen in seconds, not bars, because what a
# listener can follow is a function of time, not of bar count.
FORM_THRESHOLDS = ((165.0, "sketch"), (290.0, "single"), (430.0, "extended"))

SECTIONS = _LONG        # back-compat for anything still referring to the default


def budget_bars(n_ideas, exposure=None, mean_layers=None):
    """Bars needed to give `n_ideas` distinct ideas one clear statement each.

    bars = ideas x exposure / layers — with several loops sounding at once, the
    timeline presents more than one idea per bar, so the wall-clock need is divided
    by the average layer count. Clamped at both ends: too short and the form has no
    room, too long and we're back to restating everything three times, which is what
    the fixed nine minutes was doing.
    """
    exposure = config.EXPOSURE_BARS if exposure is None else float(exposure)
    layers = config.MEAN_LAYERS if mean_layers is None else float(mean_layers)
    bars = (max(1, int(n_ideas)) * exposure) / max(0.5, layers)
    seconds = min(config.AUTO_MAX_SECONDS,
                  max(config.AUTO_MIN_SECONDS, bars * config.SEC_PER_BAR))
    bars = int(round(seconds / config.SEC_PER_BAR / 4.0) * 4)   # phrase-aligned
    return max(8, bars)


def bars_for_minutes(minutes):
    """Bars for an explicit duration — nearest whole bar, NOT phrase-aligned.

    Auto lengths get rounded to 4-bar phrases, but an explicit request has to be
    exact: 9:00 at 120 BPM is 270 bars, and 270 is not a multiple of 4, so
    phrase-aligning turned `--length long` into 9:04.
    """
    return max(8, int(round(float(minutes) * 60.0 / config.SEC_PER_BAR)))


def resolve_length(pool=None, mode=None, bars=None, minutes=None, exposure=None):
    """Work out (total_bars, form, n_ideas) from whatever the caller specified.

    Precedence: explicit --bars, then --minutes, then a named form, then auto.
    """
    from . import loops as loopmod
    mode = (mode or config.LENGTH_MODE or "auto").lower()
    ideas = None
    if bars:
        total = int(bars)
    elif minutes:
        total = bars_for_minutes(minutes)
    elif mode in FORMS:
        return bars_for_minutes(FORMS[mode][1]), mode, None
    else:
        ideas = loopmod.count_distinct_ideas(pool or [])
        total = budget_bars(ideas, exposure=exposure)
    return total, choose_form(total * config.SEC_PER_BAR), ideas


def choose_form(seconds):
    """Pick the form whose section count suits this duration."""
    for limit, name in FORM_THRESHOLDS:
        if seconds < limit:
            return name
    return "long"


def form_sections(form):
    return FORMS.get(form or "long", FORMS["long"])[0]


def _section_bars(total_bars, form=None):
    """Scale a form to the target length, keeping the total exact.

    Largest-remainder allocation with a 1-bar floor: a naive round-then-fix pushes
    the whole rounding error into one section, which goes negative as soon as the
    target is shorter than the template.
    """
    sections = form_sections(form)
    tmpl = [s[1] for s in sections]
    tmpl_total = sum(tmpl)
    if total_bars == tmpl_total:
        return list(tmpl)
    raw = [t * total_bars / tmpl_total for t in tmpl]
    out = [max(1, int(math.floor(r))) for r in raw]
    remaining = total_bars - sum(out)
    order = sorted(range(len(raw)), key=lambda i: -(raw[i] - math.floor(raw[i])))
    i = 0
    while remaining > 0:                       # hand out spare bars, biggest fraction first
        out[order[i % len(order)]] += 1
        remaining -= 1
        i += 1
    while remaining < 0:                       # over-allocated: shave the longest
        j = max(range(len(out)), key=lambda k: out[k])
        if out[j] <= 1:
            break
        out[j] -= 1
        remaining += 1
    return out


def _build_bars(total_bars, form=None):
    bars, sections = [], []
    i = 0
    for (name, _b, e0, e1, lanes, flags), nb in zip(form_sections(form),
                                                    _section_bars(total_bars, form)):
        sections.append({"name": name, "start_bar": i, "bars": nb,
                         "energy_start": e0, "energy_end": e1, "lanes": lanes})
        for j in range(nb):
            frac = j / max(1, nb - 1)
            e = e0 + (e1 - e0) * frac
            f = dict(flags)
            if name == "breakdown-1" and j >= nb - 4:
                f["kick"] = True                 # kick creeps back before the drop
            if name == "pulse" and j < 4:
                f["sub"] = False
            if name == "groove-1" and "pulse" not in [x[0] for x in form_sections(form)] \
                    and j < 2:
                f["kick"] = True          # sketch form has no pulse: kick lands here
            if name == "outro" and frac > 0.55:
                f["kick"] = f["hats"] = False     # peel the drums off at the end
            if name == "outro" and frac > 0.8:
                f["sub"] = False
            bars.append(BarSpec(index=i, section=name, energy=round(e, 4), **f))
            i += 1
    return bars, sections


def _fx_events(sections, bars):
    """Risers into the big sections, impacts on their downbeats."""
    fx = []
    by_name = {s["name"]: s for s in sections}
    for name, riser_bars in (("groove-2", 4), ("build-2", 2), ("climax", 4)):
        s = by_name.get(name)
        if not s:
            continue
        start = s["start_bar"]
        if start - riser_bars >= 0:
            fx.append({"kind": "riser", "bar": start - riser_bars, "bars": riser_bars,
                       "gain": 0.7 if name == "build-2" else 0.9})
        fx.append({"kind": "impact", "bar": start, "gain": 0.9})
    for name in ("groove-1", "outro"):
        s = by_name.get(name)
        if s:
            fx.append({"kind": "impact", "bar": s["start_bar"], "gain": 0.6})
    return sorted(fx, key=lambda e: e["bar"])


# ---------------------------------------------------------------- loop selection

def _lane_active_bar(section, lane):
    """Lanes fade in across a section, so density grows inside it too."""
    if section["lanes"] <= 0:
        return None
    if lane >= section["lanes"]:
        return None
    # 0.75 compresses the fade-in so the last section reaches full density well
    # before it ends, instead of only in its final few bars.
    frac = 0.75 * lane / (section["lanes"] + 0.6)
    return section["start_bar"] + int(round(section["bars"] * frac))


def _targets(energy):
    """What kind of loop suits this energy level."""
    return {
        "density": 1.0 + 7.0 * energy,        # onsets/sec
        "centroid": 900.0 + 5200.0 * energy,  # Hz
        "bars": 1.0 if energy < 0.5 else 2.0,
    }


def _score_loop(loop, tgt, usage, last_track, rng, mean_usage=0.0):
    d = abs(loop["onset_density"] - tgt["density"]) / 8.0
    c = abs(math.log((loop["centroid_hz"] or 200.0) + 1e-6) - math.log(tgt["centroid"]))
    b = abs(loop["out_bars"] - tgt["bars"]) * 0.25
    s = 1.6 * loop["loopability"] - 1.0 * d - 0.6 * c - b
    # Spread the album: score against how used this song is RELATIVE TO THE MEAN, so
    # under-used songs get a bonus and over-used ones a penalty. Dividing by the
    # track count (the first attempt) made this vanish on long albums — a 22-track
    # album got a 0.016 nudge and four songs never played at all.
    s -= 0.22 * (usage.get(loop["track_no"], 0) - mean_usage)
    if loop["track_no"] == last_track:
        s -= 0.8                                                     # don't repeat a song back-to-back
    return s + float(rng.normal(0, 0.12))


TEXTURE_LANE = 90          # its own lane: one texture at a time, never two


def transpose_to(loop_key, piece_key):
    """Smallest semitone move from a loop's key to the piece's key, in [-6, +6]."""
    from .analyze import NOTE_SEMITONE
    if not loop_key or not piece_key:
        return None
    a = NOTE_SEMITONE.get(loop_key)
    b = NOTE_SEMITONE.get(piece_key)
    if a is None or b is None:
        return None
    d = (b - a) % 12
    return float(d - 12 if d > 6 else d)


def texture_is_safe(loop, piece_key, piece_scale):
    """Can this texture loop be made consonant with the piece? -> (ok, semitones).

    Guarantee 1: transpose the loop's root onto the piece's root, so it agrees with
    the drone by construction. Reject shifts beyond TEXTURE_MAX_SHIFT — past that the
    pitch shifter is doing more damage than the loop is worth.

    Guarantee 2: if the loop's key is clearly stated, its MODE must match the piece's
    too. Rooting a major loop on a minor piece still puts a major third against a
    minor one. A loop with a weak key reading has no functional third to clash, so it
    passes as pure texture.
    """
    st = transpose_to(loop.get("loop_key"), piece_key)
    if st is None:
        return False, 0.0
    if abs(st) > config.TEXTURE_MAX_SHIFT:
        return False, st
    strength = loop.get("loop_key_strength") or 0.0
    if strength >= config.TEXTURE_MODE_STRENGTH:
        lscale = (loop.get("loop_scale") or "").lower()
        pscale = (piece_scale or "minor").lower()
        if lscale and lscale[:3] != pscale[:3]:
            return False, st
    return True, st


def _schedule_texture(plan, pool, events, rng, piece_key, piece_scale,
                      enforce_rule=True):
    """Lay in the texture layer: sparse, long, one at a time.

    Texture events join the same overlap bookkeeping as drum loops, so the
    no-two-loops-from-the-same-song rule covers them too.
    """
    import numpy as np
    if not pool or not config.TEXTURE_ENABLED:
        return []
    safe = []
    for l in pool:
        ok, st = texture_is_safe(l, piece_key, piece_scale)
        if ok:
            safe.append((l, st))
    if not safe:
        return []

    out, last_track = [], None
    for section in plan.sections:
        if section["name"] == "intro":          # let the drone open alone
            continue
        bar, end = section["start_bar"], section["start_bar"] + section["bars"]
        while bar < end:
            e = plan.bars[min(bar, len(plan.bars) - 1)].energy
            if rng.random() < 0.45:             # deliberately sparse
                bar += 4
                continue
            want = min(int(rng.choice([4, 8, 8, 16])), end - bar)
            if want < 2:
                break
            if enforce_rule:
                blocked = _overlapping_tracks(events + out, bar, bar + want,
                                              TEXTURE_LANE)
                blocked_ids = set()
            else:
                blocked = set()
                blocked_ids = _overlapping_loop_ids(events + out, bar, bar + want,
                                                    TEXTURE_LANE)
            cands = [(l, st) for l, st in safe
                     if l["track_no"] not in blocked and l["id"] not in blocked_ids
                     and l["track_no"] != last_track]
            if not cands:
                bar += want
                continue
            # prefer loopable, long, and least-transposed material
            cands.sort(key=lambda ls: -(1.4 * ls[0]["loopability"]
                                        - 0.10 * abs(ls[1])
                                        + 0.15 * ls[0]["out_bars"]
                                        + float(rng.normal(0, 0.12))))
            loop, st = cands[int(rng.integers(0, min(6, len(cands))))]
            lb = max(1.0, loop["out_bars"])
            span = max(lb, math.floor(want / lb) * lb)
            if bar + span > end:
                span = math.floor((end - bar) / lb) * lb
                if span < lb:
                    break
            clash = (loop["track_no"] in _overlapping_tracks(
                         events + out, bar, bar + span, TEXTURE_LANE)
                     if enforce_rule else
                     loop["id"] in _overlapping_loop_ids(
                         events + out, bar, bar + span, TEXTURE_LANE))
            if clash:
                bar += int(math.ceil(span))
                continue
            # dub throw: long, tempo-synced, ringing on for several bars
            d_beats, d_reps = 0.0, 0
            if rng.random() < config.TEXTURE_DELAY_PROB:
                d_beats = float(rng.choice(config.TEXTURE_DELAY_BEATS))
                tail_bars = int(rng.integers(config.TEXTURE_DELAY_BARS[0],
                                             config.TEXTURE_DELAY_BARS[1] + 1))
                d_reps = int(max(3, min(24, round(tail_bars * BEATS_PER_BAR
                                                  / d_beats))))
            out.append(LoopEvent(
                bar=bar, bars=span, lane=TEXTURE_LANE, loop_id=loop["id"],
                delay_beats=d_beats, delay_repeats=d_reps,
                track_no=loop["track_no"], track_name=loop["track_name"],
                wav=loop["wav"], kind="texture", semitones=round(st, 2),
                gain_db=round(-3.0 + float(rng.normal(0, 0.8)), 2),
                lpf_hz=round(min(config.TEXTURE_TOP_HZ,
                                 700.0 * ((config.TEXTURE_TOP_HZ / 700.0) ** e)), 1),
                pan=round(float(np.clip(rng.normal(0, 0.35), -0.65, 0.65)), 3),
                reverb_wet=round(0.35 + 0.35 * (1.0 - e), 3),
                reverse=bool(rng.random() < 0.25),
                fade_beats=2.0, energy=round(e, 3),
                texture=loop.get("texture", "pad")))
            last_track = loop["track_no"]
            bar += int(math.ceil(span))
    return out


def _overlapping_tracks(events, bar, end_bar, lane):
    return {e.track_no for e in events
            if e.lane != lane and e.bar < end_bar and bar < e.end_bar}


def _overlapping_loop_ids(events, bar, end_bar, lane):
    """Loops already sounding across this range, on any lane including this one.

    Used when the same-song rule is relaxed: two different loops from one song is
    fine musically, but the SAME loop against itself is just a phase-shifted copy of
    itself, which combs rather than layers.
    """
    return {e.loop_id for e in events if e.bar < end_bar and bar < e.end_bar}


def _pick_loop(loops, tgt, blocked_tracks, usage, last_track, rng, last_loop_id=None,
               blocked_ids=()):
    pool = [l for l in loops if l["track_no"] not in blocked_tracks
            and l["id"] != last_loop_id and l["id"] not in blocked_ids]
    if not pool:
        return None
    mean_usage = sum(usage.values()) / max(1, len(usage))
    scored = sorted(pool, key=lambda l: -_score_loop(l, tgt, usage, last_track, rng,
                                                     mean_usage))
    # sample from the top of the ranking so runs don't come out identical
    k = min(len(scored), 8)
    return scored[int(rng.integers(0, k))]


def build_plan(loops, key=None, key_scale=None, total_bars=None, seed=None,
               chords=None, form=None, ideas=None, texture_pool=None):
    """Compose the full arrangement from a pool of retimed loops."""
    import numpy as np
    from . import fx
    # resolved here, not as a default argument: the tempo (and so the bar count for
    # nine minutes) can be set at runtime by --bpm
    total_bars = config.TOTAL_BARS if total_bars is None else int(total_bars)
    form = form or choose_form(total_bars * config.SEC_PER_BAR)
    seed = int(seed if seed is not None else np.random.default_rng().integers(1, 10 ** 9))
    rng = np.random.default_rng(seed)

    bars, sections = _build_bars(total_bars, form)
    plan = Plan(total_bars=total_bars, bars=bars, sections=sections,
                fx_events=_fx_events(sections, bars), key=key, key_scale=key_scale,
                seed=seed, chords=config.CHORDS if chords is None else bool(chords),
                delights=fx.roll_delights(rng, sections), form=form, ideas=ideas)
    from . import techno
    plan.bass = techno.bass_design(rng, key_scale, chords=plan.chords)
    if not loops:
        return plan

    n_songs = len({l["track_no"] for l in loops})
    enforce_rule = n_songs >= config.MIN_SONGS_FOR_RULE
    plan.same_song_rule = enforce_rule
    usage = {l["track_no"]: 0 for l in loops}
    events = []
    for section in sections:
        for lane in range(config.MAX_LAYERS):
            start = _lane_active_bar(section, lane)
            if start is None:
                continue
            sec_end = section["start_bar"] + section["bars"]
            bar = start
            last_track, last_loop_id = None, None
            while bar < sec_end:
                e = bars[min(bar, len(bars) - 1)].energy
                # rests: common when sparse, rare at full tilt
                if rng.random() < (0.32 - 0.26 * e) and bar > start:
                    bar += int(rng.integers(1, 3)) * 2
                    last_track = None
                    continue
                tgt = _targets(e)
                span_choices = [2, 4, 4, 8] if e > 0.45 else [4, 8, 8, 16]
                want = min(int(rng.choice(span_choices)), sec_end - bar)
                if want < 1:
                    break

                # Pick a loop, then quantize the span to a whole number of its
                # repeats -- which can make the event LONGER than the range we
                # checked for same-song clashes. So re-check against the final
                # range and re-pick if the growth introduced a clash; otherwise the
                # no-same-song rule could be violated by the tail of a grown event.
                loop, span = None, 0.0
                check_span = want
                for _attempt in range(4):
                    if enforce_rule:
                        blocked = _overlapping_tracks(events, bar, bar + check_span, lane)
                        blocked_ids = ()
                    else:
                        blocked = set()
                        blocked_ids = _overlapping_loop_ids(events, bar,
                                                            bar + check_span, lane)
                    cand = _pick_loop(loops, tgt, blocked, usage, last_track, rng,
                                      last_loop_id, blocked_ids=blocked_ids)
                    if cand is None:                  # every other song is sounding
                        break
                    lb = max(0.25, cand["out_bars"])
                    span = max(lb, math.floor(want / lb) * lb)
                    if bar + span > sec_end:
                        span = math.floor((sec_end - bar) / lb) * lb
                    if span < lb:
                        cand = None
                        break
                    if enforce_rule:
                        clash = _overlapping_tracks(events, bar, bar + span, lane)
                        ok_now = cand["track_no"] not in clash
                    else:
                        ok_now = cand["id"] not in _overlapping_loop_ids(
                            events, bar, bar + span, lane)
                    if ok_now:
                        loop = cand
                        break
                    check_span = max(check_span, span)   # widen and try again
                if loop is None:
                    bar += max(1, int(math.ceil(want)))
                    continue
                deep = lane * 2.2
                ev = LoopEvent(
                    bar=bar, bars=span, lane=lane, loop_id=loop["id"],
                    track_no=loop["track_no"], track_name=loop["track_name"],
                    wav=loop["wav"],
                    # energy is NOT baked in here — the renderer applies one smooth
                    # macro envelope from the same curve, so events can't stack up
                    # into a double-counted, lumpy build.
                    gain_db=round(-4.0 - deep + float(rng.normal(0, 0.8)), 2),
                    lpf_hz=round(config.LOOP_LPF_MIN_HZ *
                                 ((config.LOOP_LPF_MAX_HZ / config.LOOP_LPF_MIN_HZ)
                                  ** min(1.0, e * 1.05)), 1),
                    pan=round(float(np.clip((0.0 if lane == 0 else
                                             (-1) ** lane * (0.18 + 0.12 * lane))
                                            + rng.normal(0, 0.05), -0.6, 0.6)), 3),
                    # was 0.06 + 0.44*(1-e), i.e. 37% wet at mid energy — smeared
                    # through the whole first half of every piece
                    reverb_wet=round(0.04 + 0.26 * (1.0 - e) ** 1.5, 3),
                    reverse=bool(e < 0.3 and rng.random() < 0.22),
                    # 1-2 beats was a half to a full SECOND of fade on every
                    # event — each loop was ramped in rather than struck
                    fade_beats=0.25 if e > 0.4 else 0.75,
                    energy=round(e, 3), texture=loop["texture"])
                events.append(ev)
                usage[loop["track_no"]] += 1
                last_track, last_loop_id = loop["track_no"], loop["id"]
                bar += int(math.ceil(span))
    events += _schedule_texture(plan, texture_pool or [], events, rng,
                                key, key_scale, enforce_rule=enforce_rule)
    plan.loop_events = sorted(events, key=lambda x: (x.bar, x.lane))
    return plan


# ---------------------------------------------------------------- checks / report

def check_self_overlaps(plan):
    """Pairs where the SAME loop overlaps itself — never allowed, rule or no rule."""
    bad = []
    evs = plan.loop_events
    for i, a in enumerate(evs):
        for b in evs[i + 1:]:
            if b.bar >= a.end_bar:
                continue
            if a.loop_id == b.loop_id and a.bar < b.end_bar and b.bar < a.end_bar:
                bad.append((a.loop_id, a.bar, b.bar))
    return bad


def check_overlaps(plan):
    """Simultaneous loops from the same song — empty when the rule is relaxed."""
    if not getattr(plan, "same_song_rule", True):
        return []
    bad = []
    evs = plan.loop_events
    for i, a in enumerate(evs):
        for b in evs[i + 1:]:
            if b.bar >= a.end_bar:
                continue
            if a.bar < b.end_bar and b.bar < a.end_bar and a.track_no == b.track_no:
                bad.append((a.loop_id, b.loop_id, a.bar, b.bar))
    return bad


def concurrency_profile(plan):
    """Loops sounding per bar — the numeric version of the build."""
    prof = [0] * plan.total_bars
    for e in plan.loop_events:
        for b in range(e.bar, min(plan.total_bars, int(math.ceil(e.end_bar)))):
            prof[b] += 1
    return prof


def report(plan, meta=None):
    lines = []
    a = meta.get("artist") if meta else None
    al = meta.get("album") if meta else None
    if a:
        lines.append(f"distillery — {a} — {al}")
    mins = plan.total_bars * config.SEC_PER_BAR / 60
    lines.append(f"{plan.total_bars} bars @ {config.TARGET_BPM:g} BPM = "
                 f"{int(mins)}:{round((mins % 1) * 60):02d} "
                 f"({plan.total_bars * config.SEC_PER_BAR:.1f}s)   "
                 f"form '{plan.form}' ({len(plan.sections)} sections)"
                 + (f"   {plan.ideas} distinct ideas" if plan.ideas else "") + "   "
                 f"key {plan.key} {plan.key_scale}   "
                 f"{'chords' if plan.chords else 'drone (no chords)'}   "
                 f"seed {plan.seed}")
    if plan.bass:
        b = plan.bass
        lines.append(f"bass: {b['pattern']} pattern, degrees {b['degrees']}, "
                     f"drive {b['drive']:g}, lpf {b['lpf_hz']:g} Hz"
                     + (f", glide {b['glide_ms']:g}ms" if b['glide_ms'] else "")
                     + (", octave up" if b['octave'] == 0 else ""))
    if plan.delights:
        lines.append("delights: " + ", ".join(
            f"{d['name']}@{d['bar']}" for d in plan.delights))
    prof = concurrency_profile(plan)
    lines.append("")
    lines.append(f"{'section':<13}{'bars':>10}  {'energy':<12}{'layers':<8}loops")
    for s in plan.sections:
        s0, s1 = s["start_bar"], s["start_bar"] + s["bars"]
        evs = [e for e in plan.loop_events if s0 <= e.bar < s1]
        peak = max(prof[s0:s1]) if s1 <= len(prof) else 0
        lines.append(f"{s['name']:<13}{f'{s0}-{s1 - 1}':>10}  "
                     f"{s['energy_start']:.2f}->{s['energy_end']:.2f}  "
                     f"{f'{peak} max':<8}{len(evs)}")
    lines.append("")
    tracks = {}
    for e in plan.loop_events:
        tracks.setdefault(e.track_no, []).append(e)
    lines.append("songs used:")
    for tn in sorted(tracks):
        evs = tracks[tn]
        bars = sum(e.bars for e in evs)
        lines.append(f"  {tn:02d} {evs[0].track_name[:44]:<44} "
                     f"{len(evs):>3} events  {bars:>5.1f} bars")
    lines.append("")
    lines.append("same-song rule: " + ("enforced" if plan.same_song_rule else
                 f"RELAXED (album has under {config.MIN_SONGS_FOR_RULE} songs; "
                 "a loop still never overlaps itself)"))
    lines.append(f"loop events: {len(plan.loop_events)}   "
                 f"max concurrent: {max(prof) if prof else 0}   "
                 f"same-song overlaps: {len(check_overlaps(plan))}")
    lines.append("")
    lines.append("timeline (bar: layers)")
    row = ""
    for b in range(0, plan.total_bars, 2):
        row += str(min(9, prof[b]))
        if len(row) >= 60:
            lines.append(f"  {row}")
            row = ""
    if row:
        lines.append(f"  {row}")
    return "\n".join(lines)


def plan_to_json(plan):
    d = {"total_bars": plan.total_bars, "seed": plan.seed, "key": plan.key,
         "key_scale": plan.key_scale, "bpm": config.TARGET_BPM,
         "chords": plan.chords, "delights": plan.delights,
         "same_song_rule": plan.same_song_rule,
         "form": plan.form, "ideas": plan.ideas, "bass": plan.bass,
         "sections": plan.sections, "fx_events": plan.fx_events,
         "bars": [asdict(b) for b in plan.bars],
         "loop_events": [asdict(e) for e in plan.loop_events]}
    return json.dumps(d, indent=2)
