# distillery

**Distills a whole album into one ambient techno track.**

Point it at an album. It analyzes every song with Essentia, isolates each song's
drum track with Demucs, cuts those drums into many small beat-aligned loops, retimes
them all to one tempo, and lays them over a **fully synthesized** ambient techno bed
— arranged as a single long build, with pedalboard effects.

Nothing in the bed is sampled: the kick, bass, hats, clap, drone and noise wash are
all generated from oscillators and filtered noise, tuned to the album's own key. The
only recorded audio in the output is the album's drums, and optionally its melodic
"other" stem used as texture.

```bash
./run.sh remix --local ~/Music/SomeAlbum --mp3
```

Output lands in `data/output/` as `distillery_<slug>_<timestamp>.wav`, alongside a
`.txt` report and a `.plan.json` describing every decision the arranger made.

Each run derives its own **tempo**, **length**, **form**, **bassline** and **effects**
from the material and the seed, so two runs of the same album are different pieces —
and a given seed reproduces its render exactly.

```
distillery — Miles Davis — On The Corner (Remaster)
160 bars @ 128 BPM = 5:00 (300.0s)   form 'extended' (9 sections)   18 distinct ideas
key G major   drone (no chords)   seed 73
bass: walk pattern, degrees [0, 7, 12], drive 1.39, lpf 294 Hz, glide 12ms
delights: phaser-drift@32, drum-crush@220, dub-echo@110
same-song rule: enforced
```

## Requirements

macOS or Linux, Python 3.12, and `ffmpeg` on PATH. Two things to know before you
start:

* **Essentia has no general arm64 macOS wheel** on PyPI and no brew formula — only
  specific dev builds ship one. `requirements.txt` pins `essentia==2.1b6.dev1389`,
  which has cp310–cp313 arm64 macOS wheels. On Intel or Linux the ordinary release
  works.
* **Demucs wants a GPU.** It uses Apple MPS automatically on Apple silicon (roughly
  5–30 s per track). On CPU it still works, several times slower. This is the one slow
  stage; everything else takes seconds.

Optional: `yt-dlp`, for the YouTube source. An SMB share and `mount_smbfs` (macOS),
for `--library`.

## Install

```bash
git clone https://github.com/jeffehobbs/distillery
cd distillery
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./run.sh selftest          # 176 checks; no album, network or models needed
```

`run.sh` looks for an interpreter in this order: `$DISTILLERY_PYTHON`, `./.venv`, then
a sibling `essentia-explorer` venv.

The first Demucs run downloads the `htdemucs` model (~80 MB) from HuggingFace and
caches it under `~/.cache/huggingface`. **No HuggingFace account or token is
required** — `adefossez/HTDemucs` is public and ungated. You will see a warning
suggesting you set `HF_TOKEN` for higher rate limits and faster downloads; it is
advisory, and the download works fine without one.

## Quickstart

```bash
# a folder of audio files you already have
./run.sh remix --local ~/Music/SomeAlbum --mp3

# an album on an SMB share (see "Where the audio comes from")
./run.sh find "bumps"
./run.sh remix --library "Artist - Album" --mp3

# other arrangements of an album already processed — a few seconds each
./run.sh rearrange <slug> --seed 99 --mp3
./run.sh list
```

Every stage caches, so the expensive work happens once per album. Re-arranging is fast
enough to audition seeds by ear.

## A note on what you feed it

This is a tool for making something new out of records **you own**. It reads local
files; the `--album` flag that searches YouTube is there for convenience and is the
worst-sounding option anyway. Whatever you make from copyrighted material is yours to
keep to yourself — don't assume a distillation is free of the original's rights just
because most of what you hear is synthesized.

## Where the audio comes from

| flag | source |
|---|---|
| `--local <dir>` | a directory of audio files |
| `--library "Artist - Album"` | an album on an SMB share |
| `--slug <slug>` | an album already fetched into `data/albums/` |
| `--url <url>` | a playlist or album URL, via `yt-dlp` |
| `--album "Artist - Album"` | found on YouTube from a MusicBrainz tracklist |

`--library` mounts the share with `mount_smbfs` if it isn't already mounted, using
credentials from `secrets.txt` (copy `secrets.txt.example`, `chmod 600`) or the
environment. Set an optional SQLite index of your collection with
`DISTILLERY_INDEX_DB` and "Artist - Album" resolves to exact file paths instantly; the
index needs a `tracks` table with `path`, `rel_path`, `tag_artist`, `tag_album`,
`tag_title`, `tag_tracknumber` and `status` columns. Without one, distillery walks
`<Artist>/<Album>/` directories, which is slower but works.

Files are **copied** locally rather than symlinked, since Essentia and Demucs each
read them more than once and a run should survive the share going away mid-way.

Every cache stage keys on a **fingerprint** (size + mtime) of the source file, not its
filename, so re-fetching an album from a different source can't silently reuse
analysis, stems or loops belonging to different audio.

## The pipeline

| stage | what happens |
|---|---|
| **1. fetch** | Resolve the album, copy it into `data/albums/<slug>/` as `NN_Title.ext`. Every path lookup retries as-is, then NFC, then NFD — macOS stores these decomposed and other layers hand back composed, so an accented path can exist and still fail `os.path.exists`. |
| **2. analyze** | Essentia `MusicExtractor` per track → BPM, beat grid, key/scale/Camelot, LUFS, onset rate, spectral stats. Each track runs in a **child process**, because Essentia can SIGSEGV on a malformed file and that is not catchable in-process. |
| **3. stems** | Demucs `htdemucs`, whole track, cached. Keeps `drums` and `other` — Demucs computes all four sources regardless, so the second is free. |
| **4. loops** | Many small loops per song, scored and cut, then resampled to the chosen tempo. |
| **5. arrange** | A form and length chosen from the material, driven by an energy curve, with the no-same-song rule enforced and this run's bassline and delights rolled. |
| **6. render** | Synth bed + atmosphere + loop bus + texture → effects → sidechain → master. |

## How loops are chosen

1. Use the beat grid Essentia already found — no re-detection.
2. Pick the downbeat phase whose bar-starts are loudest.
3. Enumerate bar-aligned, non-overlapping windows at each bar size (default **0.5, 1
   and 2 bars**).
4. Score each window for loopability from frame-wise MFCC and energy:
   **seam 0.45 + homogeneity 0.35 + tempo stability 0.20**, each min-max normalized
   *per track per bar size*. Normalizing per track matters: the raw distances live on
   different scales, and without it tempo variance dominates the ranking.
5. Gate on level relative to the **full mix's** loud percentile, so loops only come
   from where the drums are genuinely prominent. A track whose drums Demucs couldn't
   find yields few loops or none, rather than a pile of near-silent ones.
6. Render with a **25 ms equal-power wrap crossfade** — the loop's own post-roll faded
   over its head, so the end flows into the start on repeat. An MFCC seam score can be
   perfect while the splice still clicks, so this is handled by construction.

Default yield: up to 6 loops × 3 bar sizes = **18 per song**.

## Length and form

`--length auto` is the default: the piece is sized from **how much distinct material
the album holds**, not from the clock. The loop pool is clustered on what distinguishes
loops (onset density, brightness, length, loopability) and `k` grows until another
cluster stops paying for itself. Then

```
bars = distinct_ideas × exposure ÷ mean_layers
```

`--exposure` (default 16 bars of loop time per idea, about two 8-bar statements) is the
main dial: raise it for something more hypnotic, lower it for something restless.
Length is clamped to 2:30–6:40.

In practice: an album yielding 18 distinct ideas from 144 loops gets about 5:00; one
yielding 10 ideas from 394 loops — a dozen grooves in variation — gets about 3:00.

**Short pieces get fewer sections, not thinner ones**, so the section count follows the
duration:

| form | sections | natural length |
|---|---|---|
| `sketch` | 5 | ~2:15 |
| `single` | 7 | ~4:00 |
| `extended` | 9 | ~6:00 |
| `long` | 10 | ~9:00 |

Override with `--length sketch|single|extended|long`, `--minutes N`, or `--bars N` —
that's the precedence order. Named forms and explicit durations are exact to the bar at
any tempo: `--length long` is 9:00 whether the piece runs at 104 or 144 BPM.

## Tempo

`--bpm auto` is the default. It picks the tempo needing the **least resampling for this
album**, which matters because retiming is plain resampling, so every tempo move is
also a pitch move. The cost is the actual pitch shift each track will get, and it
minimizes the **worst** case first, then the mean.

Minimax rather than total, because the mean often can't tell candidates apart. For an
album with two tempo clusters, ~115 and ~137 BPM:

```
  116 BPM  worst 3.27 st  mean 1.57 st
  120 BPM  worst 2.68 st  mean 1.50 st
  124 BPM  worst 2.11 st  mean 1.50 st
  128 BPM  worst 1.87 st  mean 1.50 st   <-- auto picks this
  132 BPM  worst 2.40 st  mean 1.50 st
  136 BPM  worst 2.92 st  mean 1.50 st
```

The mean is flat from 120 to 136, because moving toward one cluster moves away from the
other by the same amount; only the worst case distinguishes them. Near-ties break
toward the tempo that shifts **upward**, since resampling down darkens everything.

`--bpm 128` forces a tempo. Candidates are multiples of 4 from 104 to 144, the only
tempos giving a whole number of bars in exactly nine minutes.

Two consequences worth knowing:

* **A loop pool belongs to a tempo.** Loops are physically resampled, so the cache is
  keyed by BPM (`data/retimed/<slug>/128bpm/`) and each loop records the tempo it was
  cut for. `rearrange` adopts the pool's tempo automatically.
* **Sample-exactness depends on the tempo.** A beat at 44.1 kHz is a whole number of
  samples only when 2646000/BPM is an integer — of the candidates, that's 108, 112,
  120, 140 and 144. Elsewhere loops land within half a sample (11 µs) of the grid;
  tiling one across a long event drifts at most 0.36 ms, and each event re-anchors to
  the bar grid so nothing accumulates.

## Retiming

Plain resampling, not phase-vocoder stretching: transient smearing is worst on
percussion, and the pitch shift that rides along with a modest tempo change is
inaudible on drums. Half- and double-time are allowed, picking whichever of `r`, `2r`
or `r/2` is closest to 1.0 in log space, so an 84 BPM breakbeat becomes a half-time
loop rather than being dragged to a crawl. Each loop is then snapped to an exact whole
number of target-tempo beats, so everything stays phase-locked to the bed's grid.

## The bed

All synthesized, tuned to the album's key:

| voice | synthesis |
|---|---|
| kick | sine with an exponential pitch drop 118 → 42 Hz, HP'd click, soft-clipped; a muffled variant while energy is low |
| bass | a bassline designed per run — see below |
| hats | band-passed noise, offbeat 8ths, 16ths creeping in as energy rises |
| clap | four noise bursts 9–30 ms apart, band-passed, into reverb |
| perc / rim | brighter noise blips and a two-tone rim on syncopated 16ths |
| drone | three detuned saws, slow attack, LFO'd lowpass, 4.2 s reverb, widened, in 8-bar sustains |
| wash | lowpassed noise breathing on a 17 s / 41 s LFO pair |
| riser / impact | a noise band sweeping up plus a rising sine; a reverb-drenched downbeat hit |

Reverb is convolution with a generated decaying-noise impulse response, via overlap-add
so a nine-minute buffer is cheap. The drone and wash render to their own **atmosphere
bus**, so effects can process the atmosphere without touching the drums.

**Chords are off by default.** The bed holds a single tonal drone — one note, no third
and no seventh, so there is no chord quality and nothing moves harmonically. All the
motion comes from the drums, the loops and the filters. The note is the album's own
key, anchored to one fixed low octave (82–155 Hz) so the register is the same whatever
the key, and high-passed at 150 Hz so it sits above the sub's fundamental instead of
competing for the bottom. `--chords` restores a four-chord progression.

### The bassline

Rolled from the seed, so it differs run to run:

* **rhythm** — one of nine patterns (`long`, `half`, `root-pulse`, `offbeat`, `walk`,
  `octave-push`, `three-two`, `dub-drop`, `sixteenth`)
* **degrees** — a set drawn from a mode-appropriate pool, so the line moves
* **timbre** — drive, lowpass, second-harmonic level, attack, portamento, octave

Density follows the energy curve: the downbeat alone when things are sparse, on-grid
hits in the middle, the full pattern once the piece opens up. The sub is ducked against
the kick so bass and kick never sound in the same instant.

**No bass degree is ever a third** (3 or 4 semitones). The third is what would impose
major or minor on a piece that deliberately has no chords; root, fourth, fifth, sixth,
minor seventh and octave are all consonant over the root drone. Asserted in the
self-test, as is the fact that a seed reproduces its bassline exactly.

## The texture layer

Demucs computes all four sources whichever you ask for, so keeping `other` alongside
`drums` costs no extra compute. It holds everything that isn't drums, bass or vocals:
horns, guitar, keys, and on a dub record much of the production itself.

Texture loops are cut with the same scoring machinery as drums (longer windows — 2 and
4 bars), then made harmonically safe by three guarantees:

1. **Every texture loop is pitch-shifted onto the piece's root**, so it is consonant
   with the drone by construction. The key is detected *after* retiming, because
   retiming is resampling and therefore moves pitch.
2. **A loop with a clearly-stated key must also match the piece's mode.** Rooting a
   major loop on a minor piece still puts a major third against a minor one. A weak key
   reading has no functional third to clash, so it passes as pure texture.
3. **Only one texture sounds at a time**, so no texture can clash with another. They
   also join the same-song bookkeeping.

The shift cap is 4 semitones, and it has to be below 6: the shortest move round the
circle of fifths is always ≤ 6 semitones, so a cap of 6 would reject nothing.

Treatment keeps it texture rather than melody: highpassed at 300 Hz so the lows stay
with the kick, a 5 kHz ceiling with a little air shelf above 4 kHz, long reverb,
two-beat fades, ducked against the kick, and mixed at `--texture-db` (default −22).
`--no-texture` turns the layer off.

### Dub throws

About half of texture events get a **long trailing delay**: the event's last beat — not
the whole loop, which would only smear — is thrown into a tempo-synced,
progressively-darkening, ping-ponged echo that rings on for 4–8 bars past the end of
the event. Tails deliberately spill over the following texture, which is safe for the
reason above: every texture is rooted on the same note.

## Effects

pedalboard takes `(samples, channels)` float32 — the same layout used everywhere here —
so buffers pass through untransposed.

**Always on.** Every loop is shaped by a resonant `LadderFilter` lowpass, with
resonance easing off as energy rises, and the distant loops in quiet sections go
through `Chorus` → `Reverb`.

**Delights.** Two or three effects are rolled per run, placed at real bar positions,
recorded in the plan and printed in the report, so a run you like is reproducible from
its seed:

| delight | what it does |
|---|---|
| `filter-sweep` | resonant lowpass closing 16 k → 600 Hz over the 8 bars into the climax |
| `reverb-throw` | wet-only reverb on the last bar before a transition |
| `dub-echo` | feedback `Delay` throw coming out of a breakdown, lowpassed |
| `drum-crush` | `Bitcrush` + `Distortion` on the loop bus under the climax, peak-matched so a drive stage can't take the master's headroom |
| `phaser-drift` | slow phaser over the atmosphere bed |
| `tape-warp` | `PitchShift` + chorus drift on the loops in a breakdown |

Time-varying filters keep plugin state across chunks (`reset=False`), which is what
stops a sweep clicking at chunk seams. A delight that raises is caught and logged, so it
never costs you the mix. Dynamics are deliberately not run through pedalboard
compression, which flattens the build.

## Mix balance

The borrowed drum loops are the point of the project, so they lead the mix. Measured
against the master on well-recorded albums:

| bus | vs mix |
|---|---|
| synth bed | −1 to −2.5 dB |
| **borrowed drum loops** | **−2.6 to −3.5 dB** |
| texture | −16 to −19 dB |
| drone + wash | −27 dB |

The self-test asserts that drum content leads every section that has drums, that
texture sits at least 8 dB under the mix and 4 dB under the loops, and that the loops
stay within 8 dB of the mix.

Levels are tunable per run: `--loops-db`, `--texture-db`, `--drone-db`.

Two notes if you re-tune any of this. Calibrate on well-recorded albums — records whose
own mixes bury the drums under effects yield weak stems and will pull the defaults off
true. And `return_stems` reports each bus **post-gain and post-normalization**, so
bus-versus-master comparisons mean what they appear to mean.

## Brightness and snap

Several things work together to keep the mix from silting up in the low end:

* **The three low voices don't share an octave.** The kick's fundamental sits around
  42 Hz, the sub above it, and the drone is high-passed at 150 Hz so it contributes
  low-mid warmth rather than competing for the bottom. The sub is ducked against the
  kick, so bass and kick never occupy the same instant.
* **Bed balance favours the percussion that carries brightness**: `BED_LEVELS` keeps
  hats and clap close to the kick rather than well beneath it.
* **A snappy kick**: 0.12 s decay with the click prominent, since the click *is* the
  attack.
* **Short asymmetric event fades** — a quarter to three-quarters of a beat, fast in and
  slower out, so each loop is struck rather than faded in.
* **Transient shaping on the loop bus**: a fast envelope follower minus a slow one, used
  as gain. It is level-compensated, so it changes shape rather than loudness.
* **A master tilt EQ**: +3 dB above 4 kHz, −2 dB below 150 Hz.

Snappy transients mean a high crest factor, so the master limiter shaves 6.5 dB of peak
*relative to the mix's own peak* before peak-normalizing. Shaving relative to the mix
keeps it transient-only, which leaves section-to-section dynamics intact.

## Building excitement

Five things move together across the piece, so it reads as one climb rather than "more
loops appear":

* **layers** 0 → 4 concurrent sample loops
* **lowpass on loops** 1.5 kHz → 19 kHz, the classic filter-opening build
* **loop level** one smooth 12 dB macro envelope from the energy curve
* **reverb on loops** drenched and distant → dry and present
* **bed voices** drone/wash → +kick → +hats → +clap/perc → ghost kicks

The `long` form allocates its bars like this, and every form is scaled proportionally to
whatever length was chosen:

```
intro  pulse  groove-1  build-1  bd-1  groove-2  build-2  bd-2  climax  outro
 16     16      32        32      16     40        32      12     48      26   bars
```

Breakdowns cut against the climb so the climax lands. Loops are highpassed at 140 Hz
and ducked 4 dB on every kick, so borrowed drums sit under the synth kick instead of
fighting it. Energy is applied **once**, as the macro envelope: applying it in more
than one place double-counts and makes the build lumpy.

## The no-same-song rule

**Two loops playing at the same time never come from the same song.** Enforced at
selection time — a candidate is rejected if its track number matches any scheduled event
overlapping the new event's bar range — and re-verified afterwards by
`arrange.check_overlaps()`, which the CLI refuses to render past. There is a subtlety:
an event's span is quantized *up* to a whole number of the chosen loop's repeats, which
can make it longer than the range that was checked, so the check is re-run against the
final span and the pick retried if the growth introduced a clash.

**Except on very small albums.** The rule caps concurrency at the number of songs, so a
two-track album could never stack more than two layers. Below `MIN_SONGS_FOR_RULE` (4)
the rule is relaxed — but a loop may still never overlap **itself**, which would only
comb-filter. Every render states which regime produced it, in the report and in
`plan.json`.

## Using all the cores

Worker counts are resolved per stage from the machine it runs on:
`os.process_cpu_count()` (3.13+) → `sched_getaffinity` (which respects taskset/cpuset
pinning, unlike `os.cpu_count()`) → `os.cpu_count()`, then clamped by any **cgroup CPU
quota** so a 2-CPU container on a 64-core host uses 2, and clamped again by RAM for
stages where each worker holds a whole track. IO-bound work oversubscribes instead.
`--workers N` or `DISTILLERY_WORKERS=N` overrides.

Parallel: Essentia analysis, loop extraction, file copies, per-event loop rendering, and
the distinct pad voicings. Demucs stays serial — it's on the GPU, so concurrent jobs
contend for one device rather than adding throughput. Loop *selection* is serial by
design, since each pick depends on what's already scheduled.

Analysis and loop extraction fan out as **one subprocess per track** rather than a
process pool: Essentia can SIGSEGV on bad audio, and a dead worker takes a whole
`ProcessPoolExecutor` down with it. Here a crashed child costs exactly that track.

**A given seed always produces the same audio, whatever the worker count.** Every
parallel stage is a map over independent items whose results are collected in input
order, so reductions happen on one thread in a fixed order. The self-test asserts that a
6-worker render is bit-identical to a serial one.

## Running it nightly

`distillery nightly` picks an album from the collection, distils it, renders a video
and posts it — built to run unattended from cron:

```bash
./run.sh nightly                      # pick, distil, post
./run.sh nightly --dry-run            # render everything, post nothing
./run.sh nightly --album "Artist - Album"   # skip the picker
./run.sh history                      # what has run, and where it went
```

Album choice takes anything in the index with at least `NIGHTLY_MIN_TRACKS` tracks,
excluding albums whose tracks carry an excluded genre (podcasts, comedy, audiobooks),
anything matching a line in `excluded.txt`, and anything used within
`NIGHTLY_COOLDOWN_DAYS`. Every run is recorded in a state database with what was
posted where, so `history` shows the trail and repeats are avoided.

**Posting is conditional per platform, and each is independent** — one failing never
blocks the other, and neither costs you the audio, which is already on disk:

* **Bluesky rejects videos of three minutes or more.** Since auto lengths run to 6:40,
  the duration is checked *before* uploading and the post is skipped with a reason
  rather than failing mid-upload. `--force-bluesky` tries anyway.
* **Mastodon has a size cap**, which is read from the instance rather than assumed.
  Video is encoded to a size budget (default 80 MB) so a six-minute piece fits as
  easily as a two-minute one.

Credentials go in `secrets.txt`: `BLUESKY_HANDLE` and `BLUESKY_PASSWORD` (an app
password), `MASTODON_INSTANCE_URL` and `MASTODON_ACCESS_TOKEN`. Set
`DISTILLERY_POST=0` or pass `--no-post` to keep everything local.

**Housekeeping.** A run leaves roughly 120 MB of Demucs stems per track plus a copy of
the source album, which unattended would fill a disk in about a year. Both are
re-derivable, so they are pruned once the run succeeds; renders and videos age out
after `NIGHTLY_KEEP_DAYS`. The retimed loop pool is deliberately kept, because it is
small and lets `rearrange` build another piece from that album without a second Demucs
pass. `--no-prune` keeps everything.

A pid lock means a slow run can never overlap the next firing — worth having, since on
a CPU-only host Demucs is by far the slowest stage (about 12 minutes per track on four
cores, versus seconds for everything else).

## Commands

```bash
./run.sh remix --local <dir>|--library "Artist - Album"|--slug <slug> [options]
./run.sh rearrange <slug> --seed 99      # new arrangement from the cached loop pool
./run.sh rearrange <slug> --dry-run      # print the plan, render nothing
./run.sh find "<query>"                  # search the configured collection
./run.sh bed --key A --scale minor       # the synthesized bed alone, no samples
./run.sh list                            # what's cached
./run.sh selftest                        # invariant checks, no album needed
```

Options: `--seed`, `--mp3`, `--limit N` (first N tracks), `--length`, `--minutes`,
`--bars`, `--exposure`, `--bpm`, `--chords`, `--loops-db`, `--texture-db`,
`--drone-db`, `--no-texture`, `--bar-sizes 0.25 0.5 1`, `--per-size N`, `--workers N`,
`--keep-source-loops`, `--force-{download,analyze,stems,loops}`.

Environment: `DISTILLERY_PYTHON`, `DISTILLERY_DATA`, `DISTILLERY_WORKERS`,
`DISTILLERY_BPM`, `DISTILLERY_LENGTH`, `DISTILLERY_EXPOSURE`, `DISTILLERY_CHORDS`,
`DISTILLERY_FX`, `DISTILLERY_TEXTURE`, `DISTILLERY_LOOPS_DB`, `DISTILLERY_TEXTURE_DB`,
`DISTILLERY_DRONE_DB`, `DISTILLERY_INDEX_DB`.

## Performance

On an 18-core M5 Pro, a 22-track album is about **13 seconds of compute** once the audio
is local, plus Demucs on the first pass and the copy off the share. Re-arranging a
cached album takes a few seconds, which is what makes auditioning seeds by ear
practical.

## Layout

```
distillery/       the package: analyze, stems, loops, arrange, techno, render, fx
run.sh            launcher; finds an interpreter and runs `python -m distillery`
data/             everything generated (gitignored): albums, stems, loops, output
secrets.txt       SMB credentials, mode 600, gitignored (see secrets.txt.example)
```

`./run.sh selftest` runs the whole invariant suite — the section allocator, retime
exactness, both overlap rules, harmonic safety of the texture layer, bass-degree safety,
mix balance, worker sizing, and bit-identical output across worker counts. It needs no
album, no network and no models, and it's the fastest way to check that a change didn't
break something.

## License

MIT — see `LICENSE`.
