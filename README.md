# distillery

**Distills a whole album into one ambient techno track.**

Point it at an album. It analyzes every song with Essentia, isolates each song's
drum track with Demucs, cuts those drums into many small beat-aligned loops, retimes
them all to one tempo, and lays them over a **fully synthesized** ambient techno bed
— arranged as a single long build, with pedalboard effects.

Nothing in the bed is sampled: the kick, bass, hats, clap, drone and noise wash are
all generated from oscillators and filtered noise, tuned to the album's own key. The
only recorded audio in the output is the album's drums (and optionally its melodic
"other" stem, used as texture).

```bash
./run.sh remix --local ~/Music/SomeAlbum --mp3
```

Output lands in `data/output/` as `distillery_<slug>_<timestamp>.wav`, alongside a
`.txt` report and a `.plan.json` describing every decision the arranger made.

Each run picks its own **tempo**, **length**, **form**, **bassline** and **effects**
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

macOS or Linux, Python 3.12, and `ffmpeg` on PATH. Two caveats worth knowing before
you start:

* **Essentia has no general arm64 macOS wheel** on PyPI and no brew formula — only
  specific dev builds ship one. `requirements.txt` pins `essentia==2.1b6.dev1389`,
  which has cp310–cp313 arm64 macOS wheels. On Intel/Linux the ordinary release works.
* **Demucs wants a GPU.** It uses Apple MPS automatically on Apple silicon (~5–30 s
  per track). On CPU it still works, just several times slower. This is the slowest
  stage by far; everything else is seconds.

Optional: `yt-dlp` (only for the YouTube fallback source), and an SMB share plus
`mount_smbfs` (macOS) for `--library`.

## Install

```bash
git clone https://github.com/jeffehobbs/distillery
cd distillery
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./run.sh selftest          # ~180 checks, no album or network needed
```

`run.sh` finds an interpreter in this order: `$DISTILLERY_PYTHON`, `./.venv`, then a
sibling `essentia-explorer` venv. The first Demucs run downloads its model (~80 MB)
from HuggingFace and caches it.

## Quickstart

```bash
# a folder of audio files you already have
./run.sh remix --local ~/Music/SomeAlbum --mp3

# an album on an SMB share (see "Where the audio comes from")
./run.sh find "bumps"
./run.sh remix --library "Artist - Album" --mp3

# try other arrangements of an album you've already processed — a few seconds each
./run.sh rearrange <slug> --seed 99 --mp3
./run.sh list
```

Everything caches, so the expensive stages run once per album. Re-arranging is fast
enough to audition seeds by ear.

## A note on what you feed it

This is a tool for making something new out of records **you own**. It reads local
files; the `--album` flag that searches YouTube exists because it's how the project
started, and it's the worst-sounding option anyway. Whatever you make from
copyrighted material is yours to keep to yourself — don't assume a distillation is
free of the original's rights just because it's mostly synthesized.

## Where the audio comes from

`--library "Artist - Album"` pulls an album from a local collection on an SMB share
(configure the host and share in `secrets.txt` — copy `secrets.txt.example`). If the
share isn't mounted, distillery mounts it with
`mount_smbfs` using credentials from `secrets.txt` (mode 600, gitignored) or the
environment — nothing is hardcoded. Album lookup goes through the
an optional SQLite index of your collection when one is configured
(`DISTILLERY_INDEX_DB`), which resolves "Artist - Album" to exact file paths
instantly; walking tens of thousands of files over SMB takes minutes. Without an
index it walks `<Artist>/<Album>/` directories instead. Files are **copied** locally, not symlinked, since Essentia and
Demucs each read them more than once and the run should survive the share going away.

Other sources: `--url` (yt-dlp playlist), `--local <dir>`, `--slug` (re-use a fetched
album), and `--album "Artist - Album"` which finds tracks on YouTube via a
MusicBrainz tracklist. That last one still works but the audio quality is poor —
`--library` is the reason it's no longer the default path.

Every cache stage keys on a **fingerprint** (size + mtime) of the source file, not
just its filename. Re-fetching the same album from a better source produces
identical filenames, and stale analysis, stems, or loops would otherwise be reused
silently against completely different audio.

## The pipeline

| stage | what happens |
|---|---|
| **1. fetch** | Resolve the album (index or share walk), copy it into `data/albums/<slug>/` as `NN_Title.ext`. Unicode NFC/NFD is retried on every path — macOS records these decomposed and other layers hand back composed. |
| **2. analyze** | Essentia `MusicExtractor` per track → BPM, beat grid, key/scale/Camelot, LUFS, onset rate, spectral stats. Each track runs in a **child process** because Essentia can SIGSEGV on a malformed file, and that's uncatchable in-process. |
| **3. stems** | Demucs `htdemucs` on MPS, whole track, cached. Keeps `drums` and `other` — Demucs computes all four sources regardless, so the second one is free. |
| **4. loops** | Many small loops per song, cut and scored with the method below, then resampled to the chosen tempo. |
| **5. arrange** | A form and length chosen from the material, driven by an energy curve, with the no-same-song rule enforced and this run's bassline and delights rolled. |
| **6. render** | Synth bed + atmosphere + loop bus → effects → sidechain → master. |

## How loops are chosen

The method `mashup-app` inherited from `essentia-explorer`'s `essexp/loops.py`,
applied here to the Demucs drums stem:

1. Use the beat grid Essentia already found — no re-detection.
2. Pick the downbeat phase whose bar-starts are loudest.
3. Enumerate bar-aligned, non-overlapping windows at each bar size (default
   **0.5, 1, and 2 bars** — half-bar loops are the "much smaller" end).
4. Score each window for loopability from frame-wise MFCC/energy:
   **seam 0.45 + homogeneity 0.35 + tempo stability 0.20**, each min-max normalized
   *per track per bar size*. Without that normalization raw tempo cv dominates and
   the ranking is meaningless.
5. Gate on level relative to the **full mix's** loud percentile, so loops only come
   from where the drums are genuinely prominent.
6. Render with a **25 ms equal-power wrap crossfade** — the loop's own post-roll
   faded over its head, so the end flows into the start on repeat. An MFCC seam
   score can be perfect while the splice still clicks; 25 ms was picked by ear.

Default yield: up to 6 loops × 3 bar sizes = **18 loops per song**.

## Length and form

`--length auto` is the default: the piece is sized from **how much distinct material
the album actually holds**, not from the clock. The loop pool is clustered on what
distinguishes loops (onset density, brightness, length, loopability) and `k` grows
until another cluster stops paying for itself; then

```
bars = distinct_ideas × exposure ÷ mean_layers
```

`--exposure` (default 16 bars of loop time per idea, about two 8-bar statements) is
the real dial — raise it for something more hypnotic, lower it for something
restless. Length is clamped to 2:30–6:40.

Measured on real albums: On The Corner yields 18 distinct ideas from 144 loops and
gets 5:00; Bumps yields 10 from 394 loops (a dozen grooves in variation) and gets
2:56. A fixed nine minutes gave both the same runtime and restated every idea three
times, which is what "unfocused" sounds like.

**Short pieces get fewer sections, not thinner ones.** Scaling one 10-section
template down to three minutes produced a 4-bar "breakdown", which isn't a breakdown.
So the section count follows the duration:

| form | sections | natural length |
|---|---|---|
| `sketch` | 5 | ~2:15 |
| `single` | 7 | ~4:00 |
| `extended` | 9 | ~6:00 |
| `long` | 10 | ~9:00 |

Override with `--length sketch|single|extended|long`, `--minutes N`, or `--bars N`
(that's the precedence order). Named forms and explicit durations are exact to the
bar at any tempo — `--length long` is 9:00 whether the piece is at 104 or 144 BPM.

## Tempo

`--bpm auto` is the default: it picks the tempo that needs the **least resampling
for this album**, which matters because retiming is plain resampling, so every
tempo move is also a pitch move. The cost function is the actual pitch shift each
track will get (using the same half/double rate choice `retime()` makes), and it
minimizes the **worst** case first, then the mean.

Minimax rather than total, because summed cost can't tell the difference. On The
Corner sits in two clusters, ~115 and ~137 BPM:

```
  116 BPM  worst 3.27 st  mean 1.57 st
  120 BPM  worst 2.68 st  mean 1.50 st
  124 BPM  worst 2.11 st  mean 1.50 st
  128 BPM  worst 1.87 st  mean 1.50 st   <-- auto picks this
  132 BPM  worst 2.40 st  mean 1.50 st
  136 BPM  worst 2.92 st  mean 1.50 st
```

The mean is flat from 120 to 136 — moving toward one cluster moves away from the
other by the same amount — so only the worst case distinguishes them. At 120 the
fast half gets dragged down 2.7 semitones; at 128 nothing moves more than 1.9.

`--bpm 128` forces a tempo. Candidates are multiples of 4 from 104 to 144, because
only those give a whole number of bars in exactly 540 s — anything else lands up to
a bar short of nine minutes.

Two things worth knowing:

* **The loop pool belongs to a tempo.** Loops are physically resampled, so the
  cache is keyed by BPM (`data/retimed/<slug>/128bpm/`) and each loop records the
  tempo it was cut for. `rearrange` adopts the pool's tempo automatically. Keying
  only on the album meant a BPM change silently reused loops cut at the old tempo
  under a bed at the new one.
* **Sample-exactness depends on the tempo.** A beat at 44.1 kHz is a whole number
  of samples only when 2646000/BPM is an integer — of the candidates, that's 108,
  112, 120, 140 and 144. Elsewhere loops land within half a sample (11 µs) of the
  grid; tiling a 2-bar loop across a long event drifts at most 0.36 ms, and each
  event re-anchors to the bar grid so nothing accumulates across the piece.

## Retiming

Plain resampling, not phase-vocoder stretching: transient smearing is worst on
percussion, and the pitch shift that rides along with a modest tempo change is
inaudible on drums. Half- and double-time are allowed, picking whichever of `r`,
`2r`, `r/2` is closest to 1.0 in log space, so an 84 BPM breakbeat becomes a
half-time 120 BPM loop instead of being dragged to a crawl. Each loop is then
snapped to an exact whole number of target-tempo beats, so everything stays
phase-locked to the bed's grid.

## The bed (all synthesized, no chords)

| voice | synthesis |
|---|---|
| kick | sine with exponential pitch drop 118 → 42 Hz, HP'd click, soft-clipped; swaps to a muffled variant while energy is low |
| bass | a bassline **designed per run** from the seed — see below |
| hats | band-passed noise, offbeat 8ths, 16ths creeping in as energy rises |
| clap | four noise bursts 9–30 ms apart, band-passed, into reverb |
| perc / rim | brighter noise blips and a two-tone rim on syncopated 16ths |
| drone | three detuned saws on **root + octave**, slow attack, LFO'd lowpass, 4.2 s reverb, widened, held in 8-bar sustains |
| wash | lowpassed noise breathing on a 17 s / 41 s LFO pair |
| riser / impact | noise band sweeping up + rising sine; reverb-drenched downbeat hit |

### The bassline

The bass used to be a single root note in one of two hardcoded rhythms with fixed
timbre — identical on every album and every seed, with only its pitch class changing.
That one static element made whole runs sound alike. It's now rolled from the seed:

* **rhythm** — one of nine patterns (`long`, `half`, `root-pulse`, `offbeat`, `walk`,
  `octave-push`, `three-two`, `dub-drop`, `sixteenth`)
* **degrees** — a set drawn from a mode-appropriate pool, so the line can move
* **timbre** — drive, lowpass, second-harmonic level, attack, portamento, octave

Density follows the energy curve: the downbeat alone when things are sparse, on-grid
hits in the middle, the full pattern once the piece opens up.

**No bass degree is ever a third** (3 or 4 semitones). The third is precisely what
would impose major or minor on a piece that deliberately has no chords; root, fourth,
fifth, sixth, minor seventh and octave are all consonant over the root drone. That's
asserted in the self-test, as is the fact that a given seed still reproduces its
bassline exactly.

**Chords are off.** The bed holds a single tonal drone — root and octave, no third
and no seventh, so there's no chord quality and nothing moves harmonically. All the
motion comes from the drums, the loops, and the filters. The pitch is still the
album's own key (most common Essentia key across its tracks). `--chords` brings back
the old four-chord progression if you want to hear it.

Reverb is convolution with a generated decaying-noise IR (overlap-add, so a 9-minute
buffer is cheap). The drone/wash live on their own **atmosphere bus** so effects can
process the atmosphere without touching the drums.

## The texture layer (the "other" stem)

Demucs computes all four sources whichever ones you ask for, so keeping `other`
alongside `drums` costs **no extra compute** — it was being computed and discarded.
It holds everything that isn't drums/bass/vocals: horns, guitar, keys, and on a dub
record much of the production itself.

Texture loops are cut with the same scoring machinery as drums (longer windows —
2 and 4 bars), then made harmonically safe by three guarantees rather than by luck:

1. **Every texture loop is pitch-shifted onto the piece's root**, so it is consonant
   with the drone by construction. The key is detected *after* retiming, because
   retiming is resampling and therefore moves pitch — detecting on the source would
   describe a loop that no longer exists.
2. **A loop with a clearly-stated key must also match the piece's mode.** Rooting a
   major loop on a minor piece still puts a major third against a minor one. A weak
   key reading has no functional third to clash, so it passes as pure texture.
3. **Only one texture sounds at a time**, so no texture can clash with another. They
   also join the same-song bookkeeping, so the no-two-loops-from-one-song rule
   covers them too.

The shift cap is 4 semitones, and it has to be **below 6**: the shortest move round
the circle of fifths is always ≤ 6 semitones, so a cap of 6 rejects nothing at all.

Treatment keeps it texture rather than melody: highpassed at 200 Hz so the lows stay
with the kick, lowpassed to ≤3.2 kHz, long reverb, 4-beat fades, ducked against the
kick, and mixed at `--texture-db` (default −9). `--no-texture` turns it off.

Measured on Lounge Lizards: adding the texture layer *lowered* spectral dissonance
(0.4373 → 0.4186) and sharpened the detected key from Ab minor at 0.807 confidence
to Ab major at 0.873.

### Dub throws

About half of texture events get a **long trailing delay**: the event's last beat —
not the whole loop, which just smears — is thrown into a tempo-synced,
progressively-darkening, ping-ponged echo that rings on for 4–8 bars past the end of
the event. Tails deliberately spill over the following texture, and that is safe for
exactly the reason above: every texture is rooted on the same note.

A throw has to be pushed **up** to read at all — mashup-app's first delay
implementation was inaudible because the grabbed beat sat far below the mix it rang
into. Here the throw is boosted 5 dB and decays 3 dB per repeat, which measures at
about 5 dB under the mix it rings into, and *raises* overall level rather than
lowering it (Lee Perry: −21.1 → −19.9 dBFS RMS).

## Effects (pedalboard)

pedalboard takes `(samples, channels)` float32 — the same layout used everywhere
here — so buffers pass straight through untransposed. Two layers:

**Always on.** Every loop is shaped by a resonant `LadderFilter` lowpass rather than
a plain butterworth (resonance eases off as energy rises, so early loops are
filtered and vocal-sounding and the climax is open), and the drenched loops in the
quiet sections go through `Chorus` → `Reverb`.

**Delights.** Two or three effects are rolled per run by the arranger, placed at
real bar positions, recorded in the plan, and printed in the report — so a run you
like is reproducible from its seed:

| delight | what it does |
|---|---|
| `filter-sweep` | resonant lowpass closing 16 k → 600 Hz over the 8 bars into the climax |
| `reverb-throw` | wet-only reverb on the last bar before a transition |
| `dub-echo` | feedback `Delay` throw coming out of a breakdown, lowpassed |
| `drum-crush` | `Bitcrush` + `Distortion` on the loop bus under the climax, **peak-matched** so a drive stage can't steal the master's headroom |
| `phaser-drift` | slow phaser over the atmosphere bed |
| `tape-warp` | `PitchShift` + chorus drift on the loops in a breakdown |

Time-varying filters keep plugin state across chunks (`reset=False`), which is what
stops a sweep clicking at chunk seams. A delight that raises is caught and logged —
it never costs you the mix. Dynamics are deliberately *not* run through pedalboard
compression: that flattens the build (see below).

## Brightness and snap

The mixes came out dark and plodding, and measurement found the cause in the one
place that wasn't obvious: the **synth bed** was 98.4% of its energy below 200 Hz
with a 99 Hz centroid — its own kick and sub swamped its hats, clap and perc, which
sat 10-12 dB beneath them. Since the bed leads the mix, the master inherited the
tilt (73% sub-200 Hz, where techno sits nearer 35-50%). The borrowed drum loops were
already the brightest bus in the piece; they were simply buried.

Four changes, and the measured result across six albums:

* **Bed rebalance** (`config.BED_LEVELS`): sub −7.5 → **−13 dB**, hats −13 → **−8**,
  clap −11 → −6.5, perc −15 → −10.
* **Snappier kick**: decay 0.18 → 0.12 s and click 0.32 → 0.45 — the click *is* the
  attack, and it had been softened earlier to buy headroom.
* **Short, asymmetric event fades**: 1–2 beats → 0.25/0.75 beats (fast in, slower
  out). At 120 BPM the old setting was a half to a full *second* of ramp on every loop
  entry — each one faded in rather than struck. Textures went 4 → 2 beats.
* **Master tilt EQ**: +3 dB above 4 kHz, −2 dB below 150 Hz.

| album | low-share before → after | centroid before → after |
|---|---|---|
| On The Corner | 56% → **29%** | 1227 → **2302 Hz** |
| Fela | 73% → **50%** | 655 → **1366 Hz** |
| Kid Koala | 73% → **50%** | 470 → **1085 Hz** |
| Ron Miles | 63% → **47%** | 696 → **1134 Hz** |
| Lee Perry | 88% → **65%** | 340 → **1081 Hz** |
| Lounge Lizards | 89% → **72%** | 321 → **953 Hz** |

A second pass went after the causes rather than the tilt:

* **The three low voices no longer share an octave.** Kick fundamental ~42 Hz, sub
  65–123 Hz and drone 82–155 Hz were all crowded together. The drone is now
  high-passed at 150 Hz — above the sub's fundamental — so it contributes low-mid
  warmth instead of competing for the bottom, and **the sub is ducked against the
  kick** (only the pad was before), so bass and kick never sound in the same instant.
* **Loop lowpass floor 700 → 1500 Hz**, so the borrowed drums are bright through the
  middle of the energy curve rather than only at the very top.
* **Transient shaping on the loop bus** — a fast envelope follower minus a slow one,
  used as gain, which is what "snappy" means mechanically. It is **level-compensated**:
  the shaper only ever boosts, so without matching RMS afterwards the master's
  peak-normalization simply hands the gain back.
* **Duck release 170 → 90 ms** (slow recovery reads as pumping, not groove), and a
  gentler soft-clip that isn't rounding off the transients.
* **Loop reverb** `0.06 + 0.44(1−e)` → `0.04 + 0.26(1−e)^1.5` — it was 37% wet at mid
  energy, smearing the whole first half of every piece.
* **Texture gets air**: ceiling 3.2 → 5 kHz plus a 2.5 dB shelf at 4 kHz. At −22 dB,
  highs read as sheen rather than presence, so this doesn't undo the level work.
* **The auto-BPM tie-break now prefers upward shifts**, since resampling down darkens
  everything systematically.

Net result across six albums:

| album | low-share | centroid | crest |
|---|---|---|---|
| On The Corner | 56% → **31%** | 1227 → **2278 Hz** | 10.0 → **12.0 dB** |
| Kid Koala | 73% → **54%** | 470 → **1059 Hz** | 9.4 → **11.5 dB** |
| Fela | 73% → **54%** | 655 → **1151 Hz** | 9.0 → **11.2 dB** |
| Ron Miles | 63% → **51%** | 696 → **1049 Hz** | 9.6 → **11.8 dB** |
| Lee Perry | 88% → **69%** | 340 → **898 Hz** | 8.2 → **10.9 dB** |
| Lounge Lizards | 89% → **75%** | 321 → **862 Hz** | 8.6 → **10.8 dB** |

Snappier transients raise the crest factor, and removing sustained low energy costs
RMS (low frequencies carry most of it), so `MASTER_SHAVE_DB` went 3.5 → 5.0 → 6.5 dB.
Raising the kick to buy that level back was measured and rejected: it cost 5 points of
low-share and 90 Hz of centroid for 0.4 dB.

Two measurement traps worth knowing before you re-tune any of this: an analysis
snippet must call `config.set_tempo()` or every bar index is wrong, and it must
measure the **whole file** — a one-megasample FFT covers just the first 24 seconds,
which is the drone-only intro on every render, and will tell you the mix is 98%
sub-bass when it isn't.

## Mix balance — drums first

The borrowed drum loops are the point of the project, so they lead the mix. That is
asserted, not assumed: `LOOP_LEVEL_DB` is +9 dB and the loop-bus macro envelope
spans 12 dB (not 17, which buried the drums in the quieter sections).

Measured against the master on the well-recorded albums:

| bus | vs mix |
|---|---|
| synth bed (kick/hats/clap/perc) | −1 to −2.5 dB |
| **borrowed drum loops** | **−2.6 to −3.5 dB** |
| texture | −16 to −19 dB |
| drone + wash | −27 dB |

Before this calibration the loops measured **−19.7 dB under the mix and led no
section at all** — in the outro the texture layer beat them outright. The self-test
now asserts that drum content leads every section that has drums, that texture sits
at least 8 dB under the mix and 4 dB under the loops, and that the loops stay within
8 dB of the mix.

Two traps worth knowing if you re-tune any of this:

* **Calibrate on well-recorded albums.** Levels were set from On The Corner, Bumps
  and Ron Miles, deliberately not from the Lee Perry — Black Ark mixes bury the drums
  under effects, so its extracted stems are weak and would drag the defaults off.
* **`return_stems` is post-gain for a reason.** Stems used to be snapshotted before
  their bus gains and before the master's normalization scalar, which made every
  bus-vs-master comparison wrong: an +11.5 dB change to the loop bus appeared to move
  the loops by 1 dB. Tuning against those numbers was measuring nothing.

Levels are tunable per run: `--loops-db`, `--texture-db`, `--drone-db`.

## Building excitement

Five things move together across the piece, so it reads as one climb rather than
"more loops appear":

* **layers** 0 → 4 concurrent sample loops
* **lowpass on loops** 1.5 kHz → 19 kHz (the classic filter-opening build)
* **loop level** one smooth 12 dB macro envelope from the energy curve
* **reverb on loops** drenched and distant → dry and present
* **bed voices** drone/wash → +kick → +hats → +clap/perc → ghost kicks

The `long` form, for example, allocates its bars like this — and every form is
scaled proportionally to whatever length was chosen:

```
intro  pulse  groove-1  build-1  bd-1  groove-2  build-2  bd-2  climax  outro
 16     16      32        32      16     40        32      12     48      26   bars
```

Breakdowns cut against the climb so the climax lands. Loops are highpassed at
140 Hz and ducked 4 dB on every kick, so borrowed drums sit under the synth kick
instead of fighting it.

Two things that had to be fixed to make this real:

* A **fixed** master limiter ceiling destroyed the build — every section came out
  peaking at exactly the ceiling, flattening nine minutes of dynamics to about 3 dB.
  It now limits *relative* to the mix's own peak (`MASTER_SHAVE_DB`), shaving only
  transients, then peak-normalizes.
* Energy is applied **once**, as the macro envelope. Baking it into per-event gains
  as well double-counted it and made the build lumpy.

## The no-same-song rule

**Two loops playing at the same time never come from the same song.** Enforced at
selection time — a candidate is rejected if its track number matches any already
scheduled event overlapping the new event's bar range — and re-verified after the
fact by `arrange.check_overlaps()`, which the CLI refuses to render past. Note the
subtlety: an event's span is quantized *up* to a whole number of the chosen loop's
repeats, which can make it longer than the range that was checked, so the check is
re-run against the final span and the pick retried if growth introduced a clash.

**Except on very small albums.** The rule caps concurrency at the number of songs, so
a 2-track album could never stack more than two layers and the 0→4 build was
impossible. Below `MIN_SONGS_FOR_RULE` (4) the rule is relaxed — but a loop may still
never overlap **itself**, which would only comb-filter. Every render states which
regime produced it, in the report and in `plan.json`. On a two-track Fela album the
relaxation took mean layers from 1.23 to 2.16 and the peak from 2 to 5.

## Using all the cores

Worker counts are resolved per stage from the machine it's running on, not
hardcoded: `os.process_cpu_count()` (3.13+) → `sched_getaffinity` (respects
taskset/cpuset pinning, which `os.cpu_count()` ignores) → `os.cpu_count()`, then
clamped by any **cgroup CPU quota** so a 2-CPU container on a 64-core host uses 2,
and clamped again by RAM for stages where each worker holds a whole track. IO-bound
work (copying off the NAS) oversubscribes instead, since those workers sit blocked
on the network. `--workers N` or `DISTILLERY_WORKERS=N` overrides everything.

Parallelized: essentia analysis, loop extraction, NAS copies, per-event loop
rendering, and the distinct pad voicings. Demucs stays serial — it's on the GPU
(MPS), so concurrent jobs contend for one device rather than adding throughput.
Loop *selection* also stays serial by design: each pick depends on what's already
scheduled (the no-same-song rule), and it costs milliseconds anyway.

Analysis and loop extraction fan out as **one subprocess per track** rather than a
process pool. That's deliberate: Essentia can SIGSEGV on bad audio, and a dead
worker takes a whole `ProcessPoolExecutor` down with it — the failure mode
essentia-explorer's indexer had to be rewritten around. Here a crashed child costs
exactly that track. Threads supervise, blocked on the child.

**A given seed always produces the same audio, whatever the worker count.** Every
parallel stage is a map over independent items whose results are collected in input
order, so reductions (mixing) happen on one thread in a fixed order. The self-test
asserts a 6-worker render is bit-identical to a serial one.

22-track album, all compute stages forced to re-run, on an 18-core M5 Pro:

| stage | serial | parallel |
|---|---|---|
| essentia analysis | 46.2 s | **5.0 s** |
| loop extraction | 35.3 s | **3.7 s** |
| demucs stems | 3.0 s | 0.5 s (cached) |
| render | 6.6 s | **3.9 s** |
| **total compute** | **91 s** | **13 s** |

Worth knowing how that render number got there, because it wasn't threads: the bed
originally took **273 s**, of which 268 s was a single `savgol_filter` call
smoothing the wash envelope with a 44,101-sample window — a direct convolution over
24M samples, roughly a trillion operations, to smooth what is really a
one-value-per-bar curve. Interpolating from the 270 bar values instead is O(n) and
continuous by construction: **273 s → 3.0 s** before any parallelism. Profile
first; the loop events everyone would assume were the bottleneck were 2.6 s.

## Commands

```bash
./run.sh find "<query>"                  # search the local collection
./run.sh remix --library "Artist - Album" --mp3
./run.sh rearrange <slug> --seed 99      # new arrangement from the cached loop pool
./run.sh rearrange <slug> --dry-run      # print the plan, render nothing
./run.sh bed --key A --scale minor       # the synthesized bed alone, no samples
./run.sh list                            # what's cached
./run.sh selftest                        # invariant checks, no album needed
```

Useful flags: `--limit N` (first N tracks), `--bars N` (length; 270 = 9:00 at 120),
`--seed`, `--chords`, `--drone-db X`, `--workers N`, `--bar-sizes 0.25 0.5 1`,
`--per-size N`, `--mp3`, `--keep-source-loops`, and
`--force-{download,analyze,stems,loops}`.
Env: `DISTILLERY_PYTHON`, `DISTILLERY_DATA`, `DISTILLERY_CHORDS=1`,
`DISTILLERY_FX=0`, `DISTILLERY_DRONE_DB`, `DISTILLERY_WORKERS`.

Every stage caches, so re-runs and re-arrangements are fast. The loop pool caches
per track, so adding songs to an album only processes the new ones.

## Performance

On an 18-core M5 Pro, a 22-track album is about **13 seconds of compute** once the
audio is local, plus Demucs on the first pass (its model runs on the GPU and is the
one slow stage) and the copy off the share. Re-arranging a cached album takes a few
seconds, which is what makes auditioning seeds by ear practical.

## Layout

```
distillery/       the package: analyze, stems, loops, arrange, techno, render, fx
run.sh            launcher; finds an interpreter and runs `python -m distillery`
data/             everything generated (gitignored): albums, stems, loops, output
secrets.txt       SMB credentials, mode 600, gitignored (see secrets.txt.example)
```

`./run.sh selftest` runs the whole invariant suite — allocator, retime exactness, the
overlap rules, harmonic safety of the texture layer, bass-degree safety, mix balance,
worker sizing, and bit-identical output across worker counts. It needs no album, no
network and no models, and it is the fastest way to check a change didn't break
something.

## License

MIT — see `LICENSE`.
