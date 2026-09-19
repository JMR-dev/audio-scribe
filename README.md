# audio-scribe

Download a video, extract its audio losslessly as FLAC, and transcribe it with
Whisper on whatever GPU the machine actually has.

Resumable from any completed stage, with errors that tell you what to do next.

```bash
uv sync
uv run audio-scribe doctor                      # check the environment first
uv run audio-scribe run 'https://youtu.be/...'
uv run audio-scribe run -i urls.txt --formats txt,srt
```

Hardware setup for the Intel path is documented separately in
[README-intel.md](README-intel.md).

## What it does

```
yt-dlp (aria2c)  ->  ffmpeg  ->  Whisper        ->  txt / srt / vtt / json
   video.mkv         audio.flac   OpenVINO GenAI
```

Each video becomes one job directory under `--workdir` (default `transcripts/`):

```
transcripts/index.json                    <- which jobs each URL expanded to
transcripts/jobs/youtube-<id>/
  state.json          media/video.mkv     out/transcript.{txt,srt,vtt,json}
  events.jsonl        media/audio.flac    logs/{job,aria2}.log
```

## Resume

**Artifacts on disk are the source of truth.** `state.json` records only what the
filesystem cannot answer — chiefly *why* a file is absent. There is deliberately
no stored "current stage": persisting one is how "marked done but the file is
gone" bugs happen. Every run verifies what is actually present (`ffprobe`
duration > 0, non-empty, audio length matching the video) and enters at the first
stage whose output does not check out.

| Transcript | FLAC | Video | Runs |
|---|---|---|---|
| missing | ok | anything | transcribe, outputs |
| missing | missing | ok | extract, transcribe, outputs |
| missing | missing | missing | everything |
| ok | — | — | nothing |

Killing a run mid-download resumes rather than restarting, because yt-dlp keeps
its `.part` and aria2 keeps its control file. How much survives depends on the
checkpoint interval, not on the size of the `.part`: with `--file-allocation=falloc`
the file is full-size from the first second, and only the pieces recorded in the
control file are skipped. A `.part` **without** that control file is discarded and
restarted, because aria2 writes segments out of order, so such a file is sparse
with holes rather than a valid prefix -- resuming from its length would produce a
video that stops being the source partway through.

A video that exists but fails verification is unlinked before re-downloading.
yt-dlp's overwrite default is off, so it would otherwise report the corrupt file
as "already downloaded" and skip it forever. Verification checks the recorded byte
count as well as the duration: Matroska and WebM write the duration into the
header, so a file truncated to a kilobyte still probes as its full length.

Deleting one subtitle file re-renders it from the segments saved in
`transcript.json` instead of re-transcribing the whole recording.

`state.json` is a convenience, not a dependency. If it is unreadable the record
is rebuilt: the job directory is scanned for the video, the FLAC and the written
outputs, `events.jsonl` supplies *why* an artifact is absent so a deletion you
asked for is not mistaken for data loss, and `transcript.json` supplies how the
transcript was produced so a finished job is not transcribed again. Delete it on
a completed job and the next run still reports "already complete".

`index.json` maps each URL to the jobs it expanded to. Re-running a single video
is therefore answered from disk with no metadata request at all — it works with
no network — and a URL that has been seen before still resumes when the source
has become unreadable, saying so. A playlist is always re-read, since that
request is the only way to notice videos added to it since the last run.

`--dry-run` prints the plan for every job without touching anything.

## Keeping or discarding media

Video and audio are kept by default.

| Flag | Effect |
|---|---|
| `--no-retain-video` | delete the video once the FLAC is extracted and verified |
| `--no-retain-audio` | delete the FLAC once every output is written |
| `--no-retain` | both |

Deletion always happens **after** the next stage is committed, so interrupting a
run is still safe and resume still works. What you actually give up is re-run
flexibility: re-transcribing later with a different `--language`, `--model` or
`--task` will need another download.

These flags prompt once per run. On a non-TTY they refuse outright rather than
proceeding — a piped or cron invocation should never delete gigabytes with nobody
seeing the warning. Pass `--yes` to confirm non-interactively.

## Hardware routing

Detection runs in two phases: a cheap hardware probe that imports nothing, then a
toolchain probe that may import. The toolchain probe is only reached once the
hardware is confirmed, which is what keeps torch off a machine with no NVIDIA or
AMD GPU.

Order is **NVIDIA → AMD → Intel GPU → CPU**, and `--device` overrides it.

Fallback happens at runtime, not at detection: constructing the backend and its
first `generate()` are both guarded, because a `/dev/dri` permission failure and
an OpenCL JIT failure surface there rather than at device enumeration. A backend
that fails is demoted for the rest of the process, so a long batch does not retry
it once per job. An explicit `--device` never falls back silently — getting the
CPU when you asked for the GPU would make any measurement a lie.

**NVIDIA and AMD are interface-only in this build.** Detection and routing are
real and the error says what to implement, but there is no inference code: the
cached model is OpenVINO IR, which cannot load on CUDA or ROCm, and neither path
could be tested here. CTranslate2 has no ROCm support, so they are genuinely two
implementations rather than one.

## Download speed

yt-dlp already passes aria2c `-x16 -j16 -s16 --min-split-size 1M`, and `-x16` is
aria2's per-server cap, so re-sending connection counts changes nothing. What
this adds is the flags that do:

| Flag | aria2 default | Why |
|---|---|---|
| `--retry-wait=3` | 0 | retrying with no delay burns all five tries in under a second |
| `--timeout=30` / `--connect-timeout=15` | 60 | a dead connection otherwise holds a slot for a full minute |
| `--lowest-speed-limit=50K` | 0 (off) | without it a wedged connection hangs forever |
| `--disk-cache=64M` | 16M | sixteen writers at out-of-order offsets thrash a small cache |
| `--file-allocation=falloc` | none | instant on btrfs/ext4/xfs; avoids fragmenting multi-GB files |
| `--auto-save-interval=20` | 60 | the control file is what a resume reads; a kill inside the first minute otherwise preserves zero completed pieces |
| `--log=logs/aria2.log` | none | the console level hides everything, leaving no forensics |

`--continue` is deliberately never passed — see the `.part` note above.

aria2c only handles `http/https/ftp/ftps`, so HLS and DASH fall through to
yt-dlp's native downloader; `--concurrent-fragments` is the knob there. A missing
aria2c downgrades with a warning rather than failing.

Because yt-dlp calls an external downloader exactly once and does not retry it,
the retry loop lives here instead.

## Options worth knowing

`--formats txt,srt,vtt,json` · `--language en` · `--task translate` ·
`--playlist` (a URL carrying `&list=` stays one video unless you ask) ·
`--audio-profile whisper` (16 kHz mono FLAC instead of source quality) ·
`--force` / `--force-stage` · `--cookies-from-browser firefox` ·
`--js-runtime` · `-v`

## Requirements

`ffmpeg`, `ffprobe`, and optionally `aria2c`. Everything else comes from
`uv sync`, including the deno binary YouTube needs for its signature challenges —
yt-dlp defaults to deno only, and installing it as a dependency means it works
under cron without any PATH setup.

## Installing

Either route puts `audio-scribe` on your PATH; `audio-scribe doctor` tells you
whether the machine can actually run it.

**As a tool** (needs uv; tracks nothing but what it installed):

```bash
uv tool install .          # then: audio-scribe doctor
uv tool install . --reinstall     # pick up later commits
uv tool install . --editable      # or track the checkout instead
```

**As a self-contained directory** (needs neither Python nor uv at runtime):

```bash
scripts/build-binary.sh    # then: dist/audio-scribe/audio-scribe doctor
```

~450 MB, most of it OpenVINO and its 47 runtime-loaded plugins. It is `onedir`
rather than `onefile` on purpose: `onefile` extracts the whole bundle to `/tmp`
on every launch. Move or copy the whole `audio-scribe/` directory, not just
the executable inside it. `ffmpeg`, `ffprobe` and `aria2c` are still expected on
the system either way — they are not bundled.

## Development

```bash
uv sync
pre-commit install --hook-type pre-commit --hook-type pre-push --hook-type commit-msg
uv run pytest
```

Commits run ruff, pyright and the full suite at 100% coverage; pushes run pylint
and mypy. No unit test touches the GPU, the network or a real model, which is
what keeps the commit gate fast. Live checks sit behind `AUDIO_SCRIBE_LIVE=1`.

Conventional commits are enforced.
