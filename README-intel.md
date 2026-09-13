# Running Whisper on an Intel GPU (Linux)

Setup for GPU-accelerated Whisper transcription on Intel integrated or discrete
graphics, using [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai).

Verified on: Fedora 44, kernel 7.1, Intel Iris Xe (Tiger Lake GT2, Gen12LP, 80 EU),
i5-1145G7, Python 3.14.7, OpenVINO 2026.3.1.

---

## 1. Prerequisites

| Requirement | Notes |
|---|---|
| Intel GPU, Gen9 or newer | OpenVINO's GPU plugin targets Gen9-Gen12LP and Arc. Gen8/9/11 need the `legacy1` runtime packages instead of the mainline ones. |
| Python 3.10+ | 3.14 works; `openvino-genai` ships cp310-cp314 wheels. |
| `ffmpeg` | Used to decode audio. Avoids a `librosa`/`numba` dependency. |

Check which GPU you have and that a kernel driver is bound:

```bash
lspci -nn | grep -Ei 'vga|display'      # e.g. Intel ... [8086:9a49]
lspci -k -s 00:02.0 | grep 'Kernel driver'   # expect i915 (older) or xe (Lunar Lake+)
```

## 2. Install the Intel compute runtime

This is the piece that's usually missing. OpenVINO's GPU plugin is OpenCL-based, so
without it OpenVINO sees only the CPU. The Mesa/Vulkan driver that ships with your
desktop is **not** sufficient.

**Fedora / RHEL:**
```bash
sudo dnf install intel-compute-runtime oneapi-level-zero clinfo
```

**Ubuntu / Debian:**
```bash
sudo apt install intel-opencl-icd libze1 clinfo
```

**Arch:**
```bash
sudo pacman -S intel-compute-runtime level-zero-loader clinfo
```

> Package names other than the Fedora set are the usual spellings but were not
> verified here — check your distro's repos if they don't resolve. On older Intel
> hardware (Gen8/9/11) look for `-legacy1` variants.

**Verify — this must list your GPU before anything else will work:**

```bash
clinfo -l
# Platform #0: Intel(R) OpenCL Graphics
#  `-- Device #0: Intel(R) Iris(R) Xe Graphics
```

### Device permissions

Your user needs access to the render node. Check it:

```bash
ls -l /dev/dri/renderD128
```

If the mode is `0660` (rather than `0666`), add yourself to the owning group and
log out and back in:

```bash
sudo usermod -aG render $USER    # some distros use 'video' instead
```

## 3. Python environment

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install openvino-genai huggingface-hub numpy
```

Install from PyPI, **not** your distro's `python3-openvino` package — distro builds
tend to lag several releases behind and will conflict with the pip install.

**Verify the plugin sees the GPU:**

```bash
python -c "import openvino as ov; print(ov.Core().available_devices)"
# ['CPU', 'GPU']
```

If `clinfo -l` succeeded but `GPU` is missing here, the problem is the OpenVINO
plugin rather than the driver — the two checks are deliberately separate.

## 4. Get a model

Use Intel's pre-converted models; this avoids pulling `torch`, `optimum-intel` and
`nncf` (several GB) just to convert weights:

```bash
hf download OpenVINO/whisper-large-v3-turbo-int8-ov
```

Other sizes exist under the same org — `whisper-{tiny,base,small,medium,large-v3}`
and `distil-whisper-large-v3`, each in `fp16`, `int8` and `int4`. The `int8` builds
are weight-only compressed; on an iGPU the win is memory bandwidth, which is the
binding constraint when the GPU shares system RAM with the CPU.

## 5. Verify end to end

Fetch a test clip and loop it so long-form (>30 s) chunking is exercised:

```bash
curl -sLO https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav
ffmpeg -stream_loop 10 -i jfk.wav -ar 16000 -ac 1 sample.wav   # ~2 min
```

```bash
python scripts/smoke.py sample.wav       # transcript + GPU/CPU timings
python scripts/bench.py GPU sample.wav   # 3 passes, one device per process
python scripts/bench.py CPU sample.wav
```

### Expected results

Measured on Iris Xe (80 EU) + i5-1145G7, `large-v3-turbo-int8`, 121 s of audio:

| device | wall | realtime factor | CPU-time | cores busy |
|---|---|---|---|---|
| GPU | ~23 s | ~5.2x | ~24 s | 1.0 |
| CPU | ~39 s | ~3.1x | ~147 s | 3.8 |

The iGPU is ~1.7x faster and uses ~6x less CPU time. Note it still occupies **one
full core** — the plugin busy-waits — so the offload is 3.8 cores down to 1.0, not
to zero. Budget for that if transcription runs alongside other CPU work.

Don't expect Arc-class numbers from an integrated part: Xe-LP has no XMX matrix
engines, so the gain comes from bandwidth and offload rather than raw matrix
throughput.

**Benchmark with at least a minute of audio.** On a short clip the fixed per-call
overhead dominates and the GPU advantage disappears entirely - on an 11 s sample the
same hardware measured 7.9 s on GPU vs 7.8 s on CPU, a dead heat, versus the 1.7x
gap on 121 s. Short-clip numbers will tell you the GPU is pointless when it isn't.

## 6. Usage notes

**Enable the kernel cache.** The GPU plugin JIT-compiles OpenCL kernels; `CACHE_DIR`
works as a plain keyword argument:

```python
pipe = ov_genai.WhisperPipeline(model_dir, "GPU", CACHE_DIR=".ov_cache_gpu")
result = pipe.generate(speech, task="transcribe", return_timestamps=True)
for c in result.chunks:
    print(c.start_ts, c.end_ts, c.text)
```

Model load is ~7 s cold and ~0.7 s cached. Separately, the **first transcription**
on a fresh cache costs ~31 s vs ~23 s steady-state, because shape-specific kernels
are compiled lazily at first inference rather than at load. A slow first run is
expected, not a fault.

**Audio must be 16 kHz mono float32.** Decode with ffmpeg rather than a Python audio
library:

```python
raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", path,
                      "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                     capture_output=True, check=True).stdout
speech = np.frombuffer(raw, dtype=np.float32)
```

Long-form audio needs no chunking code of your own — `WhisperPipeline` applies a
sequential 30-second sliding window internally.

**Always guard your entrypoints:**

```python
if __name__ == "__main__":
    main()
```

Python 3.14 changed the default multiprocessing start method on Linux from `fork` to
`forkserver`, which re-imports `__main__`. Without the guard, a script whose
dependencies touch multiprocessing runs a **second copy of itself concurrently**. It
fails silently — the symptom is duplicated output lines and timings inflated by
self-contention. This cost us a benchmark that reported the GPU 3x slower than it is.

## 7. Troubleshooting

| Symptom | Cause |
|---|---|
| `clinfo -l` lists no platform | Compute runtime not installed, or too old for your GPU. |
| `clinfo` works, OpenVINO shows only `['CPU']` | Plugin problem, not a driver problem. Check you're in the venv and not shadowed by a distro `python3-openvino`. |
| `Permission denied` on `/dev/dri/renderD128` | Not in the `render` (or `video`) group; needs a re-login. |
| First run very slow, later runs fine | Kernel JIT populating `CACHE_DIR`. Expected. |
| Output lines appear twice; everything slow | Missing `if __name__ == "__main__"` guard — see above. |

## Alternative: whisper.cpp + Vulkan

If the OpenCL stack won't cooperate, whisper.cpp's Vulkan backend needs no Intel
runtime at all — the Mesa driver most desktops already have is enough:

```bash
git clone https://github.com/ggml-org/whisper.cpp && cd whisper.cpp
cmake -B build -DGGML_VULKAN=ON && cmake --build build -j$(nproc)
```

Generally slower than OpenVINO on Intel hardware, but a useful fallback and an
independent check on your numbers.
