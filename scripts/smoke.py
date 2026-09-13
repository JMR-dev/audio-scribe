"""Throwaway smoke test: prove Whisper runs on the Intel iGPU via OpenVINO GenAI."""

import subprocess
import sys
import time

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from huggingface_hub import snapshot_download

MODEL = "OpenVINO/whisper-large-v3-turbo-int8-ov"
AUDIO = sys.argv[1] if len(sys.argv) > 1 else "sample.wav"
SR = 16000


def decode(path):
    """ffmpeg -> 16 kHz mono float32, no librosa/numba needed."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-loglevel",
            "error",
            "-i",
            path,
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(SR),
            "-",
        ],
        capture_output=True,
        check=True,
    ).stdout
    return np.frombuffer(raw, dtype=np.float32)


def make_pipe(model_dir, device, cache):
    """CACHE_DIR passthrough form is unconfirmed for WhisperPipeline; try each."""
    try:
        return ov_genai.WhisperPipeline(model_dir, device, CACHE_DIR=cache), "kwarg"
    except Exception:
        try:
            return ov_genai.WhisperPipeline(model_dir, device, {"CACHE_DIR": cache}), "dict"
        except Exception:
            ov.Core().set_property(device, {"CACHE_DIR": cache})
            return ov_genai.WhisperPipeline(model_dir, device), "set_property"


def run(model_dir, device, speech, audio_sec):
    cache = f".ov_cache_{device.lower()}"
    print(f"\n{'=' * 62}\n{device}\n{'=' * 62}", flush=True)

    t = time.perf_counter()
    pipe, form = make_pipe(model_dir, device, cache)
    cold = time.perf_counter() - t
    print(f"  compile, cold cache : {cold:7.1f}s   (CACHE_DIR form: {form})", flush=True)

    t = time.perf_counter()
    res = pipe.generate(speech, task="transcribe", return_timestamps=True)
    gen = time.perf_counter() - t
    print(f"  generate            : {gen:7.1f}s   RTF {audio_sec / gen:.2f}x realtime", flush=True)

    del pipe
    t = time.perf_counter()
    pipe, _ = make_pipe(model_dir, device, cache)
    warm = time.perf_counter() - t
    print(f"  compile, warm cache : {warm:7.1f}s", flush=True)
    del pipe

    return {
        "device": device,
        "cold": cold,
        "warm": warm,
        "gen": gen,
        "rtf": audio_sec / gen,
        "text": str(res),
    }


def main():
    print("OpenVINO", ov.__version__)
    devices = ov.Core().available_devices
    print("available_devices:", devices)

    model_dir = snapshot_download(MODEL)
    print("model:", model_dir)

    speech = decode(AUDIO)
    audio_sec = len(speech) / SR
    print(f"audio: {AUDIO}  {audio_sec:.1f}s  ({len(speech):,} samples)")

    targets = [d for d in ("GPU", "CPU") if d in devices]
    if "GPU" not in devices:
        print("\n!! GPU NOT PRESENT -- intel-compute-runtime not installed yet")

    results = [run(model_dir, d, speech, audio_sec) for d in targets]

    print(f"\n{'=' * 62}\nTRANSCRIPT ({results[0]['device']}), first 400 chars\n{'=' * 62}")
    print(results[0]["text"][:400])

    print(f"\n{'=' * 62}\nSUMMARY  ({audio_sec:.0f}s of audio)\n{'=' * 62}")
    print(f"{'device':<8}{'cold':>9}{'warm':>9}{'generate':>11}{'RTF':>9}")
    for r in results:
        print(
            f"{r['device']:<8}{r['cold']:>8.1f}s{r['warm']:>8.1f}s"
            f"{r['gen']:>10.1f}s{r['rtf']:>8.2f}x"
        )
    if len(results) == 2:
        g, c = results[0]["gen"], results[1]["gen"]
        verdict = f"GPU {c / g:.2f}x faster" if g < c else f"GPU {g / c:.2f}x SLOWER"
        print(f"\n  -> {verdict} than CPU on generate")


if __name__ == "__main__":
    main()
