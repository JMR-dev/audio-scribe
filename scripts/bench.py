"""One device per process: wall time + CPU time consumed (offload evidence)."""
import subprocess, sys, time, resource
import numpy as np
from huggingface_hub import snapshot_download
import openvino_genai as ov_genai

SR = 16000


def main():
    dev = sys.argv[1]
    audio = sys.argv[2] if len(sys.argv) > 2 else "sample.wav"
    raw = subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", audio,
                          "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
                         capture_output=True, check=True).stdout
    speech = np.frombuffer(raw, dtype=np.float32)
    secs = len(speech) / SR

    pipe = ov_genai.WhisperPipeline(
        snapshot_download("OpenVINO/whisper-large-v3-turbo-int8-ov"),
        dev, CACHE_DIR=f".ov_cache_{dev.lower()}")

    for i in range(3):
        r0 = resource.getrusage(resource.RUSAGE_SELF)
        t = time.perf_counter()
        pipe.generate(speech, task="transcribe", return_timestamps=True)
        wall = time.perf_counter() - t
        r1 = resource.getrusage(resource.RUSAGE_SELF)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        print(f"{dev} pass{i+1}: wall {wall:6.1f}s  RTF {secs/wall:.2f}x  "
              f"cpu-time {cpu:6.1f}s  ({cpu/wall:4.1f} cores busy)", flush=True)


if __name__ == "__main__":          # forkserver re-imports __main__; guard is required
    main()
