"""Throughput plus peak RAM / VRAM per job.

    python benchmarks/memory_bench.py <folder> [<folder> ...] --frames 64 --video-out <dir>

VRAM is CuPy's pool (what this process holds on the GPU); RAM is the
process's peak resident memory above its starting point.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timelapse_engine as te  # noqa: E402
from timelapse_engine import backend, files  # noqa: E402


class PeakMemory:
    def __init__(self):
        self.proc = psutil.Process()

    def __enter__(self):
        backend.free_gpu_memory()
        self.base_ram = self.proc.memory_info().rss
        self.peak_ram = self.base_ram
        self.peak_vram = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def _poll(self):
        pool = backend.cp.get_default_memory_pool() if backend.HAS_CUPY else None
        while not self._stop.wait(0.02):
            self.peak_ram = max(self.peak_ram, self.proc.memory_info().rss)
            if pool is not None:
                self.peak_vram = max(self.peak_vram, pool.total_bytes())

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    def __str__(self):
        return f"RAM +{(self.peak_ram - self.base_ram) / 1e9:5.1f} GB   VRAM {self.peak_vram / 1e9:5.1f} GB"


def job(label, n_out, fn):
    with PeakMemory() as mem:
        t = time.perf_counter()
        fn()
        dt = time.perf_counter() - t
    print(f"  {label:<40} {n_out / dt:6.1f} fps   {mem}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--frames", type=int, default=64)
    ap.add_argument("--processors", default="average,maximum,temporal_entropy,motion_overlay")
    ap.add_argument("--video-out")
    args = ap.parse_args()

    for folder in args.folders:
        paths = sorted(files.list_images(folder), key=files.natural_sort_key)[: args.frames]
        seq = te.ImageSequence(paths=paths, sort="name", validate=False)
        shape, dtype = seq.probe()
        n = len(seq)
        print(f"\n{Path(folder).name}: {n} frames {shape[1]}x{shape[0]} {dtype}, "
              f"{Path(paths[0]).stat().st_size / 1e6:.0f} MB per file")
        for _ in te.iter_raw(seq, [range(s, min(s + 8, n)) for s in range(0, n, 8)]):
            pass  # warm the file cache so runs compare decode, not disk
        for name in args.processors.split(","):
            job(f"stack {name}", n, lambda: te.stack(seq, te.create_processor(name)))
        if args.video_out:
            for blend, step, master, preview in ((1, 1, "prores", False), (1, 1, "hevc_nvenc", False),
                                                 (8, 3, "prores", True), (8, 3, "hevc_nvenc", True),
                                                 (8, 3, None, True), (30, 1, "hevc_nvenc", False),
                                                 (60, 1, "hevc_nvenc", False)):
                outs = len(te.engine.window_starts(n, blend, step))
                label = f"video b{blend} s{step} -> {master or ''}{' + 1080p' if preview else ''}"
                job(label, outs, lambda: te.render_video(seq, "average", args.video_out, "mem", blend=blend,
                                                         step=step, master=master, preview_1080p=preview,
                                                         overwrite="overwrite"))
        seq.close()


if __name__ == "__main__":
    main()
