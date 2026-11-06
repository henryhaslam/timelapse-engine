"""Where does the time go? Benchmarks each stage on real frames.

    python benchmarks/bench.py "E:\\TL temp\\G10_handle\\export" --frames 96

Stages: disk read, JPEG decode (Pillow vs OpenCV, 1 vs N threads), float
conversion, processor maths (CPU vs GPU), and end-to-end ``te.stack``.
Note: the second run over the same frames is served from the Windows file
cache, so "disk" numbers are only meaningful on a cold first run.
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import timelapse_engine as te  # noqa: E402
from timelapse_engine import files  # noqa: E402


def timed(label, fn, n_frames, mb=None):
    t = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t
    extra = f"  {mb / dt:7.0f} MB/s" if mb else ""
    print(f"{label:<44} {n_frames / dt:7.1f} frames/s{extra}")
    return out


def decode_pillow(data):
    import io
    with Image.open(io.BytesIO(data)) as im:
        return np.asarray(im)


def decode_cv2(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--processors", default="average,maximum,temporal_gradient,temporal_entropy,motion_overlay")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument("--stages-only", action="store_true", help="skip end-to-end runs")
    ap.add_argument("--video-out", help="folder for render_video benchmarks (files are overwritten)")
    args = ap.parse_args()

    paths = sorted(files.list_images(args.folder), key=files.natural_sort_key)[: args.frames]
    n, T = len(paths), args.threads
    mb = sum(Path(p).stat().st_size for p in paths) / 1e6
    print(f"{n} frames, {mb / n:.1f} MB each, {T} threads, CuPy: {te.HAS_CUPY}\n")

    with ThreadPoolExecutor(T) as pool:
        blobs = timed("disk read (bytes only)", lambda: list(pool.map(lambda p: Path(p).read_bytes(), paths)), n, mb)
        sample = blobs[: min(16, n)]
        timed("decode Pillow, 1 thread", lambda: [decode_pillow(b) for b in sample], len(sample))
        timed("decode OpenCV, 1 thread", lambda: [decode_cv2(b) for b in sample], len(sample))
        timed(f"decode Pillow, {T} threads", lambda: list(pool.map(decode_pillow, blobs)), n)
        imgs = timed(f"decode OpenCV, {T} threads", lambda: list(pool.map(decode_cv2, blobs)), n)
        timed(f"to float32 0-1, {T} threads", lambda: list(pool.map(files.to_float, imgs)), n)
        timed("to float32 0-1, 1 thread", lambda: [files.to_float(i) for i in imgs[:16]], min(16, n))
        timed(f"load_image (current), {T} threads", lambda: list(pool.map(files.load_image, paths)), n)

    batch_u8 = np.stack(imgs[: args.batch])
    batch_f = batch_u8.astype(np.float32) / 255
    print()
    for name in ("average", "maximum", "temporal_contrast", "temporal_gradient", "motion_overlay"):
        proc = te.create_processor(name)
        timed(f"{name} maths, CPU", lambda: te.process_window(proc, batch_f), len(batch_f))

    if te.HAS_CUPY:
        import cupy as cp
        cp.cuda.Device().synchronize()
        t = time.perf_counter()
        for _ in range(3):
            g = cp.asarray(batch_u8)
            cp.cuda.Device().synchronize()
        dt = (time.perf_counter() - t) / 3
        print(f"{'upload uint8 batch to GPU':<44} {len(batch_u8) / dt:7.1f} frames/s  {batch_u8.nbytes / 1e6 / dt:7.0f} MB/s")
        t = time.perf_counter()
        for _ in range(3):
            acc = (g.astype(cp.float32) * (1 / 255)).sum(axis=0)
            cp.cuda.Device().synchronize()
        dt = (time.perf_counter() - t) / 3
        print(f"{'average maths, GPU (from uint8)':<44} {len(batch_u8) / dt:7.1f} frames/s")
        del g, acc

    print()
    seq = te.ImageSequence(paths=paths, sort="name", validate=False, workers=T)
    devices = ["gpu", "cpu"] if te.HAS_CUPY and not args.gpu_only else ["cpu"] if not te.HAS_CUPY else ["gpu"]
    for name in args.processors.split(","):
        for device in devices:
            proc = te.create_processor(name)
            if device == "gpu" and not proc.gpu:
                continue
            timed(f"END TO END stack {name} [{device}]",
                  lambda: te.stack(seq, proc, args.batch, device=device), n)

    if args.video_out:
        print()
        for blend, step in ((1, 1), (8, 3)):
            timed(f"END TO END render_video average b{blend} s{step} [prores]",
                  lambda: te.render_video(seq, "average", args.video_out, f"bench_b{blend}", blend=blend,
                                          step=step, master="prores", preview_1080p=False, overwrite="overwrite"),
                  len(te.engine.window_starts(n, blend, step)))
        timed("END TO END render_video average b8 s3 [1080p only]",
              lambda: te.render_video(seq, "average", args.video_out, "bench_1080", blend=8, step=3, master=None,
                                      preview_1080p=True, overwrite="overwrite"),
              len(te.engine.window_starts(n, 8, 3)))
    seq.close()


if __name__ == "__main__":
    main()
