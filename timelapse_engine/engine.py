"""The processing engine: whole-sequence stacks and rolling-window renders.

Data flow (GPU path)::

    disk --decode threads--> pinned uint8/uint16 buffer A  ┐  double-buffered:
                             pinned uint8/uint16 buffer B  ┘  load B while A is processed
         --upload (raw ints, 4x smaller than float)--> GPU --to float 0-1 in chunks--> processor
         --result to uint16 on GPU--> host --writer thread--> FFmpeg / image files

Nothing here prints or asks questions. Progress goes to the ``progress``
callback, messages to the ``timelapse_engine`` logger, and a ``CancelToken``
can stop any job between batches/frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

from . import backend, files, metadata
from .backend import HAS_CUPY, asnumpy, cp
from .outputs import (FFmpegVideoSink, ImageSequenceSink, Sink, h264_preview, hevc_lossless_master,
                      hevc_nvenc_master, prores_master)
from .processors import Processor, create_processor
from .progress import CancelToken, ProgressCallback, check_cancel, logger, report
from .sources import FrameSource, ImageSequence, SubsetSource, VideoSource

ProcessorSpec = Union[str, Processor]


def _make_processor(process: ProcessorSpec, params: dict) -> Processor:
    if isinstance(process, Processor):
        if params:
            raise ValueError("Pass parameters to the processor's constructor, not to the engine.")
        return process
    return create_processor(process, **params)


def open_source(source) -> FrameSource:
    """Accept a FrameSource, an image folder, or a video file path."""
    if isinstance(source, FrameSource):
        return source
    path = Path(source)
    if path.is_dir():
        return ImageSequence(path)
    if path.is_file():
        return VideoSource(path)
    raise FileNotFoundError(f"Not a folder or video file: {source}")


def use_gpu(processor: Processor, device: str = "auto") -> bool:
    """Decide where a processor runs. device: "auto", "gpu" or "cpu"."""
    if device not in ("auto", "gpu", "cpu"):
        raise ValueError("device must be 'auto', 'gpu' or 'cpu'")
    capable = HAS_CUPY and processor.gpu
    if device == "gpu" and not capable:
        raise RuntimeError(f"{processor.name} can't run on the GPU here "
                           f"({'CuPy not available' if not HAS_CUPY else 'CPU-only processor'}).")
    return capable and device != "cpu"


# ---------------------------------------------------------------------------
# Raw frame loading
# ---------------------------------------------------------------------------

def iter_raw(source: FrameSource, index_lists: Sequence[Sequence[int]], pinned: bool = False) -> Iterator[np.ndarray]:
    """Yield raw (native dtype) frames for each list of indices, as an (N, H, W, C) array.

    Decoding goes straight into three reusable buffers: while the caller works
    on one, the next two are being decoded, so the decode threads never sit
    idle waiting for the slowest frame of a batch. A yielded array is only
    valid until the next iteration, so copy or upload it.
    """
    if not index_lists:
        return
    shape, dtype = source.probe()
    capacity = max(len(ix) for ix in index_lists) or 1
    in_flight = 2
    buffers = [backend.empty_host((capacity, *shape), dtype, pinned) for _ in range(in_flight + 1)]

    def load(k):
        out = buffers[k % len(buffers)][: len(index_lists[k])]
        if len(out):
            source.read_into(index_lists[k], out)
        return out

    with ThreadPoolExecutor(max_workers=in_flight, thread_name_prefix="prefetch") as prefetch:
        pending = [prefetch.submit(load, k) for k in range(min(in_flight, len(index_lists)))]
        for k in range(len(index_lists)):
            batch = pending.pop(0).result()
            if k + in_flight < len(index_lists):
                # Reuses the buffer the caller finished with on the previous iteration.
                pending.append(prefetch.submit(load, k + in_flight))
            yield batch


def _feed(feed, raw, gpu: bool, scale: float, chunk: int, gains=None, positions=None) -> None:
    """Pass a raw host batch to ``feed`` as float 0-1 chunks, uploading one chunk at a time.
    ``gains`` (indexed by sequence position) applies deflicker on the way."""
    from .adjust import apply_gain

    for s in range(0, raw.shape[0], chunk):
        part = cp.asarray(raw[s:s + chunk]) if gpu else raw[s:s + chunk]
        frames = backend.to_float(part, scale)
        if gains is not None:
            frames = apply_gain(frames, gains[positions[s:s + chunk]])
        feed(frames)


class MemoryPlan:
    """How a job is split up so it fits in RAM and VRAM.

    batch: frames per read batch (three batches live in host RAM).
    chunk: frames converted to float and processed at once.
    cache_on_gpu: rolling-window frames are kept on the GPU (else in RAM).
    """

    # Fraction of free VRAM the plan may use; the rest covers CuPy pool
    # fragmentation and other apps. Going over makes Windows page VRAM to RAM,
    # which is far slower than just using smaller chunks.
    VRAM_BUDGET = 0.7

    def __init__(self, processor: Processor, frame_shape, dtype, gpu: bool, n_frames: int,
                 batch_size: Optional[int] = None, window: int = 0, step: int = 1):
        pixels = int(np.prod(frame_shape))
        f32 = pixels * 4
        raw = pixels * np.dtype(dtype).itemsize
        self.batch = batch_size or backend.auto_batch_size(frame_shape, dtype)
        self.cache_on_gpu = False
        self.sliding = bool(window > 1 and processor.sliding)

        stored = processor.stored_bytes(window or n_frames, frame_shape)
        if stored:
            free_ram = backend.host_free_bytes() - 3 * self.batch * raw
            if stored > 0.85 * free_ram:
                raise MemoryError(
                    f"{processor.name} keeps every frame in RAM: about {backend.fmt_gb(stored)} for "
                    f"{window or n_frames} frames, but only {backend.fmt_gb(free_ram)} is free. "
                    f"Process fewer frames at a time.")

        if not gpu:
            self.chunk = backend.frames_per_chunk(frame_shape)
            return

        free = backend.gpu_free_bytes()
        budget = self.VRAM_BUDGET * free
        state = processor.state_bytes(frame_shape)
        # Window cache: the window, the frames about to be subtracted, and decode look-ahead.
        cache = (window + max(step, self.batch)) * raw if window else 0
        output = 3 * f32 if window else 0  # result -> uint16 -> Y'CbCr packing for video
        per_chunk_frame = 3 * f32 + raw  # float chunk + temporaries + its upload
        # Keeping the window in VRAM (even if that forces small chunks) beats re-uploading it
        # from RAM for every output frame, especially for two-pass processors.
        if window and state + cache + output + per_chunk_frame <= budget:
            self.cache_on_gpu = True
        avail = budget - state - output - (cache if self.cache_on_gpu else 0)
        if avail < per_chunk_frame:
            raise MemoryError(
                f"Not enough GPU memory for {processor.name}: needs about "
                f"{backend.fmt_gb(state + per_chunk_frame)}, {backend.fmt_gb(free)} free. "
                f"Close other GPU apps, reduce settings (e.g. hist_bins), or use device='cpu'.")
        self.chunk = int(max(1, min(avail // per_chunk_frame, backend.CHUNK_BYTES // f32)))

    def __repr__(self):
        return (f"batch {self.batch}, chunk {self.chunk}, window cache in {'VRAM' if self.cache_on_gpu else 'RAM'}"
                f"{', running sums' if self.sliding else ''}")


def iter_batches(source: FrameSource, batch_size: int) -> Iterator[np.ndarray]:
    """Yield float32 0-1 NumPy batches (convenience for custom processing loops)."""
    n = len(source)
    scale = backend.scale_for(source.probe()[1])
    for raw in iter_raw(source, [range(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]):
        yield backend.to_float(raw, scale)


# ---------------------------------------------------------------------------
# Whole-sequence stacks
# ---------------------------------------------------------------------------

def stack(source: FrameSource, processor: Processor, batch_size: Optional[int] = None,
          progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
          device: str = "auto", gains: Optional[np.ndarray] = None) -> np.ndarray:
    """Run ``processor`` over every frame of ``source`` and return the float 0-1 result (NumPy).

    ``batch_size`` defaults to a size based on frame size (about 1 GB of raw frames).
    ``gains``: per-frame deflicker gains (see :mod:`timelapse_engine.adjust`).
    """
    gpu = use_gpu(processor, device)
    processor.reset()
    n_total = len(source)
    shape, dtype = source.probe()
    needed = processor.frames_needed(n_total, shape)
    positions = np.asarray(needed if needed is not None else range(n_total), dtype=np.int64)
    if needed is not None:
        source = SubsetSource(source, needed)  # read only the frames the effect uses
    processor.begin(n_total, shape, positions.tolist())
    n = len(source)
    scale = backend.scale_for(dtype)
    plan = MemoryPlan(processor, shape, dtype, gpu, n, batch_size)
    chunk = min(plan.batch, plan.chunk)
    ranges = [range(s, min(s + plan.batch, n)) for s in range(0, n, plan.batch)]
    passes = 2 if processor.two_pass else 1
    logger.debug("stack: %s on %s, %s", processor.name, "GPU" if gpu else "CPU", plan)

    try:
        for pass_index in range(passes):
            if pass_index == 1:
                processor.begin_second_pass()
            feed = processor.update if pass_index == 0 else processor.update_second_pass
            label = processor.name + (f" (pass {pass_index + 1}/2)" if passes == 2 else "")
            done = 0
            report(progress, 0, n, label)
            for r, raw in zip(ranges, iter_raw(source, ranges, pinned=gpu)):
                check_cancel(cancel)
                _feed(feed, raw, gpu, scale, chunk, gains, positions[r.start:r.stop])
                done += raw.shape[0]
                report(progress, done, n, label)
        return asnumpy(processor.result()).astype(np.float32, copy=False)
    finally:
        processor.reset()
        backend.free_gpu_memory()


def _deflicker_gains(src: FrameSource, window: int, progress, cancel) -> np.ndarray:
    from .adjust import deflicker_gains, measure_brightness

    levels = measure_brightness(src, progress, cancel)
    gains = deflicker_gains(levels, window)
    logger.info("Deflicker: brightness varied %.1f%% frame to frame; corrections up to %.1f%%.",
                100 * float(np.median(np.abs(np.diff(np.log(np.maximum(levels, 1e-6)))))) if len(levels) > 1 else 0,
                100 * float(np.max(np.abs(gains - 1))))
    return gains


def _with_options(params: dict, deflicker_window=None, interpolate: int = 1) -> dict:
    """Processor settings plus engine options, for the output's caption."""
    out = dict(params)
    if deflicker_window:
        out["deflicker"] = deflicker_window
    if interpolate and interpolate > 1:
        out["interpolate"] = interpolate
    return out


def stack_to_file(source, process: ProcessorSpec, export_dir, export_name: Optional[str] = None,
                  ext: str = "tiff", batch_size: Optional[int] = None, overwrite: str = "increment",
                  timestamp_offset: Optional[timedelta] = timedelta(minutes=1),
                  progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                  device: str = "auto", deflicker: bool = False, deflicker_window: int = 15,
                  **params) -> Path:
    """Stack a whole sequence into one image and save it as ``<export_name>_<process>.<ext>``.

    * ``ext``: "tiff" (16-bit, keeps the last frame's XMP / Lightroom settings) or "jpg" (8-bit).
    * The output's capture time is the latest frame's capture time + ``timestamp_offset``
      (set it to None to skip timestamping).
    * ``device``: "auto" (GPU when possible), "gpu" or "cpu".
    * ``params`` go to the processor, e.g. ``hist_bins=32`` or ``motion_threshold=20``.
    """
    src = open_source(source)
    processor = _make_processor(process, params)
    export_name = export_name or src.name
    out_path = files.resolve_output_path(
        Path(export_dir) / f"{export_name}_{processor.name}.{ext.lstrip('.')}", overwrite)

    logger.info("Stacking %d frames with %s on %s -> %s", len(src), processor.name,
                "GPU" if use_gpu(processor, device) else "CPU", out_path.name)
    gains = _deflicker_gains(src, deflicker_window, progress, cancel) if deflicker else None
    result = stack(src, processor, batch_size, progress, cancel, device, gains=gains)

    timestamp = src.latest_capture_time()
    if timestamp is not None and timestamp_offset is not None:
        timestamp += timestamp_offset
    meta = metadata.summarise(src, range(len(src)), processor.name,
                              _with_options(params, deflicker and deflicker_window), kind="stack")
    saved = files.save_image(out_path, result, timestamp=timestamp, metadata=meta)
    logger.info("Saved %s", saved)
    return saved


# ---------------------------------------------------------------------------
# Rolling windows
# ---------------------------------------------------------------------------

def window_starts(n_frames: int, blend: int, step: int, include_partial: bool = True) -> range:
    """Start index of each window. With ``include_partial`` the last few windows may hold fewer than ``blend`` frames."""
    if blend < 1 or step < 1:
        raise ValueError("blend and step must be >= 1")
    stop = n_frames if include_partial else max(n_frames - blend + 1, 0)
    return range(0, stop, step)


def _window_plan(n: int, blend: int, step: int, include_partial: bool,
                 load_batch: int) -> Tuple[List[Tuple[int, int]], List[List[int]]]:
    """Windows as (start, end), plus the frames they need grouped into load batches.
    Frames used by several windows appear once; frames no window uses are skipped."""
    windows = [(s, min(s + blend, n)) for s in window_starts(n, blend, step, include_partial)]
    needed, loaded_until = [], 0
    for start, end in windows:
        needed.extend(range(max(start, loaded_until), end))
        loaded_until = max(loaded_until, end)
    loads = [needed[i:i + load_batch] for i in range(0, len(needed), load_batch)]
    return windows, loads


def _iter_window_frames(source: FrameSource, blend: int, step: int, include_partial: bool,
                        gpu: bool, load_batch: int = 16,
                        transform=None) -> Iterator[Tuple[int, int, Dict[int, object]]]:
    """Yield (start, end, cache) where cache[i] holds raw frame i, on the GPU if ``gpu`` else in RAM.

    Frames are decoded ``load_batch`` at a time, ahead of the windows that need
    them, so every decode thread stays busy even when step is 1. Each frame is
    decoded (and uploaded) once, however many windows it appears in.
    """
    windows, loads = _window_plan(len(source), blend, step, include_partial, load_batch)
    raw_batches = iter_raw(source, loads, pinned=True)
    cache: Dict[int, object] = {}
    loaded_max, next_load, prev_start = -1, 0, 0
    try:
        for start, end in windows:
            while loaded_max < end - 1:
                raw = next(raw_batches)
                for k, i in enumerate(loads[next_load]):
                    # One upload per frame so evicted frames free their memory individually.
                    frame = cp.asarray(raw[k]) if gpu else raw[k].copy()
                    cache[i] = transform(i, frame) if transform is not None else frame
                loaded_max = loads[next_load][-1]
                next_load += 1
            # Keep the previous window's frames too: running sums subtract them.
            for old in [i for i in cache if i < prev_start]:
                del cache[old]
            yield start, end, cache
            prev_start = start
    finally:
        raw_batches.close()


def iter_windows(source: FrameSource, blend: int, step: int,
                 include_partial: bool = True) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_index, float32 0-1 frames) for each window (NumPy, CPU)."""
    scale = backend.scale_for(source.probe()[1])
    for start, end, cache in _iter_window_frames(source, blend, step, include_partial, gpu=False):
        yield start, backend.to_float(np.stack([cache[i] for i in range(start, end)]), scale)


def _run_window(processor: Processor, frames: list, scale: float, chunk: int, gpu: bool):
    """Run a processor over one window of raw frames (on the GPU or in RAM).
    With ``gpu`` the result is a GPU array; RAM-cached frames are uploaded a chunk at a time."""
    processor.reset()
    processor.begin(len(frames), tuple(frames[0].shape), range(len(frames)))
    feeds = [processor.update] + ([processor.update_second_pass] if processor.two_pass else [])
    for pass_index, feed in enumerate(feeds):
        if pass_index == 1:
            processor.begin_second_pass()
        for s in range(0, len(frames), chunk):
            part = frames[s:s + chunk]
            data = backend.xp_of(part[0]).stack(part)
            if gpu and not backend.xp_of(data) is cp:
                data = cp.asarray(data)
            feed(backend.to_float(data, scale))
    return processor.result()


def process_window(processor: Processor, frames, device: str = "auto") -> np.ndarray:
    """Run ``processor`` on one window of float 0-1 frames (N, H, W, C); returns a NumPy result."""
    gpu = use_gpu(processor, device)
    frames = np.asarray(frames) if not (HAS_CUPY and isinstance(frames, cp.ndarray)) else frames
    plan = MemoryPlan(processor, frames.shape[1:], frames.dtype, gpu, len(frames))
    result = _run_window(processor, list(frames), 1.0, plan.chunk, gpu)
    return asnumpy(result).astype(np.float32, copy=False)


def render_windows(source, process: ProcessorSpec, sinks: Sequence[Sink], blend: int = 1, step: int = 1,
                   include_partial: bool = True, progress: Optional[ProgressCallback] = None,
                   cancel: Optional[CancelToken] = None, device: str = "auto",
                   batch_size: Optional[int] = None, deflicker: bool = False, deflicker_window: int = 15,
                   interpolate: int = 1, **params) -> int:
    """Blend each rolling window of ``blend`` frames (advancing by ``step``) and write
    the result to every sink. Returns the number of blended output frames.

    deflicker: even out frame-to-frame brightness first (see :mod:`timelapse_engine.adjust`).
    interpolate: add motion-compensated in-between frames (2 = twice as many frames).
    """
    from .adjust import apply_gain

    src = open_source(source)
    processor = _make_processor(process, params)
    gpu = use_gpu(processor, device)
    shape, dtype = src.probe()
    scale = backend.scale_for(dtype)
    transform = None
    if deflicker:
        gains = _deflicker_gains(src, deflicker_window, progress, cancel)
        raw_scale = scale

        def transform(i, frame):  # corrected frames are cached as 16-bit
            return backend.to_uint16(apply_gain(backend.to_float(frame[None], raw_scale), gains[[i]])[0])

        dtype, scale = np.dtype(np.uint16), 1.0 / 65535
    plan = MemoryPlan(processor, shape, dtype, gpu, len(src), batch_size, window=blend, step=step)
    interpolator = None
    if interpolate and interpolate > 1:
        from .interpolate import FrameInterpolator

        interpolator = FrameInterpolator(interpolate)
    prev_float = None
    caption_params = _with_options(params, deflicker and deflicker_window, interpolate)
    total = len(window_starts(len(src), blend, step, include_partial))
    logger.info("Rendering %d frames from %d sources with %s on %s (blend %d, step %d; %s).",
                total, len(src), processor.name, "GPU" if gpu else "CPU", blend, step, plan)

    def on_device(frame):
        return cp.asarray(frame) if gpu and not plan.cache_on_gpu else frame

    written = 0
    summed = (0, 0)  # [lo, hi) frames currently in the running sums
    try:
        processor.reset()
        report(progress, 0, total, processor.name)
        frames_iter = _iter_window_frames(src, blend, step, include_partial, plan.cache_on_gpu, plan.batch,
                                          transform)
        for start, end, cache in frames_iter:
            check_cancel(cancel)
            if blend == 1:
                result = on_device(cache[start])  # passthrough, no maths needed
            elif plan.sliding:
                # Running sums: subtract frames that left the window, add the new ones.
                lo, hi = summed
                if start >= hi:  # windows don't overlap: start afresh
                    processor.reset()
                    lo = hi = start
                for i in range(lo, start):
                    processor.window_remove(on_device(cache[i]))
                for i in range(max(hi, start), end):
                    processor.window_add(on_device(cache[i]))
                summed = (start, end)
                result = processor.window_result(scale)
            else:
                result = _run_window(processor, [cache[i] for i in range(start, end)], scale, plan.chunk, gpu)
            meta = (metadata.summarise(src, range(start, end), processor.name, caption_params, kind="window")
                    if any(getattr(s, "wants_metadata", False) for s in sinks) else None)
            if interpolator is not None:
                current = result if result.dtype.kind == "f" else backend.to_float(result, scale)
                if prev_float is not None:
                    for between in interpolator.in_between(prev_float, current):
                        for sink in sinks:
                            sink.write(backend.to_uint16(between), meta)
                prev_float = current
            frame16 = backend.to_uint16(result)  # stays on the GPU; sinks convert/download what they need
            for sink in sinks:
                sink.write(frame16, meta)
            written += 1
            report(progress, written, total, processor.name)
    finally:
        errors = []
        for sink in sinks:
            try:
                sink.close()
            except Exception as exc:  # report every sink, raise the first
                errors.append(exc)
        processor.reset()
        backend.free_gpu_memory()
        if errors:
            raise errors[0]

    for sink in sinks:
        outs = sink.outputs
        if isinstance(sink, ImageSequenceSink) and outs:
            logger.info("Wrote %d frames to %s", len(outs), outs[0].parent)
        else:
            for path in outs:
                logger.info("Wrote %s", path)
    return written


def render_video(source, process: ProcessorSpec, export_dir, export_name: Optional[str] = None,
                 blend: int = 1, step: int = 1, fps: Optional[float] = None,
                 master: Optional[str] = "prores", preview_1080p: bool = True, preview_4k: bool = False,
                 ffmpeg_path: str = "ffmpeg", overwrite: str = "increment", include_partial: bool = True,
                 progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                 device: str = "auto", batch_size: Optional[int] = None, deflicker: bool = False,
                 deflicker_window: int = 15, interpolate: int = 1, **params) -> list:
    """Render a rolling-window blend straight to video.

    master: "prores" (ProRes 422 HQ .mov, CPU encoder, ~8 fps at 20 MP),
            "hevc_nvenc" (10-bit HEVC .mov on the NVIDIA encoder, ~3x faster),
            "hevc_lossless" (.mov, slow) or None.
    preview_1080p / preview_4k: H.264 .mp4 previews.
    fps: defaults to the source video's frame rate, or 24 for image sequences.
    Returns the list of files written.
    """
    src = open_source(source)
    export_dir = Path(export_dir)
    export_name = export_name or src.name
    fps = fps or src.fps or 24

    outputs = []
    if master == "prores":
        outputs.append(prores_master(export_dir / f"{export_name}_MASTER.mov"))
    elif master == "hevc_lossless":
        outputs.append(hevc_lossless_master(export_dir / f"{export_name}_MASTER.mov"))
    elif master == "hevc_nvenc":
        outputs.append(hevc_nvenc_master(export_dir / f"{export_name}_MASTER.mov"))
    elif master is not None:
        raise ValueError("master must be 'prores', 'hevc_nvenc', 'hevc_lossless' or None")
    if preview_1080p:
        outputs.append(h264_preview(export_dir / f"{export_name}_1080p.mp4", height=1080))
    if preview_4k:
        outputs.append(h264_preview(export_dir / f"{export_name}_4K.mp4", width=3840))
    if not outputs:
        raise ValueError("No video outputs selected.")

    name = process.name if isinstance(process, Processor) else process
    n_out = len(window_starts(len(src), blend, step, include_partial))
    meta = metadata.summarise(src, range(len(src)), name, _with_options(params, deflicker and deflicker_window,
                                                                            interpolate),
                              kind="video", blend=blend, step=step,
                              fps=fps, outputs=n_out)
    sink = FFmpegVideoSink(outputs, fps=fps, ffmpeg_path=ffmpeg_path, overwrite=overwrite, metadata=meta)
    render_windows(src, process, [sink], blend, step, include_partial, progress, cancel, device, batch_size,
                   deflicker=deflicker, deflicker_window=deflicker_window, interpolate=interpolate, **params)
    return sink.outputs


def render_frames(source, process: ProcessorSpec, export_dir, export_name: Optional[str] = None,
                  blend: int = 1, step: int = 1, ext: str = "tiff", overwrite: str = "increment",
                  include_partial: bool = True, progress: Optional[ProgressCallback] = None,
                  cancel: Optional[CancelToken] = None, device: str = "auto", deflicker: bool = False,
                  deflicker_window: int = 15, interpolate: int = 1, **params) -> list:
    """Render a rolling-window blend to numbered still images. Returns the files written."""
    src = open_source(source)
    sink = ImageSequenceSink(export_dir, export_name or src.name, ext, overwrite)
    render_windows(src, process, [sink], blend, step, include_partial, progress, cancel, device, None,
                   deflicker=deflicker, deflicker_window=deflicker_window, interpolate=interpolate, **params)
    return sink.outputs


def sweep_parameter(source, process: ProcessorSpec, param: str, values: Sequence, export_dir,
                    export_name: Optional[str] = None, ext: str = "jpg", batch_size: Optional[int] = None,
                    progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                    device: str = "auto", **params) -> list:
    """Run ``stack_to_file`` once per value of ``param`` (e.g. motion_threshold) to compare results."""
    src = open_source(source)
    name = export_name or src.name
    written = []
    for value in values:
        check_cancel(cancel)
        logger.info("%s = %s", param, value)
        written.append(stack_to_file(src, process, export_dir, f"{name}_{param}_{value}", ext, batch_size,
                                     progress=progress, cancel=cancel, device=device, **{**params, param: value}))
    return written
