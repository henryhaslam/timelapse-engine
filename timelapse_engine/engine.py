"""The processing engine: whole-sequence stacks and rolling-window renders.

Nothing here prints or asks questions. Progress goes to the ``progress``
callback, messages to the ``timelapse_engine`` logger, and a ``CancelToken``
can stop any job between batches/frames.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from . import files
from .outputs import FFmpegVideoSink, ImageSequenceSink, Sink, h264_preview, hevc_lossless_master, prores_master
from .processors import Processor, create_processor
from .progress import CancelToken, ProgressCallback, check_cancel, logger, report
from .sources import FrameSource, ImageSequence, VideoSource

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


# ---------------------------------------------------------------------------
# Batch iteration
# ---------------------------------------------------------------------------

def iter_batches(source: FrameSource, batch_size: int) -> Iterator[np.ndarray]:
    """Yield (N, H, W, C) float32 batches, reading the next batch in the background
    while the caller processes the current one."""
    n = len(source)
    ranges = [range(s, min(s + batch_size, n)) for s in range(0, n, batch_size)]
    if not ranges:
        return
    with ThreadPoolExecutor(max_workers=1) as prefetch:
        pending = prefetch.submit(source.read_many, ranges[0])
        for k in range(len(ranges)):
            frames = pending.result()
            if k + 1 < len(ranges):
                pending = prefetch.submit(source.read_many, ranges[k + 1])
            yield np.stack(frames)


def stack(source: FrameSource, processor: Processor, batch_size: int = 16,
          progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None) -> np.ndarray:
    """Run ``processor`` over every frame of ``source`` and return the float 0-1 result."""
    processor.reset()
    n = len(source)
    passes = 2 if processor.two_pass else 1
    for pass_index in range(passes):
        if pass_index == 1:
            processor.begin_second_pass()
        feed = processor.update if pass_index == 0 else processor.update_second_pass
        label = f"{processor.name}" + (f" (pass {pass_index + 1}/2)" if passes == 2 else "")
        done = 0
        report(progress, 0, n, label)
        for batch in iter_batches(source, batch_size):
            check_cancel(cancel)
            feed(batch)
            done += batch.shape[0]
            report(progress, done, n, label)
    return processor.result()


def stack_to_file(source, process: ProcessorSpec, export_dir, export_name: Optional[str] = None,
                  ext: str = "tiff", batch_size: int = 16, overwrite: str = "increment",
                  timestamp_offset: Optional[timedelta] = timedelta(minutes=1),
                  progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                  **params) -> Path:
    """Stack a whole sequence into one image and save it as ``<export_name>_<process>.<ext>``.

    * ``ext``: "tiff" (16-bit, keeps the last frame's XMP / Lightroom settings) or "jpg" (8-bit).
    * The output's capture time is the latest frame's capture time + ``timestamp_offset``
      (set it to None to skip timestamping).
    * ``params`` go to the processor, e.g. ``hist_bins=32`` or ``motion_threshold=20``.
    """
    src = open_source(source)
    processor = _make_processor(process, params)
    export_name = export_name or src.name
    out_path = files.resolve_output_path(
        Path(export_dir) / f"{export_name}_{processor.name}.{ext.lstrip('.')}", overwrite)

    logger.info("Stacking %d frames with %s -> %s", len(src), processor.name, out_path.name)
    result = stack(src, processor, batch_size, progress, cancel)

    timestamp = src.latest_capture_time()
    if timestamp is not None and timestamp_offset is not None:
        timestamp += timestamp_offset
    saved = files.save_image(out_path, result, xmp=src.xmp(len(src) - 1), timestamp=timestamp)
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


def iter_windows(source: FrameSource, blend: int, step: int,
                 include_partial: bool = True) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield (start_index, frames) for each window. Frames shared by overlapping
    windows are decoded once and reused."""
    n = len(source)
    cache: Dict[int, np.ndarray] = {}
    for start in window_starts(n, blend, step, include_partial):
        indices = range(start, min(start + blend, n))
        for old in [i for i in cache if i < start]:
            del cache[old]
        missing = [i for i in indices if i not in cache]
        if missing:
            cache.update(zip(missing, source.read_many(missing)))
        yield start, np.stack([cache[i] for i in indices])


def process_window(processor: Processor, frames: np.ndarray) -> np.ndarray:
    processor.reset()
    processor.update(frames)
    if processor.two_pass:
        processor.begin_second_pass()
        processor.update_second_pass(frames)
    return processor.result()


def render_windows(source, process: ProcessorSpec, sinks: Sequence[Sink], blend: int = 1, step: int = 1,
                   include_partial: bool = True, progress: Optional[ProgressCallback] = None,
                   cancel: Optional[CancelToken] = None, **params) -> int:
    """Blend each rolling window of ``blend`` frames (advancing by ``step``) and write
    the result to every sink. Returns the number of output frames."""
    src = open_source(source)
    processor = _make_processor(process, params)
    total = len(window_starts(len(src), blend, step, include_partial))
    logger.info("Rendering %d frames from %d sources with %s (blend %d, step %d).",
                total, len(src), processor.name, blend, step)

    written = 0
    try:
        report(progress, 0, total, processor.name)
        for _, frames in iter_windows(src, blend, step, include_partial):
            check_cancel(cancel)
            result = frames[0] if blend == 1 else process_window(processor, frames)
            for sink in sinks:
                sink.write(result)
            written += 1
            report(progress, written, total, processor.name)
    finally:
        for sink in sinks:
            sink.close()
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
                 **params) -> list:
    """Render a rolling-window blend straight to video.

    master: "prores" (ProRes 422 HQ .mov), "hevc_lossless" (.mov) or None.
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
    elif master is not None:
        raise ValueError("master must be 'prores', 'hevc_lossless' or None")
    if preview_1080p:
        outputs.append(h264_preview(export_dir / f"{export_name}_1080p.mp4", scale="-2:1080"))
    if preview_4k:
        outputs.append(h264_preview(export_dir / f"{export_name}_4K.mp4", scale="3840:-2"))
    if not outputs:
        raise ValueError("No video outputs selected.")

    sink = FFmpegVideoSink(outputs, fps=fps, ffmpeg_path=ffmpeg_path, overwrite=overwrite)
    render_windows(src, process, [sink], blend, step, include_partial, progress, cancel, **params)
    return sink.outputs


def render_frames(source, process: ProcessorSpec, export_dir, export_name: Optional[str] = None,
                  blend: int = 1, step: int = 1, ext: str = "tiff", overwrite: str = "increment",
                  include_partial: bool = True, progress: Optional[ProgressCallback] = None,
                  cancel: Optional[CancelToken] = None, **params) -> list:
    """Render a rolling-window blend to numbered still images. Returns the files written."""
    src = open_source(source)
    sink = ImageSequenceSink(export_dir, export_name or src.name, ext, overwrite)
    render_windows(src, process, [sink], blend, step, include_partial, progress, cancel, **params)
    return sink.outputs


def sweep_parameter(source, process: ProcessorSpec, param: str, values: Sequence, export_dir,
                    export_name: Optional[str] = None, ext: str = "jpg", batch_size: int = 16,
                    progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None,
                    **params) -> list:
    """Run ``stack_to_file`` once per value of ``param`` (e.g. motion_threshold) to compare results."""
    src = open_source(source)
    name = export_name or src.name
    written = []
    for value in values:
        check_cancel(cancel)
        logger.info("%s = %s", param, value)
        written.append(stack_to_file(src, process, export_dir, f"{name}_{param}_{value}", ext, batch_size,
                                     progress=progress, cancel=cancel, **{**params, param: value}))
    return written
