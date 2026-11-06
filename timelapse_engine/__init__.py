"""timelapse_engine - stacking and rolling-window blending for timelapse sequences and video."""

from .engine import (
    iter_batches,
    iter_windows,
    open_source,
    process_window,
    render_frames,
    render_video,
    render_windows,
    stack,
    stack_to_file,
    sweep_parameter,
)
from .files import SequenceMismatchError, load_image, save_image
from .outputs import FFmpegVideoSink, ImageSequenceSink, VideoOutput, h264_preview, hevc_lossless_master, prores_master
from .processors import HAS_CUPY, PROCESSORS, Processor, create_processor, list_processors
from .progress import CancelToken, Cancelled, ConsoleProgress, enable_console_logging
from .sources import FrameSource, ImageSequence, VideoSource

__version__ = "0.1.0"

__all__ = [name for name in dir() if not name.startswith("_")]
