"""timelapse_engine - stacking and rolling-window blending for timelapse sequences and video."""

from .backend import HAS_CUPY
from .engine import (
    iter_batches,
    iter_raw,
    iter_windows,
    open_source,
    process_window,
    render_frames,
    render_video,
    render_windows,
    stack,
    stack_to_file,
    sweep_parameter,
    use_gpu,
)
from .files import SequenceMismatchError, load_image, load_raw, save_image
from .outputs import FFmpegVideoSink, ImageSequenceSink, VideoOutput, h264_preview, hevc_lossless_master, prores_master
from .processors import PROCESSORS, Processor, create_processor, list_processors
from .progress import CancelToken, Cancelled, ConsoleProgress, enable_console_logging
from .jobs import Job, load_queue, run_job, save_queue
from .sources import FrameSource, ImageSequence, SubsetSource, VideoSource

__version__ = "0.3.0"

__all__ = [name for name in dir() if not name.startswith("_")]
