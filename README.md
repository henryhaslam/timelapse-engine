# timelapse-engine

Stacking and rolling-window blending for large timelapse sequences (JPEG / 16-bit TIFF) and video files, with
optional CuPy GPU acceleration.

This is a cleaned-up rewrite of `image_functions_2.py`; the original is kept unchanged in
[`legacy/`](legacy/) for reference.

## Install

```bash
pip install -e .            # from this folder
pip install -e .[gpu]       # + CuPy (CUDA 12)
```

FFmpeg must be on `PATH` for video output. exiftool is optional (used for fast EXIF sorting); it is found via the
`TIMELAPSE_EXIFTOOL` environment variable, `PATH`, or an `exiftool-*` folder next to this repo. Without it, EXIF
is read with Pillow.

## Quick start

```python
import timelapse_engine as te
te.enable_console_logging()
progress = te.ConsoleProgress()

# Whole sequence -> one image
te.stack_to_file(r"E:\TL\shoot", "maximum", r"E:\TL\out", ext="tiff", progress=progress)

# Rolling window -> video (ProRes master + 1080p preview)
te.render_video(r"E:\TL\shoot", "average", r"E:\TL\out", blend=8, step=3, fps=24, progress=progress)

# Video file in, still out
te.stack_to_file(r"E:\clip.mp4", "motion_overlay", r"E:\out", motion_threshold=20, progress=progress)
```

See [`examples/run_pipeline.py`](examples/run_pipeline.py) for the full batch workflow.

## Layout

| Module | Purpose |
|---|---|
| `sources.py` | `ImageSequence` (folder, EXIF-sorted, validated) and `VideoSource`; both yield float32 0–1 frames |
| `processors.py` | All stacking effects behind one `reset / update / result` interface |
| `engine.py` | `stack`, `stack_to_file`, `render_windows`, `render_video`, `render_frames`, `sweep_parameter` |
| `outputs.py` | Numbered-still sink and a single-process multi-output FFmpeg sink |
| `files.py` | Loading, sorting, validation, XMP/EXIF, saving |
| `progress.py` | Progress callback, `CancelToken`, console progress bar |
| `stabilise.py` | Point picking, LK tracking, translation stabilisation |

The engine never prints or calls `input()`. Progress goes to a `progress(done, total, message)` callback, and a
`CancelToken` stops a job between batches, which is what a GUI needs.

## Processors

`average`, `maximum`, `minimum`, `median`, `mode`, `temporal_contrast`, `temporal_std`, `temporal_gradient`,
`temporal_hue_motion`, `temporal_entropy`, `optical_flow_accum`, `motion_overlay`. `te.list_processors()` gives
descriptions.

The same processor code is used for still stacks and video windows, so a blended video frame matches the still you
would get from the same frames.

## Changes from `image_functions_2.py`

Bug fixes:

- The first batch was fed to `init()` and then `update()`, so it was counted twice (biased `average` and
  `median`, false jump in `temporal_gradient`). Processors now only accumulate in `update()`.
- `median` and `mode` cast 0–1 floats to `uint8`, which gave black images.
- Output timestamps were never applied (`get_image_timestamp` didn't exist). Outputs now get the latest capture
  time + 1 minute, written into JPEG EXIF on the first save (no second JPEG re-encode).
- exiftool returns `E:/a/b.jpg` while glob returns `E:\a\b.jpg`, so dates never matched and sorting quietly fell
  back to a plain string sort. Paths are now normalised before matching.
- The EXIF-sort fallback called `natural_sort_key`, which only existed inside another function (`NameError`).
- 8-bit TIFFs were divided by 65535 (near black). Scaling now follows the file's real bit depth.
- `temporal_entropy` counted in `float16` (wrong above about 2048 frames), and its noise filter carried over between
  windows. It now uses `uint16` counts, switching to `uint32` past 65 535 frames, plus a small CUDA kernel instead
  of `cp.add.at` with three full-size index grids. That uses roughly half the GPU memory.
- `temporal_hue_motion` and `optical_flow_accum` returned `uint8` arrays into a 0–1 pipeline (white output). Optical
  flow also passed float images to Farneback, and averaged angles without handling wrap-around. Both are fixed;
  the direction is now the angle of the summed flow vector.
- Video stills and image stills used different maths for `temporal_contrast`, `temporal_gradient`,
  `temporal_entropy` and `motion_overlay`. There is now one implementation of each. The old video
  `temporal_contrast` (standard deviation) is still available as `temporal_std`.
- Stabilisation used frame 0 as the origin even when tracking started later, so earlier frames were all zero.
  It now tracks forwards and backwards from the reference frame. Lost points keep their last position. Output
  JPEGs keep their EXIF (so they still sort by capture time), and 16-bit TIFFs stay 16-bit.
- `os.fsync(0)` (which synced stdin) and the `sleep` calls are gone.

Clean-up:

- `percentile` was removed.
- The global `load_image` monkey-patch for video was replaced by `FrameSource` classes.
- Six copies of the progress bar code are now one callback. `input()` prompts are replaced by an
  `overwrite="increment" | "overwrite" | "error"` option.
- `process_frames`, `process_video_to_video` and `render_timelapse_pipeline` are merged into `render_windows`
  with pluggable outputs. Overlapping windows reuse decoded frames instead of re-reading them (about 2.7× fewer
  decodes at blend 8 / step 3).
- The next batch now loads in the background while the current one is processed.
- FFmpeg writes every output (master + previews) from one process with a `split` filter, so each frame goes
  through the pipe once. Odd dimensions are cropped to even. The `hevc_lossless` master now uses real
  `lossless=1`, since `-crf 0` is not lossless in x265.
- Motion overlay closing uses OpenCV morphology (much faster than `scipy.ndimage`).

## Tests

```bash
pytest            # or: python tests/test_engine.py
```
