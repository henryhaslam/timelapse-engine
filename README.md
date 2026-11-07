# timelapse-engine

Stacking and rolling-window blending for large timelapse sequences (JPEG / 16-bit TIFF) and video files, with
optional CuPy GPU acceleration.

This is a cleaned-up rewrite of `image_functions_2.py`; the original is kept unchanged in
[`legacy/`](legacy/) for reference.

## Install

```bash
pip install -e .            # from this folder
pip install -e .[gpu,gui]   # + CuPy (CUDA 12) and the desktop app (PySide6)
```

FFmpeg must be on `PATH` for video output. exiftool is optional (used for fast EXIF sorting); it is found via the
`TIMELAPSE_EXIFTOOL` environment variable, `PATH`, or an `exiftool-*` folder next to this repo. Without it, EXIF
is read with Pillow.

## Desktop app

Double-click **`Timelapse Engine.bat`**, or run `python -m timelapse_engine.gui` (`timelapse-gui` once installed).
The app has two tabs: **Sort raw files** (below) and **Stack and render**.

- **Source:** pick an image folder or a video. Frames are sorted by capture time and checked once, and you can
  limit the job to a frame range. The scrubber steps through frames, showing each file name and capture time.
- **Effect:** choose what to create (single image, video or image sequence), the effect and its settings. The
  settings form is built from each effect's parameters, with tooltips.
- **Preview** (Ctrl+P) renders the current settings in about a second at reduced resolution, using frames spread
  across the sequence for a still, or the rolling window at the scrubber for video. With *Update automatically*
  on, it re-renders as you change settings or scrub. Wheel to zoom, drag to pan, double-click to fit.
- **Queue:** add as many jobs as you like and start them. Each shows its speed and time left, and Cancel stops
  the running job. Double-click a job to load its settings back into the form; double-click its output to open
  it in Explorer. Queues save and load as JSON (`timelapse_engine.jobs`), and you can also run those files from a
  script with `run_job`.
- **Other details:** the log tab shows engine messages, and the status bar shows free GPU memory. Window layout
  and your last settings are remembered in `%APPDATA%\timelapse-engine\gui.ini`.

## Sorting raw files into sequences

A card dump is one long list of shots. The **Sort raw files** tab (or `timelapse-sort`) splits it into one folder
per timelapse sequence before you import into Lightroom.

**How it groups shots.** It reads every shot's capture time (with exiftool, to the millisecond where the camera
records it) and looks at the spacing between shots:

- A run of evenly spaced shots is one sequence. The spacing is compared with the recent spacing, so a slowly
  ramped interval (day to night) stays together.
- A gap that doesn't fit the spacing ends the sequence. The next one can have the same spacing or a new one.
- Where the camera numbers the shots of each burst or interval run (Panasonic `SequenceNumber`), a restart of
  that counter also ends the sequence.
- Each camera body is grouped separately. Shots in no sequence are single shots and stay where they are unless
  you tick them.
- Files that share a name (`P1000123.RW2`, `.JPG`, `.xmp`, `.RW2.xmp`) are one shot and always move together.

**In the app:**
1. Pick the folder and the table fills in with one row per sequence. Each row shows the start time, length,
   frame count, spacing, camera and file range.
2. Select a row to see its first and last frame. The chart underneath plots the spacing of every shot (log
   scale): flat coloured lines are sequences, grey dots are the gaps between them.
3. If the grouping looks wrong, adjust **Spacing tolerance** (default 25 %) or **Fewest frames** (default 5).
   The table regroups instantly without re-reading the files.
4. Rename folders in the table (double-click) or untick any you want left where they are.
5. **Move into folders** (instant on the same drive) or **Copy into folders**.

**Safety:** nothing is ever overwritten. If a name is taken (e.g. two card folders both have `P1000001.RW2`),
every file of that shot gets the same `_2` suffix, so the raw and its sidecar stay paired. Each sort writes a log
(`_timelapse-sort-<date>.json`) next to the new folders, even if it stops part way, and **Undo last sort** uses it
to put everything back.

On the G9 II sample card (638 RW2 files, 25 GB), it found all six sequences in about 5 s: a 10 fps burst, two
0.48 s runs 17 s apart, and three 1 s runs, including a 7-frame false start.

```bash
timelapse-sort "E:\card\DCIM" --subfolders           # show what it would do
timelapse-sort "E:\card\DCIM" --subfolders --move    # do it (or --copy, --dest D:\TL)
timelapse-sort "E:\card\DCIM" --undo                 # put everything back
```

Folder names come from a pattern, by default `{date}_{time}_{interval}_{frames}f` (e.g.
`2026.06.07_131442_2s_270f`). The fields are `{date}`, `{time}`, `{interval}`, `{frames}`, `{camera}` and
`{number}`.

## Raw stacks to DNG (prototype)

This is an alternative to exporting every frame from Lightroom as a TIFF. The engine stacks the raw sensor
data directly and saves one DNG, which you develop in Lightroom like any raw from the camera.

```bash
pip install -e .[raw]                                              # rawpy
python -m timelapse_engine.rawstack "E:\TL\raw\2026.11.07_162035_0.48s_177f" --process average
```

**Why this route:**
- **True long exposure.** Raw data is linear, so the average is what a real ND filter would record. Averaging
  gamma-encoded TIFFs makes moving highlights too dark: on the terifa sea it was 5/255 darker overall and
  14/255 darker in the glitter.
- **No TIFF export.** On the terifa card, 177 frames stacked in 14 to 24 s into a 48 MB DNG, against about
  26 GB of 16-bit TIFFs.
- **Full latitude in the final edit.** Lightroom still does the demosaicing, colour profile, white balance and
  highlight recovery.

**How it works:**
- `RawSequence` reads the sensor mosaic, before demosaicing, as (H, W, 1) `uint16` frames at about 25 frames/s
  on 8 threads. The engine's normal GPU `stack()` processes them.
- `write_dng` writes the camera's CFA layout, black and white levels, colour matrix and as-shot white balance.
  The camera EXIF, lens name and caption come from the last frame.
- A 1-frame DNG reads back bit-identical to the RW2 and renders identically in LibRaw.

**Limits:**
- Only per-pixel stacks work on sensor data: `average`, `maximum`, `minimum`, `median`, `clipped_average` and
  `comet_trails`. The entropy, colour-map and motion effects need developed RGB frames.
- Clipped highlights are made neutral white in each frame before stacking. Otherwise clipped green turns
  averaged glitter magenta. `--keep-clipped` turns this off.
- **Still to check in Lightroom:**
  - Whether it offers the camera's profiles for the DNG.
  - Its brightness against the RW2: there is no BaselineExposure tag yet.
  - Micro Four Thirds lens correction: the built-in distortion data in the RW2 isn't carried over yet.

## Quick start

```python
import timelapse_engine as te
te.enable_console_logging()
progress = te.ConsoleProgress()

# Whole sequence -> one image
te.stack_to_file(r"E:\TL\shoot", "maximum", r"E:\TL\out", ext="tiff", progress=progress)

# Rolling window -> video (ProRes master + 1080p preview)
te.render_video(r"E:\TL\shoot", "average", r"E:\TL\out", blend=8, step=3, fps=24, progress=progress)

# Same, with the much faster GPU HEVC master
te.render_video(r"E:\TL\shoot", "average", r"E:\TL\out", blend=8, step=3, master="hevc_nvenc", progress=progress)

# Video file in, still out
te.stack_to_file(r"E:\clip.mp4", "motion_overlay", r"E:\out", motion_threshold=20, progress=progress)
```

See [`examples/run_pipeline.py`](examples/run_pipeline.py) for the full batch workflow.

## Layout

| Module | Purpose |
|---|---|
| `sources.py` | `ImageSequence` (folder, EXIF-sorted, validated) and `VideoSource`; frames come out in their native 8/16-bit form |
| `processors.py` | All stacking effects behind one `reset / update / result` interface; the same code runs on NumPy or CuPy |
| `engine.py` | `stack`, `stack_to_file`, `render_windows`, `render_video`, `render_frames`, `sweep_parameter`, memory planning |
| `outputs.py` | Numbered-still sink and FFmpeg video sink (one encoder process per output) |
| `colour.py` | BT.709 RGB → Y'CbCr packing and area downscaling for video, on the GPU |
| `backend.py` | CPU/GPU helpers |
| `files.py` | Loading, sorting, validation, XMP/EXIF, saving |
| `progress.py` | Progress callback, `CancelToken`, console progress bar |
| `stabilise.py` | Point picking, LK tracking, translation stabilisation |
| `jobs.py` | `Job` (one saveable render description), queue files, previews, parameter metadata |
| `effects.py` | Entropy family, temporal colour maps, time-slice, clean-up and trail effects |
| `adjust.py` | Deflicker (brightness measurement, smoothing, linear-light gain) |
| `interpolate.py` | Motion-compensated frame interpolation |
| `sorter.py` | Raw sorter: groups a card dump into sequences by capture-time spacing; move/copy/undo; CLI |
| `rawstack.py` | Prototype: stack raw sensor data (`RawSequence`) and write a DNG (`write_dng`) for Lightroom |
| `gui/` | PySide6 desktop app (`sort_panel.py` is the Sort raw files tab) |

The engine never prints or calls `input()`. Progress goes to a `progress(done, total, message)` callback, and a
`CancelToken` stops a job between batches, which is what a GUI needs.

## Effects

The same effect code is used for still stacks and video windows, so a blended video frame matches the still you
would get from the same frames. Every effect runs on the GPU except `median`, `mode` and `optical_flow_accum`.
`te.list_processors()` gives descriptions. In the app, the settings for each effect appear with tooltips.

| Group | Effect | What it shows |
|---|---|---|
| Stacks | `average`, `maximum`, `minimum`, `median`, `mode` | classic long exposure, light trails, clean plate |
| Temporal colour | `temporal_entropy` | how varied each colour channel was: moving clouds and water go vivid, static scenery black |
| | `change_entropy` | entropy of frame-to-frame *changes*: turbulence (boiling cloud, choppy water) vs smooth drift |
| | `colour_entropy` | entropy of colour and brightness separately (R = red/cyan shifts, G = brightness, B = blue/yellow shifts) |
| | `temporal_shape` | skewness (mostly dark with bright moments, or the opposite) or kurtosis (rare spikes) per channel |
| | `temporal_contrast`, `temporal_std`, `temporal_gradient` | range, variation and accumulated change |
| Motion and time | `time_of_peak` | hue = *when* each pixel was brightest / darkest / changed most |
| | `rhythm_map` | hue = dominant repeating period (waves, traffic, blinking lights, drifting cloud) |
| | `eigen_colour` | the 3 strongest patterns of change (PCA) as red, green and blue |
| | `memory_map` | how long pixels stay similar, at 3 time scales as RGB |
| | `temporal_hue_motion`, `optical_flow_accum`, `motion_overlay` | motion coloured by time or direction; movers on a clean plate |
| Time slice | `time_slice` | each part of the image from a different moment (see below) |
| Clean-up and trails | `temporal_denoise` | motion-aware median for videos (blend 3-9) |
| | `comet_trails` | maximum where older light fades: tapering star/light trails |
| | `clipped_average` | average that ignores outliers (planes, people, birds) |

All the entropy effects accept these settings:

- **`renyi_order`** (alpha): 1 is standard entropy. Below 1 lifts rare events; above 1 keeps only the dominant changes.
- **`neighbourhood`**: also counts the pixels this many px around each pixel, for a smoother, painterly look.
- **`hist_bins`** and **`noise_threshold`**.

Speed at 20 MP / 16-bit on the RTX 3080 Ti: 28-73 fps for every new effect (170 fps for a banded time-slice,
which reads only the frames it uses), within about 8 GB of VRAM.

### Time slice

`time_slice` replaces `slice_functions.py`, with these differences:

- No width or divisibility limits, and every frame can contribute.
- Frames are in capture-time order.
- It works for stills and as a rolling-window video.

| Setting | |
|---|---|
| `pattern` | `left_to_right`, `top_to_bottom`, `diagonal` (at `angle`), `radial` (centre outwards), `circular` (clock sweep from `angle`), `brightness` (bright parts of the first frame come from late frames) |
| `slices` | `0` = smooth: every column/ring from its own moment, blended between neighbouring frames. `N` = N bands, one frame each; only those N frames are read |
| `feather` | 0-1: band edges blend through the frames in between |
| `reverse`, `mirror` | run time backwards; there and back (e.g. night in the middle) |
| `curve` | below 1 gives more of the image to the end of the sequence, above 1 to the start |

```python
te.stack_to_file(r"E:\TL\shoot", "time_slice", r"E:\TL", pattern="radial", slices=12, feather=0.3, mirror=True)
```

### Deflicker and smooth motion

These options apply to any job (`deflicker=True` on `stack_to_file`, `render_video` or `render_frames`, or the
checkbox in the app).

- **Deflicker:** first measures each frame's brightness from fast reduced-size decodes. It smooths that curve over
  `deflicker_window` frames and corrects each frame in linear light, like a small exposure change. Your G10
  sequence measured 2.5 % frame-to-frame flicker.
- **Smooth motion** (`interpolate=2|3|4`, videos and image sequences): adds motion-compensated in-between frames.
  Optical flow is computed at 960 px and warped at full resolution on the GPU. It suits clouds and water; fast
  small objects can warp.

## Output metadata

Every output records what it is and where it came from, so Lightroom (or any EXIF viewer) shows it:

| Field | Example |
|---|---|
| Title | `Maximum stack - 270 frames` |
| Caption / description | `Maximum stack of 270 frames \| source frames 21-290 of 312 from 2026.06.07_copenhill \| 1 s exposures every 2 s \| 2026-06-07 13:14:42 to 13:23:40 (8 min 58 s) \| total exposure 4 min 30 s \| Panasonic DC-GX9, LUMIX G 20/F1.7 II, 20 mm f/4.5 ISO 400 \| settings: motion_threshold=30` |
| Camera and lens | make, model, lens, focal length, aperture, ISO and exposure time from the source frames |
| Colour profile | the source's ICC profile (e.g. ProPhoto RGB), so colours display as in the originals |
| Capture time | stills: last frame + 1 minute (sorts after the sequence); sequence frames: last frame of each window |
| EXIF UserComment | the same facts as JSON (`frames`, `interval_s`, `exposure_s`, `start`/`end`, `params`, …) |

Details:

- **With exiftool** (found automatically), all EXIF, IPTC and XMP is copied from the last source frame, including
  Lightroom's develop settings. Those are marked `AlreadyApplied`, so Lightroom won't apply them twice. Document IDs
  are not copied, so each output counts as a new image.
- **Without exiftool**, Pillow and tifffile still write the caption, camera, profile and dates.
- **Image sequences:** each frame describes its own window.
- **Videos:** these get a title, a description and a creation time.

## Performance

How a frame moves through the engine (GPU path):

1. **Decode:** 16 threads decode JPEGs with OpenCV straight into reusable pinned buffers. Uncompressed TIFFs are
   read straight into the buffer with tifffile. Three buffers are kept: two being decoded while one is processed.
2. **Upload:** frames stay 8/16-bit until they are on the GPU, so far less data crosses PCIe than float32 would.
3. **Process:** frames are converted to float in chunks sized to free VRAM. All processors except median, mode and
   optical flow run on the GPU.
4. **Rolling windows:** each frame is decoded and uploaded once. `average` and `temporal_std` use exact integer
   running sums, so cost per output frame doesn't grow with `blend`.
5. **Video output:** BT.709 Y'CbCr conversion and preview downscaling happen on the GPU. Each output gets its own
   FFmpeg process and writer thread, and data goes through the pipe in 4 MB pieces (single 100 MB writes are ~5×
   slower on Windows).

Batch size, GPU chunk size and where the window cache lives (VRAM or RAM) are picked per job from frame size and
free memory (`engine.MemoryPlan`). Median and mode, which must hold every frame in RAM, fail at the start with a
clear message if they can't fit.

Measured on a Ryzen 7 7700X, RTX 3080 Ti 12 GB and 64 GB RAM, with 96 frames at 5184×3888 and files already in
the OS cache. Numbers are frames per second.

**Stills**

| | JPEG, v0.1 (CPU) | JPEG, now | 16-bit TIFF, now |
|---|---|---|---|
| average | 7.3 | 34–38 | 72 |
| maximum | ~10 | 33 | 82 |
| temporal_entropy | – | 30 | 44 |
| motion_overlay (2 passes) | 1.1 | 17 | 37 |

**Video** (output frames per second)

| | JPEG | 16-bit TIFF |
|---|---|---|
| blend 1 → ProRes HQ | 2.2 (encoder limit 2.7 on this footage) | 7.2 (encoder limit 7.6) |
| blend 1 → `hevc_nvenc` | 17 | 19 |
| blend 8 / step 3 → 1080p preview | 10 | 15 |
| blend 30 / step 1 average → `hevc_nvenc` | 9.5 | 21 (was 2.8) |
| blend 60 / step 1 average → `hevc_nvenc` | 6.6 | 13 (was 1.1) |

Peak use on the TIFF runs is about 3.5–4.5 GB of RAM (it was 12 GB before memory planning) and up to 8.7 GB of
VRAM.

What limits speed:

- **Disk, on a first (cold) read.** A SATA SSD reads ~14 of these 30 MB JPEGs per second, so put big jobs on NVMe.
- **JPEG decode:** ~40 fps on 16 threads.
- **ProRes encoding,** which runs on the CPU and slows down a lot on detailed or noisy footage (2.7–7.6 fps at
  20 MP here). `master="hevc_nvenc"` (10-bit HEVC on the GPU) is 3–7× faster. A ProRes master at 4K resolution
  would also help.

`benchmarks/bench.py` and `benchmarks/memory_bench.py` reproduce these numbers.

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
- Odd frame dimensions are cropped to even for video. The `hevc_lossless` master now uses real `lossless=1`,
  since `-crf 0` is not lossless in x265.
- Video is now converted to Y'CbCr with the **BT.709** matrix and tagged as such. FFmpeg's default for untagged
  RGB input (what the old pipeline used) is BT.601, so players and Resolve, which assume BT.709 for HD and
  larger video, showed slightly shifted colours.

## Tests

```bash
pytest            # or run each file: tests/test_engine.py, test_effects.py, test_sorter.py, test_gui.py
```
