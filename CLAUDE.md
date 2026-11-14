# CLAUDE.md: handover notes for timelapse-engine

Read this first in a new session. It covers what the README doesn't: how the code fits together, conventions
that are easy to break, where things stand, and what's next. The README covers features, usage and performance.

## What this is

A GPU-accelerated engine (CuPy, with a NumPy fallback) for stacking and blending timelapse sequences, plus a
PySide6 desktop app. Sequences are thousands of Lightroom-exported JPEGs or 16-bit TIFFs, often ProPhoto RGB and
hundreds of GB in total. It is a rewrite of `legacy/image_functions_2.py`, which is kept for reference only:
never edit it. The originals outside the repo (`image_functions_2.py`, `image_tests.ipynb`, `slice_functions.py`
in the parent `OneDrive\Python` folder) also stay untouched.

The owner mostly uses the desktop app. Temporal entropy is their favourite effect: they want vivid colours from
moving clouds and water, so new effects in that spirit are welcome.

## Machines and syncing

| | Desktop | Laptop |
|---|---|---|
| Hardware | Ryzen 7 7700X, RTX 3080 Ti 12 GB, 64 GB RAM | Ryzen AI 9 HX 370, RTX 4070 laptop GPU (8 GB), 32 GB RAM |
| Repo | `C:\Users\Henry\OneDrive\Python\timelapse-engine` (to move to `C:\dev` too) | `C:\dev\timelapse-engine`, with exiftool in `C:\dev\exiftool-13.59_32` |
| Real footage | `C:\Users\Henry\Videos\TL_temp\...`, e.g. `copenhagen\2026.06.07_copenhill\export_tiff` (16-bit ProPhoto TIFF, 312 frames) | `C:\Users\henry\Videos\tl_temp\terifa\raw`: 638 G9 II RW2 raws, 6 sequences (sorter test card) |

- Code moves between machines through GitHub (`henryhaslam/timelapse-engine`), never through OneDrive. Two
  machines editing the same synced git folder can corrupt it.
- Pull when you sit down at a machine. Commit and push before switching.
- **Ask before every push.** Approval covers one push only.
- On the desktop, `git` and `gh` are not on `PATH`. Use `"/c/Program Files/Git/cmd/git.exe"` and
  `"/c/Program Files/GitHub CLI/gh.exe"`.
- On the laptop, Git (installed with winget) is on `PATH` and signed in through Git Credential Manager; there is
  no `gh`. Python is 3.14 from the Python install manager. If `python` opens the Microsoft Store, use `py`, or the
  full path `%LOCALAPPDATA%\Python\bin\python.exe`.
- The laptop has 8 GB of VRAM, so check heavy effects there with `benchmarks/memory_bench.py`. `clipped_average`
  and `rhythm_map` reached about 7.7 GB at 20 MP on the desktop.

## Branches

- `main`: v0.4 plus the raw sorter. Merged on 2026-11-07: `feature/effects` (entropy family, temporal colour
  maps, time-slice, deflicker, frame interpolation) and `feature/raw-sorter` (the Sort raw files tab).
- `feature/raw-dng`: parked prototype that stacks raw sensor data into a DNG (`rawstack.py`). Lightroom
  refused the DNG ("does not contain a supported format"). The owner dropped the idea because the entropy
  effects can't use raw mosaics. Leave it as is unless asked.
- `feature/effects`, `feature/raw-sorter` and `feature/gui` are merged; new work branches from `main`.
- `feature/effect-list`: the Settings menu and "Edit this list…" for choosing which effects the Effect list
  shows.

## Setup, run, test

```bash
pip install -e ".[gpu,gui]"          # gpu = cupy-cuda12x; leave it out on a machine without an NVIDIA GPU
python -m timelapse_engine.gui        # or double-click "Timelapse Engine.bat"
python tests/test_engine.py           # 23 tests: engine, outputs, metadata, YUV packing
python tests/test_effects.py          # 21 tests: each new effect does what it claims
python tests/test_sorter.py           # 22 tests: finding sequences, move/copy/undo safety, folder layout
python tests/test_gui.py              # headless (Qt offscreen): load, preview, queue, run; Sort tab
```

- **pytest:** not installed on the desktop, so the test files run themselves. `pytest` also works where it is
  installed.
- **Test data:** tests use synthetic arrays (`ArraySource`, `CaptureSink`), so they need no footage.
- **CPU and GPU:** when CuPy is present, each test runs on both. All three suites also pass with CuPy hidden.
- **GUI screenshots:** to take one headless, also set `QT_QPA_FONTDIR=C:\Windows\Fonts` so Qt can find
  fonts.

## How it fits together

**Frame flow.** `sources.py` (`ImageSequence`, `VideoSource`, `SubsetSource`) yields raw uint8/uint16 frames.
`engine.iter_raw` decodes them on 16 threads into three reusable pinned buffers. Frames stay integer until they
are on the GPU, then become float32 0-1 in chunks sized by `engine.MemoryPlan`, which uses
`VRAM_BUDGET = 0.7` of free VRAM. Processors see `(N, H, W, C)` float32 batches.

**Processor contract** (`processors.Processor`):
- **Class attributes:** `name`, `category`, `description`, `gpu`, `two_pass`, `state_frames` (memory held,
  counted in float32 frames, which the memory plan relies on).
- **Main methods:** `reset()`, `update(batch)` and `result()`. A two-pass processor adds `begin_second_pass()`
  and `update_second_pass(batch)`.
- **Optional hooks:**
  - `begin(n_frames, frame_shape, indices)`: told how many frames are coming, in which order. Used by
    time-based effects.
  - `frames_needed(n_frames, frame_shape)`: return a subset of frames, and the engine reads only those, e.g.
    a banded time-slice.
  - `window_add`, `window_remove`, `window_result`: exact running sums for rolling windows (`average`,
    `temporal_std`).
- **Same code on CPU and GPU:** write it once with `xp = xp_of(array)`, and it runs on NumPy or CuPy.

**Rolling windows** (`render_windows`, `render_video`, `render_frames`): each frame is decoded and uploaded once.
The window cache lives in VRAM or RAM, as `MemoryPlan` decides. Deflicker corrects frames as they enter the cache.
Interpolation writes the in-between frames to the sinks.

**Outputs:**
- `outputs.py`: numbered stills (4 writer threads), plus one FFmpeg process per video output.
- Video data goes through the pipe in 4 MB writes, because huge single writes are slow on Windows.
- `colour.py` packs BT.709 limited-range Y'CbCr on the GPU.

**Metadata** (`metadata.py`):
- `summarise()` builds the title, caption and facts (frames, interval, exposure, camera, lens).
- `files.save_image` writes them. With exiftool, EXIF, XMP and IPTC are copied from the last source frame:
  `xmpMM` document IDs are excluded, and Lightroom `crs` settings are marked `AlreadyApplied=True`.
- The ICC profile is copied. tifffile is called with `metadata=None`, so it doesn't add its own JSON.

**Jobs and GUI:**
- `jobs.Job` is one saveable render description, run with `run_job` and `render_preview`.
- The GUI (`gui/app.py`) builds each effect's settings form from its constructor signature through
  `jobs.processor_params`.
- Long work runs on QThreads (`gui/worker.py`). Settings are saved in `%APPDATA%\timelapse-engine\gui.ini`.
- **Effect list** (`gui/effect_list.py`):
  - `_fill_process_combo` rebuilds the Effect list from the ticked effects (`effects_shown` in gui.ini; empty
    means all, so new effects appear) unless `effects_show_all` is on. The current or loaded effect is always
    listed, marked "not in your list" if it's hidden.
  - The first item, `EDIT_LIST`, opens `EffectListDialog`. When it's picked, the combo goes back to the
    previous effect.
  - **Trap:** Qt 6 closes a combo's pop-up *before* it changes the current item. `EffectCombo.picked_from_popup`
    counts "just closed" so that picking from the pop-up opens the dialog but the mouse wheel doesn't.

**Raw sorter** (`sorter.py`, GUI `gui/sort_panel.py`, CLI `timelapse-sort`). This is the step before Lightroom
and the engine: it splits a card dump of raws into one folder per sequence.
- **Scan:** `collect_shots` groups same-name files into a `Shot` (raw preferred as the main file; JPEG and
  `.xmp` / `.RW2.xmp` sidecars are companions). `read_shot_times` runs exiftool (`-fast`) in up to 4 parallel
  processes, with chunks passed in UTF-8 argfiles. It reads `SubSecDateTimeOriginal`, camera and serial, and
  `SequenceNumber`. Fallbacks: Pillow for JPEG/TIFF, then the file modified time.
- **Find sequences:** `spacing_runs` works on one camera at a time. A run continues while each gap is within
  `tolerance` (default 25 %) of the median of the last 5 gaps. The slack is 0.1 s with sub-seconds, 1 s
  without. A shot counter that restarts also cuts, but only for cameras whose counter really counts up.
  `find_sequences` turns runs of at least `min_frames` shots into `Group`s; everything else is single shots.
- **Move:** `plan_moves` never overwrites; on a name clash every file of the shot gets the same `_2`.
  `sort_files` logs each move as it goes, in a `try/finally`, so a cancelled or failed sort can still be undone.
- **Layout:** files go to `<sequence>\raw` with an empty `<sequence>\export` beside it. That's the owner's
  workflow: import the raw folder into Lightroom, export TIFFs into export, and load export in the engine, which
  names outputs after `<sequence>`. `_make_dirs` logs every folder it creates, so undo removes them, but only
  while they're empty. Single shots are always flat. `subfolders=False` (`--flat`, or the GUI checkbox) puts
  the files straight into each folder.
- **GUI:** tolerance, fewest-frames and counter changes call `find_sequences` again on the shots already read,
  without re-reading the files. Previews come from the embedded `PreviewImage` / `JpgFromRaw` (about 0.4 s
  each through exiftool).

## Adding an effect

1. **Class:** in `effects.py`, add a `Processor` subclass decorated with `@register`, and set `name`,
   `category` (one of `processors.CATEGORY_ORDER`), `description` and `state_frames`. Settings are keyword
   arguments to `__init__` with defaults. Choice settings are strings.
2. **Settings form:** add a label, range, step and help text for each new setting to `jobs._PARAM_HINTS`, and
   list the options of string settings in `jobs._PARAM_CHOICES`.
3. **Names:** add a display name to `gui/app.py` `DISPLAY_NAMES` and a caption label to `metadata.PROCESS_LABELS`.
4. **Test:** add a behaviour test to `tests/test_effects.py`. It should check what the effect claims (e.g.
   earlier events get earlier hues), not just that it runs.
5. **Check on real footage:** run it at full resolution on real footage, watch peak VRAM, and add it to the
   README effects table.

## Conventions and traps

- **Engine output:** the engine never prints or calls `input()`. Progress goes through the callback, and a
  `CancelToken` stops a job. Logging goes to the `"timelapse_engine"` logger.
- **Linear algebra on the GPU fails on the desktop.** CuPy's cuSOLVER DLL is missing, so `cupy.linalg.eigh`
  and SVD fail. Do the products on the GPU and the decompositions in NumPy on the CPU, as `eigen_colour` does.
- **CuPy reductions over the last (channel) axis are slow.** Use `processors._channel_max` and
  `_channel_mean`.
- **VRAM peaks:** accumulate in float64 one frame at a time, never as a float64 copy of the whole batch. That
  change took `clipped_average` from 11.9 to 7.7 GB. Prefer in-place results.
- **Histogram entropies:** put "no change" or "neutral" values in the middle of a bin, not on a bin edge, or grey
  areas pick up false entropy.
- **Video tests:** FFmpeg's own rgb48 to YUV path is about 0.3% off, so YUV tests check exact greys and allow a
  small tolerance.
- **Spelling:** British throughout (colour, stabilise, neighbourhood). Keep it consistent, including setting
  names.
- **The sorter handles the owner's original raws.** Never overwrite or delete them. Every file operation goes
  through the log, so Undo always works. Try new sorting logic with a dry run (no `--move`) on real cards before
  moving anything.

## Known issues and next ideas

- **ProPhoto sources give desaturated video.** The video path assumes sRGB/Rec.709 primaries. It needs a GPU
  ProPhoto-to-Rec.709 conversion, plus a gamut decision, before `colour.pack`. Stills are fine because they
  keep the ICC profile.
- **Preview** doesn't apply deflicker or interpolation.
- **Time-slice clock sweep:** `pattern="circular"` has a visible seam where the first and last frames meet.
  `mirror=True` avoids it.
- **Thin rings in a feathered radial time-slice** come from flicker in the footage, not a bug. Deflicker removes
  them.
- **Measured flicker:** the G10 sequence has about 2.5% frame-to-frame flicker, the Copenhill one 0.18%.
- **ProRes masters** are limited by the CPU encoder (2.7-7.6 fps at 20 MP). `master="hevc_nvenc"` is 3-7x faster.
- **Sorter and bracketed (HDR) timelapses:** the spacing alternates (short, short, long), so each bracket set
  looks like its own run and they end up as single shots. A fix would group the bracket sets first, then run the
  spacing test on them.
- **The sorter can't merge or split groups by hand.** You can only rename, untick, or change the tolerance and
  fewest frames.
- **Averages of TIFFs are gamma-space averages.** Only deflicker works in linear light, so `average` of encoded
  TIFFs is darker in moving highlights than a real long exposure. On terifa it was 5/255 darker overall and
  14/255 darker in the sea glitter. An optional linear-light average would fix this.
- **Laptop GPU:** CuPy once reported "no CUDA-capable device" while other GPU tests were running, and recovered
  afterwards. The app checks for the GPU once at start-up, so restart it if the status bar says CPU mode.
