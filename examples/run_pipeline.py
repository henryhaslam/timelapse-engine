# %% [markdown]
# # Timelapse pipeline
# Equivalent of the original `image_tests.ipynb` driver cell, using `timelapse_engine`.
# The `# %%` markers make each block a runnable cell in VS Code / Jupyter.

# %%
import timelapse_engine as te

te.enable_console_logging()
progress = te.ConsoleProgress()

# %%
# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
JOBS = [
    # (input folder or video,           output folder,              name)
    (r"E:\TL temp\G10_handle\export",    r"E:\TL temp\G10_handle",   "G10_handle"),
]

STILL_FORMAT = "tiff"          # "tiff" (16-bit, keeps Lightroom XMP) or "jpg"
STILL_PROCESSES = [            # any of te.list_processors()
    # "average",
    # "maximum",
    # "temporal_entropy",
    # "motion_overlay",
]
PROCESS_PARAMS = {             # per-process settings
    "temporal_entropy": {"hist_bins": 32},
    "motion_overlay": {"motion_threshold": 20, "intensity_priority": True},
}

RUN_STILL_STACKS = False
RUN_CLEAN_VIDEO = True         # 1:1 frames, no blending
RUN_BLENDED_VIDEO = True

BLEND_PROCESS = "average"
BLEND, STEP, FPS = 8, 3, 24

# %%
# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
for source_path, export_dir, name in JOBS:
    with te.open_source(source_path) as source:   # scans, EXIF-sorts and validates once

        if RUN_STILL_STACKS:
            for process in STILL_PROCESSES:
                te.stack_to_file(source, process, export_dir, name, ext=STILL_FORMAT,
                                 batch_size=3 if process == "temporal_entropy" else 16,
                                 progress=progress, **PROCESS_PARAMS.get(process, {}))

        if RUN_CLEAN_VIDEO:
            te.render_video(source, "average", export_dir, f"{name}_RAW", blend=1, step=1, fps=FPS,
                            master="prores", preview_1080p=False, progress=progress)

        if RUN_BLENDED_VIDEO:
            te.render_video(source, BLEND_PROCESS, export_dir, f"{name}_{BLEND_PROCESS}_{BLEND}_s{STEP}",
                            blend=BLEND, step=STEP, fps=FPS, master="prores", preview_1080p=False,
                            progress=progress, **PROCESS_PARAMS.get(BLEND_PROCESS, {}))

# %% [markdown]
# ## Other jobs
# ```python
# # Video file -> single still
# te.stack_to_file(r"E:\clip.mp4", "motion_overlay", r"E:\out", motion_threshold=20, progress=progress)
#
# # Video -> blended video
# te.render_video(r"E:\clip.mp4", "maximum", r"E:\out", blend=10, step=1, progress=progress)
#
# # Rolling window -> numbered stills
# te.render_frames(r"E:\TL\shoot", "average", r"E:\TL\frames", blend=8, step=3, ext="jpg", progress=progress)
#
# # Compare motion thresholds
# te.sweep_parameter(r"E:\clip.mp4", "motion_overlay", "motion_threshold", [5, 10, 20, 40], r"E:\out")
#
# # Stabilise
# from timelapse_engine import stabilise
# seq = te.ImageSequence(r"E:\TL\shoot")
# points, ref = stabilise.PointSelector(seq.paths).run()
# traj = stabilise.track_points(seq.paths, points, ref, progress=progress)
# stabilise.plot_point_drift(traj, ref)
# dx, dy = stabilise.compute_translation(traj, reference_index=ref)
# stabilise.apply_translation(seq.paths, r"E:\TL\shoot_stab", dx, dy, progress=progress)
# ```
