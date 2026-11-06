"""Translation stabilisation: pick reference points, track them, and shift each frame.

Typical use::

    seq = ImageSequence(r"E:\\TL\\shoot")
    points, ref = PointSelector(seq.paths).run()
    traj = track_points(seq.paths, points, ref)
    dx, dy = compute_translation(traj, reference_index=ref)
    apply_translation(seq.paths, r"E:\\TL\\shoot_stab", dx, dy)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .progress import CancelToken, ProgressCallback, check_cancel, report

LK_PARAMS = dict(winSize=(21, 21), maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))


def _imread(path, flags=cv2.IMREAD_COLOR) -> np.ndarray:
    """cv2.imread that also works with non-ASCII Windows paths."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"Could not read image: {path}")
    return img


def _imwrite(path, img, jpeg_quality: int = 100) -> None:
    ext = Path(path).suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if not ok:
        raise IOError(f"Could not encode image: {path}")
    buf.tofile(str(path))


def _gray(path) -> np.ndarray:
    return _imread(path, cv2.IMREAD_GRAYSCALE)


class PointSelector:
    """OpenCV window for clicking stable reference points.

    Keys: A/D previous/next frame, Z undo last point, S save, Esc cancel.
    ``run()`` returns (points, frame_index) or (None, None) if cancelled.
    Points are in full-resolution image coordinates.
    """

    def __init__(self, paths: Sequence[str], max_display_size: int = 1200):
        if not paths:
            raise ValueError("No frames given to PointSelector.")
        self.paths = list(paths)
        self.max_display_size = max_display_size
        self.frame_index = 0
        self.points: List[Tuple[int, int]] = []
        self.scale = 1.0
        self.window_name = "Point Selector"

    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append((int(x / self.scale), int(y / self.scale)))

    def _display_frame(self) -> np.ndarray:
        img = _imread(self.paths[self.frame_index])
        longest = max(img.shape[:2])
        self.scale = min(1.0, self.max_display_size / longest)
        if self.scale < 1.0:
            img = cv2.resize(img, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        for x, y in self.points:
            cv2.circle(img, (int(x * self.scale), int(y * self.scale)), 5, (0, 255, 0), -1)
        cv2.putText(img, f"{self.frame_index + 1}/{len(self.paths)}  points: {len(self.points)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_mouse)
        try:
            while True:
                cv2.imshow(self.window_name, self._display_frame())
                key = cv2.waitKey(30) & 0xFF
                if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
                    return None, None  # window closed with the X button
                if key == ord("a"):
                    self.frame_index = max(0, self.frame_index - 1)
                elif key == ord("d"):
                    self.frame_index = min(len(self.paths) - 1, self.frame_index + 1)
                elif key == ord("z") and self.points:
                    self.points.pop()
                elif key == ord("s"):
                    return list(self.points), self.frame_index
                elif key == 27:
                    return None, None
        finally:
            cv2.destroyWindow(self.window_name)


def track_points(paths: Sequence[str], points, start_index: int = 0,
                 progress: Optional[ProgressCallback] = None, cancel: Optional[CancelToken] = None) -> np.ndarray:
    """Track points with Lucas-Kanade optical flow, forwards and backwards from ``start_index``.

    Returns an array (num_frames, num_points, 2). If a point is lost in a frame
    it keeps its last known position.
    """
    n = len(paths)
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    trajectory = np.zeros((n, len(pts), 2), np.float32)
    trajectory[start_index] = pts[:, 0, :]
    total = n - 1
    done = 0

    for direction in (1, -1):
        prev_gray = _gray(paths[start_index])
        p0 = pts.copy()
        stop = n if direction == 1 else -1
        for i in range(start_index + direction, stop, direction):
            check_cancel(cancel)
            gray = _gray(paths[i])
            p1, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **LK_PARAMS)
            lost = status.ravel() == 0
            p1[lost] = p0[lost]
            trajectory[i] = p1[:, 0, :]
            prev_gray, p0 = gray, p1
            done += 1
            report(progress, done, total, "tracking")
    return trajectory


def compute_translation(trajectory: np.ndarray, reference_index: int = 0,
                        smooth_sigma: float = 1.25) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame (dx, dy) drift relative to ``reference_index``: median over points, Gaussian-smoothed."""
    origin = trajectory[reference_index]
    dx = np.median(trajectory[:, :, 0] - origin[:, 0], axis=1)
    dy = np.median(trajectory[:, :, 1] - origin[:, 1], axis=1)
    if smooth_sigma:
        dx, dy = gaussian_filter1d(dx, smooth_sigma), gaussian_filter1d(dy, smooth_sigma)
    return dx, dy


def plot_point_drift(trajectory: np.ndarray, reference_index: int = 0):
    """Two plots: ΔX vs ΔY coloured by frame, and total drift over time."""
    import matplotlib.pyplot as plt

    n, num_points, _ = trajectory.shape
    origin = trajectory[reference_index]
    t = np.arange(n)

    fig1, ax1 = plt.subplots(figsize=(10, 10))
    scatter = None
    for p in range(num_points):
        dx = trajectory[:, p, 0] - origin[p, 0]
        dy = trajectory[:, p, 1] - origin[p, 1]
        scatter = ax1.scatter(dx, dy, c=t, cmap="viridis", s=20, alpha=0.8, label=f"Point {p}")
        ax1.plot(dx, dy, alpha=0.3)
        ax1.scatter(dx[reference_index], dy[reference_index], c="red", s=80, marker="x")
    ax1.set(title="XY drift of tracking points (colour = frame)", xlabel="ΔX (px)", ylabel="ΔY (px)")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")
    fig1.colorbar(scatter, ax=ax1, label="Frame")

    fig2, ax2 = plt.subplots(figsize=(12, 8))
    for p in range(num_points):
        d = trajectory[:, p] - origin[p]
        ax2.plot(np.hypot(d[:, 0], d[:, 1]), label=f"Point {p}")
    ax2.set(title="Point drift over time", xlabel="Frame", ylabel="Drift (px)")
    ax2.grid(True)
    ax2.legend()
    plt.show()
    return fig1, fig2


def apply_translation(paths: Sequence[str], out_dir, dx: np.ndarray, dy: np.ndarray,
                      jpeg_quality: int = 100, progress: Optional[ProgressCallback] = None,
                      cancel: Optional[CancelToken] = None) -> List[Path]:
    """Shift each frame by (-dx, -dy) and save to ``out_dir`` with the same filename.

    Keeps 16-bit TIFF depth, copies JPEG EXIF (so capture-time sorting still
    works on the output) and preserves file modified times.
    """
    if len(dx) != len(paths) or len(dy) != len(paths):
        raise ValueError(f"Got {len(paths)} frames but {len(dx)} offsets.")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, src in enumerate(paths):
        check_cancel(cancel)
        frame = _imread(src, cv2.IMREAD_UNCHANGED)
        h, w = frame.shape[:2]
        matrix = np.float32([[1, 0, -dx[i]], [0, 1, -dy[i]]])
        shifted = cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR)
        dst = out_dir / Path(src).name
        _imwrite(dst, shifted, jpeg_quality)
        if dst.suffix.lower() in (".jpg", ".jpeg"):
            try:
                import piexif
                piexif.transplant(str(src), str(dst))
            except Exception:
                pass  # no EXIF in source
        st = os.stat(src)
        os.utime(dst, (st.st_atime, st.st_mtime))
        written.append(dst)
        report(progress, i + 1, len(paths), "stabilising")
    return written
