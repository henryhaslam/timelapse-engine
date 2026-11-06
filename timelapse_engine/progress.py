"""Progress reporting and cancellation.

The engine never prints or prompts. Instead every long-running function accepts:

* ``progress`` - any callable ``progress(done, total, message)``. Use
  :class:`ConsoleProgress` in scripts/notebooks; a GUI passes its own callback.
* ``cancel`` - a :class:`CancelToken`. Calling ``token.cancel()`` from another
  thread makes the running job raise :class:`Cancelled` at the next frame/batch.

Text messages go through the standard ``logging`` module under the
``"timelapse_engine"`` logger; call :func:`enable_console_logging` to see them.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Callable, Optional

ProgressCallback = Callable[[int, int, str], None]

logger = logging.getLogger("timelapse_engine")
logger.addHandler(logging.NullHandler())


def enable_console_logging(level: int = logging.INFO) -> None:
    """Print engine log messages to stdout (handy in notebooks)."""
    if not any(getattr(h, "_timelapse_console", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._timelapse_console = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    logger.setLevel(level)


class Cancelled(Exception):
    """Raised inside a job when its CancelToken has been triggered."""


class CancelToken:
    """Thread-safe cancellation flag shared between a job and its caller."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def check(self) -> None:
        if self._event.is_set():
            raise Cancelled("Job cancelled")


def check_cancel(cancel: Optional[CancelToken]) -> None:
    if cancel is not None:
        cancel.check()


def report(progress: Optional[ProgressCallback], done: int, total: int, message: str = "") -> None:
    if progress is not None:
        progress(done, total, message)


class ConsoleProgress:
    """Single-line terminal progress bar with ETA.

    The timer restarts whenever the message changes or ``done`` goes back to 0,
    so one instance can be reused across several phases/jobs.
    """

    def __init__(self, bar_length: int = 30, stream=None) -> None:
        self.bar_length = bar_length
        self.stream = stream or sys.stdout
        self._start = time.time()
        self._message: Optional[str] = None
        self._last_done = -1

    def __call__(self, done: int, total: int, message: str = "") -> None:
        if message != self._message or done < self._last_done or done == 0:
            self._start = time.time()
            self._message = message
        self._last_done = done

        total = max(total, 1)
        fraction = min(done / total, 1.0)
        filled = int(self.bar_length * fraction)
        bar = "█" * filled + "-" * (self.bar_length - filled)

        elapsed = time.time() - self._start
        eta = (elapsed / done) * (total - done) if done else 0.0
        eta_str = time.strftime("%H:%M:%S", time.gmtime(eta))
        prefix = f"{message} " if message else ""

        self.stream.write(f"\r{prefix}[{bar}] {fraction * 100:5.1f}%  ETA: {eta_str}  ({done}/{total})")
        if done >= total:
            self.stream.write("\n")
        self.stream.flush()
