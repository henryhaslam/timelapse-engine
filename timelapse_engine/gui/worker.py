"""Background work for the GUI: one-off tasks on a thread pool, and the job queue runner."""

from __future__ import annotations

import logging
import traceback
from typing import Callable, List, Optional, Tuple

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

from ..jobs import Job, SourceCache, run_job
from ..progress import CancelToken, Cancelled, logger


def describe_error(exc: BaseException) -> str:
    if isinstance(exc, (ValueError, MemoryError, FileNotFoundError, FileExistsError, RuntimeError, TypeError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


class _TaskSignals(QObject):
    done = Signal(object)
    failed = Signal(str)


class _Task(QRunnable):
    def __init__(self, fn: Callable, args: tuple):
        super().__init__()
        self.fn, self.args = fn, args
        self.signals = _TaskSignals()

    def run(self):
        try:
            result = self.fn(*self.args)
        except Cancelled:
            return
        except BaseException as exc:  # report everything to the GUI instead of crashing a pool thread
            logger.debug(traceback.format_exc())
            self.signals.failed.emit(describe_error(exc))
            return
        self.signals.done.emit(result)


class TaskRunner(QObject):
    """Runs functions off the GUI thread and calls back on it."""

    def __init__(self, parent=None, max_threads: int = 2):
        super().__init__(parent)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(max_threads)
        self._live = set()

    def submit(self, fn: Callable, *args, on_done: Optional[Callable] = None,
               on_failed: Optional[Callable] = None) -> None:
        task = _Task(fn, args)
        task.setAutoDelete(False)
        self._live.add(task)
        if on_done:
            task.signals.done.connect(on_done)
        if on_failed:
            task.signals.failed.connect(on_failed)
        task.signals.done.connect(lambda _=None, t=task: self._live.discard(t))
        task.signals.failed.connect(lambda _=None, t=task: self._live.discard(t))
        self.pool.start(task)

    def wait(self, msecs: int = -1) -> bool:
        return self.pool.waitForDone(msecs)


class QueueRunner(QObject):
    """Runs jobs one after another on its own QThread.

    Emits progress for the job currently running; ``stop()`` cancels it and
    stops the queue.
    """

    job_started = Signal(int)
    job_progress = Signal(int, int, int, str)   # row, done, total, message
    job_finished = Signal(int, list)            # row, output paths
    job_failed = Signal(int, str)
    job_cancelled = Signal(int)
    finished = Signal()

    def __init__(self, jobs: List[Tuple[int, Job]], sources: SourceCache):
        super().__init__()
        self.jobs = jobs
        self.sources = sources
        self._stop = False
        self._cancel: Optional[CancelToken] = None

    @Slot()
    def run(self):
        try:
            for row, job in self.jobs:
                if self._stop:
                    break
                self._cancel = CancelToken()
                self.job_started.emit(row)
                try:
                    outs = run_job(job, self.sources, progress=lambda d, t, m, r=row: self.job_progress.emit(r, d, t, m),
                                   cancel=self._cancel)
                    self.job_finished.emit(row, [str(p) for p in outs])
                except Cancelled:
                    self.job_cancelled.emit(row)
                    break
                except BaseException as exc:
                    logger.error("Job failed: %s", describe_error(exc))
                    logger.debug(traceback.format_exc())
                    self.job_failed.emit(row, describe_error(exc))
        finally:
            self.finished.emit()

    def stop(self):
        self._stop = True
        if self._cancel is not None:
            self._cancel.cancel()


class _LogBridge(QObject):
    message = Signal(str, int)


class QtLogHandler(logging.Handler):
    """Forwards engine log records to the GUI log panel (thread-safe via a queued signal)."""

    def __init__(self):
        super().__init__()
        self.bridge = _LogBridge()
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record):
        try:
            self.bridge.message.emit(self.format(record), record.levelno)
        except RuntimeError:
            pass  # window already closed
