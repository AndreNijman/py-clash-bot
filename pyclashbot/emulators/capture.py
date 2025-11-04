"""Screen capture management for emulator windows."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from pyclashbot.utils.logger import Logger
from pyclashbot.utils.runtime_config import get_runtime_config


@dataclass(slots=True)
class FrameData:
    """Container for a captured frame."""

    bgr: np.ndarray
    scaled_bgr: np.ndarray
    gray: np.ndarray
    timestamp: float
    downscale: float
    original_shape: tuple[int, int, int]
    scaled_shape: tuple[int, int, int]

    def __array__(self, dtype=None) -> np.ndarray:  # pragma: no cover - numpy protocol
        return np.asarray(self.bgr, dtype=dtype)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.original_shape

    def __getitem__(self, key):  # pragma: no cover - passthrough indexing
        return self.bgr[key]

    def scale_rect(self, rect: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        """Convert a rectangle in emulator space to scaled frame space."""

        if self.downscale == 1.0:
            return rect
        x1, y1, x2, y2 = rect
        scale_x = self.scaled_shape[1] / max(1, self.original_shape[1])
        scale_y = self.scaled_shape[0] / max(1, self.original_shape[0])
        return (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )

    def to_original_coords(self, x: int, y: int) -> tuple[int, int]:
        """Map scaled coordinates back to emulator space."""

        if self.downscale == 1.0:
            return x, y
        scale_x = self.scaled_shape[1] / max(1, self.original_shape[1])
        scale_y = self.scaled_shape[0] / max(1, self.original_shape[0])
        return int(round(x / scale_x)), int(round(y / scale_y))


class CaptureBackend:
    """Base class for capture backends."""

    def start(self) -> None:
        """Prepare backend resources."""

    def stop(self) -> None:  # pragma: no cover - platform dependent cleanup
        """Release backend resources."""

    def grab(self) -> Optional[np.ndarray]:  # pragma: no cover - runtime only
        """Return the latest frame as a BGR numpy array."""


class DxcamBackend(CaptureBackend):
    """Capture backend using dxcam."""

    def __init__(self, window_title: Optional[str] = None):
        try:
            import dxcam
        except Exception as exc:  # pragma: no cover - dxcam only on Windows
            raise RuntimeError("dxcam backend is unavailable") from exc

        self._target = window_title
        self._camera = dxcam.create(output_idx=0, target_fps=120)

    def start(self) -> None:  # pragma: no cover - requires Windows
        if self._target:
            self._camera.start(target_fps=120, video_mode=True)
        else:
            self._camera.start(target_fps=120)

    def stop(self) -> None:  # pragma: no cover - requires Windows
        self._camera.stop()

    def grab(self) -> Optional[np.ndarray]:  # pragma: no cover - requires Windows
        frame = self._camera.get_latest_frame()
        if frame is None:
            return None
        return frame[:, :, ::-1].copy()


class MSSBackend(CaptureBackend):
    """Fallback backend using mss."""

    def __init__(self, window_title: Optional[str] = None):
        from mss import mss  # type: ignore

        self._sct = mss()
        self._monitor = self._sct.monitors[1]
        self._window_title = window_title

    def grab(self) -> Optional[np.ndarray]:  # pragma: no cover - requires Windows GUI
        raw = self._sct.grab(self._monitor)
        frame = np.array(raw, dtype=np.uint8)
        if frame.shape[2] == 4:
            frame = frame[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


class CaptureManager:
    """Manage a dedicated capture thread with a 2-frame ring buffer."""

    def __init__(self, logger: Logger, *, downscale: Optional[float] = None) -> None:
        self.logger = logger
        self._config = get_runtime_config()
        self._downscale = downscale if downscale is not None else self._config.capture_downscale
        backend_name = (self._config.capture_backend or "dxcam").lower()
        window_title = self._config.capture_title

        backend: CaptureBackend
        try:
            if backend_name == "dxcam":
                backend = DxcamBackend(window_title)
            else:
                backend = MSSBackend(window_title)
        except Exception as exc:
            if backend_name != "mss":
                logger.log(f"dxcam backend unavailable ({exc}); falling back to mss")
                backend = MSSBackend(window_title)
                backend_name = "mss"
            else:
                raise

        self._backend_name = backend_name
        self._backend = backend
        self._full_buffer = [None, None]  # type: ignore[list-item]
        self._scaled_buffer = [None, None]  # type: ignore[list-item]
        self._gray_buffer = [None, None]  # type: ignore[list-item]
        self._timestamps = [0.0, 0.0]
        self._original_shapes: list[Optional[tuple[int, int, int]]] = [None, None]
        self._scaled_shapes: list[Optional[tuple[int, int, int]]] = [None, None]
        self._index = -1
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._scratch_full: Optional[np.ndarray] = None
        self._scratch_scaled: Optional[np.ndarray] = None
        self._scratch_gray: Optional[np.ndarray] = None
        self._fps_window: list[float] = []
        self._logged_geometry = False

        logger.log(
            f"Capture backend: {backend_name} | title: {window_title or 'auto'} | downscale: {self._downscale:.2f}"
        )

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        try:
            self._backend.start()
        except Exception:
            pass

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="CaptureThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.5)
        try:
            self._backend.stop()
        except Exception:
            pass

    def _run(self) -> None:
        while not self._stop_event.is_set():
            full_frame = self._backend.grab()
            if full_frame is None:
                time.sleep(0.002)
                continue

            scaled_frame = full_frame
            if self._downscale and self._downscale != 1.0:
                scaled_frame = cv2.resize(
                    full_frame, None, fx=self._downscale, fy=self._downscale, interpolation=cv2.INTER_AREA
                )
            gray = cv2.cvtColor(scaled_frame, cv2.COLOR_BGR2GRAY)

            with self._lock:
                next_index = (self._index + 1) % 2

                if self._full_buffer[next_index] is None or self._full_buffer[next_index].shape != full_frame.shape:
                    self._full_buffer[next_index] = np.empty_like(full_frame)
                if self._scaled_buffer[next_index] is None or self._scaled_buffer[next_index].shape != scaled_frame.shape:
                    self._scaled_buffer[next_index] = np.empty_like(scaled_frame)
                if self._gray_buffer[next_index] is None or self._gray_buffer[next_index].shape != gray.shape:
                    self._gray_buffer[next_index] = np.empty_like(gray)

                np.copyto(self._full_buffer[next_index], full_frame)
                np.copyto(self._scaled_buffer[next_index], scaled_frame)
                np.copyto(self._gray_buffer[next_index], gray)
                self._timestamps[next_index] = time.perf_counter()
                self._original_shapes[next_index] = full_frame.shape
                self._scaled_shapes[next_index] = scaled_frame.shape
                self._index = next_index

                if not self._logged_geometry:
                    self.logger.log(
                        f"Capture geometry: full={full_frame.shape[1]}x{full_frame.shape[0]} scaled={scaled_frame.shape[1]}x{scaled_frame.shape[0]}"
                    )
                    self._logged_geometry = True

            if len(self._fps_window) > 200:
                self._fps_window.pop(0)
            self._fps_window.append(self._timestamps[self._index])

    def get_latest_frame(self) -> Optional[FrameData]:
        with self._lock:
            if self._index == -1:
                return None
            idx = self._index
            full = self._full_buffer[idx]
            scaled = self._scaled_buffer[idx]
            gray = self._gray_buffer[idx]
            if full is None or scaled is None or gray is None:
                return None

            if self._scratch_full is None or self._scratch_full.shape != full.shape:
                self._scratch_full = np.empty_like(full)
            if self._scratch_scaled is None or self._scratch_scaled.shape != scaled.shape:
                self._scratch_scaled = np.empty_like(scaled)
            if self._scratch_gray is None or self._scratch_gray.shape != gray.shape:
                self._scratch_gray = np.empty_like(gray)

            np.copyto(self._scratch_full, full)
            np.copyto(self._scratch_scaled, scaled)
            np.copyto(self._scratch_gray, gray)
            ts = self._timestamps[idx]
            original_shape = self._original_shapes[idx] or full.shape
            scaled_shape = self._scaled_shapes[idx] or scaled.shape

        return FrameData(
            bgr=self._scratch_full,
            scaled_bgr=self._scratch_scaled,
            gray=self._scratch_gray,
            timestamp=ts,
            downscale=self._downscale,
            original_shape=original_shape,
            scaled_shape=scaled_shape,
        )

    def fps(self) -> float:
        with self._lock:
            if len(self._fps_window) < 2:
                return 0.0
            duration = self._fps_window[-1] - self._fps_window[0]
            if duration <= 0:
                return 0.0
            return (len(self._fps_window) - 1) / duration
