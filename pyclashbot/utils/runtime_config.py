"""Runtime configuration shared across modules."""

from __future__ import annotations

import multiprocessing
import threading
from dataclasses import dataclass, field
from typing import Optional

import cv2


@dataclass
class RuntimeConfig:
    """Container for runtime-tunable options."""

    capture_backend: str = "dxcam"
    capture_title: Optional[str] = None
    capture_downscale: float = 0.75
    opencv_threads: int = field(default_factory=lambda: multiprocessing.cpu_count() or 1)

    def apply_opencv_settings(self) -> None:
        """Ensure OpenCV uses the optimized execution path."""

        try:
            cv2.setUseOptimized(True)
        except Exception:  # pragma: no cover - platform dependent
            pass
        try:
            cv2.setNumThreads(max(1, int(self.opencv_threads)))
        except Exception:  # pragma: no cover - platform dependent
            pass


_runtime_config = RuntimeConfig()
_lock = threading.Lock()


def configure_runtime(
    *, capture_backend: Optional[str] = None, capture_title: Optional[str] = None, capture_downscale: Optional[float] = None
) -> RuntimeConfig:
    """Update runtime configuration values."""

    with _lock:
        if capture_backend:
            _runtime_config.capture_backend = capture_backend
        if capture_title is not None:
            _runtime_config.capture_title = capture_title or None
        if capture_downscale:
            _runtime_config.capture_downscale = max(0.1, min(1.0, capture_downscale))
        _runtime_config.opencv_threads = multiprocessing.cpu_count() or _runtime_config.opencv_threads
        _runtime_config.apply_opencv_settings()
        return _runtime_config


def get_runtime_config() -> RuntimeConfig:
    """Return the current runtime configuration instance."""

    with _lock:
        return _runtime_config
