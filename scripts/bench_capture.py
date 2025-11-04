"""Simple benchmark for capture throughput."""

from __future__ import annotations

import statistics
import time

from pyclashbot.emulators.capture import CaptureManager
from pyclashbot.utils.logger import Logger


def main() -> None:
    logger = Logger(timed=False)
    manager = CaptureManager(logger)
    manager.start()
    samples: list[float] = []

    end_time = time.perf_counter() + 5.0
    while time.perf_counter() < end_time:
        loop_start = time.perf_counter()
        frame = manager.get_latest_frame()
        if frame is None:
            time.sleep(0.01)
            continue
        _ = frame.gray.mean()
        samples.append((time.perf_counter() - loop_start) * 1000.0)
        time.sleep(0.005)

    fps = manager.fps()
    manager.stop()

    median_ms = statistics.median(samples) if samples else 0.0
    print(f"Capture FPS: {fps:.1f}")
    print(f"Median loop time: {median_ms:.2f} ms")


if __name__ == "__main__":
    main()
