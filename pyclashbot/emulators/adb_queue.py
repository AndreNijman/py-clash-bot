"""Persistent ADB command queue for low-latency input."""

from __future__ import annotations

import queue
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class ADBCommand:
    """Represent a queued command."""

    payload: str
    debounce_key: Optional[str] = None
    timestamp: float = 0.0


class ADBCommandQueue:
    """Maintain a persistent adb shell session and enqueue commands."""

    def __init__(self, adb_path: str, server_port: int, device_serial: str | None, *, debounce_window: float = 0.12):
        self._adb_path = adb_path
        self._server_port = server_port
        self._device_serial = device_serial
        self._debounce_window = debounce_window
        self._queue: "queue.Queue[ADBCommand]" = queue.Queue(maxsize=64)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._process: Optional[subprocess.Popen[str]] = None
        self._lock = threading.Lock()
        self._last_sent: dict[str, float] = {}

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        args = [self._adb_path, "-P", str(self._server_port)]
        if self._device_serial:
            args.extend(["-s", self._device_serial])
        args.append("shell")

        self._process = subprocess.Popen(
            args,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ADBInputThread", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
            except Exception:
                pass

    def __del__(self) -> None:  # pragma: no cover - cleanup
        try:
            self.stop()
        except Exception:
            pass

    def _should_debounce(self, command: ADBCommand) -> bool:
        if not command.debounce_key:
            return False
        now = time.perf_counter()
        with self._lock:
            last = self._last_sent.get(command.debounce_key)
            if last and (now - last) < self._debounce_window:
                return True
            self._last_sent[command.debounce_key] = now
        return False

    def enqueue(self, payload: str, *, debounce_key: Optional[str] = None) -> None:
        cmd = ADBCommand(payload=payload, debounce_key=debounce_key, timestamp=time.perf_counter())
        try:
            self._queue.put_nowait(cmd)
        except queue.Full:
            # Drop oldest entry to preserve responsiveness
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(cmd)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                command = self._queue.get(timeout=0.05)
            except queue.Empty:
                continue

            if self._should_debounce(command):
                continue

            proc = self._process
            if not proc or proc.stdin is None or proc.poll() is not None:
                continue

            try:
                proc.stdin.write(command.payload + "\n")
                proc.stdin.flush()
            except Exception:
                # Attempt to restart on failure
                self._restart_process()

    def _restart_process(self) -> None:
        self.stop()
        self.start()
