"""Template caching and ROI management for vision routines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from pyclashbot.emulators.capture import FrameData
from pyclashbot.utils.runtime_config import get_runtime_config

from .rois import ROI_REGISTRY, get_roi_for_template


@dataclass(slots=True)
class TemplateEntry:
    name: str
    gray: np.ndarray
    shape: tuple[int, int]


class TemplateCache:
    """Cache templates scaled to the configured capture size."""

    def __init__(self) -> None:
        runtime_cfg = get_runtime_config()
        self._downscale = runtime_cfg.capture_downscale
        self._base_path = Path(__file__).resolve().parent / "reference_images"
        self._templates: dict[str, list[TemplateEntry]] = {}
        self._thresholds: dict[str, float] = self._default_thresholds()
        self._load_templates()

    def _default_thresholds(self) -> dict[str, float]:
        return {
            "deck_tabs": 0.95,
            "deck_tabs/switch_deck": 0.98,
            "fight_mode_1v1": 0.85,
            "fight_mode_2v2": 0.85,
            "fight_mode_trophy_road": 0.85,
            "ok_post_battle_button": 0.85,
            "exit_battle_button": 0.9,
            "selected_1v1_on_main": 0.9,
            "selected_2v2_on_main": 0.9,
            "selected_trophy_road_on_main": 0.9,
        }

    @property
    def roi_count(self) -> int:
        return len(ROI_REGISTRY)

    def _iter_template_files(self) -> Iterable[Path]:
        for path in sorted(self._base_path.rglob("*.png")):
            yield path

    def _key_for_path(self, path: Path) -> str:
        rel = path.relative_to(self._base_path)
        parts = rel.parts
        if len(parts) == 1:
            return rel.stem
        return str(rel.parent).replace("\\", "/")

    def _load_templates(self) -> None:
        for image_path in self._iter_template_files():
            key = self._key_for_path(image_path)
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            if self._downscale and self._downscale != 1.0:
                image = cv2.resize(
                    image,
                    None,
                    fx=self._downscale,
                    fy=self._downscale,
                    interpolation=cv2.INTER_AREA,
                )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            entry = TemplateEntry(name=image_path.name, gray=gray, shape=gray.shape[::-1])
            self._templates.setdefault(key, []).append(entry)

    def find(self, frame: FrameData, folder: str, tolerance: float, subcrop: tuple[int, int, int, int] | None):
        templates = self._templates.get(folder)
        if not templates:
            # try prefix match for grouped templates (e.g., deck_tabs/deck_1)
            prefix = folder.split("/")[0]
            templates = []
            for key, entries in self._templates.items():
                if key.startswith(prefix):
                    templates.extend(entries)
            if not templates:
                return None

        roi = subcrop or get_roi_for_template(folder)
        search_gray = frame.gray
        offset_x = 0
        offset_y = 0
        if roi is not None:
            x1, y1, x2, y2 = frame.scale_rect(roi)
            x1 = max(0, min(search_gray.shape[1] - 1, x1))
            y1 = max(0, min(search_gray.shape[0] - 1, y1))
            x2 = max(x1 + 1, min(search_gray.shape[1], x2))
            y2 = max(y1 + 1, min(search_gray.shape[0], y2))
            search_gray = search_gray[y1:y2, x1:x2]
            offset_x = x1
            offset_y = y1

        threshold = self._thresholds.get(folder, tolerance)
        if threshold == tolerance:
            prefix = folder.split("/")[0]
            threshold = self._thresholds.get(prefix, threshold)
        best_location: tuple[int, int] | None = None
        best_score = -1.0

        for template in templates:
            if search_gray.shape[0] < template.gray.shape[0] or search_gray.shape[1] < template.gray.shape[1]:
                continue
            res = cv2.matchTemplate(search_gray, template.gray, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_location = (max_loc[0], max_loc[1])
                if max_val >= threshold:
                    match_x = max_loc[0] + offset_x
                    match_y = max_loc[1] + offset_y
                    return frame.to_original_coords(match_x, match_y)
        if best_location and best_score >= threshold:
            return frame.to_original_coords(best_location[0] + offset_x, best_location[1] + offset_y)
        return None


TEMPLATE_CACHE = TemplateCache()
