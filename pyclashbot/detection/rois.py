"""Region-of-interest definitions for template searches."""

from __future__ import annotations

ROI_REGISTRY: dict[str, tuple[int, int, int, int]] = {
    "deck_tabs": (0, 80, 420, 160),
    "deck_tabs/switch_deck": (0, 80, 420, 160),
    "fight_mode_1v1": (20, 140, 260, 610),
    "fight_mode_2v2": (20, 140, 260, 610),
    "fight_mode_trophy_road": (20, 140, 260, 610),
    "selected_1v1_on_main": (260, 440, 380, 560),
    "selected_2v2_on_main": (260, 440, 380, 560),
    "selected_trophy_road_on_main": (260, 440, 420, 560),
    "ok_post_battle_button": (120, 500, 360, 700),
    "exit_battle_button": (120, 500, 360, 700),
}


def get_roi_for_template(name: str) -> tuple[int, int, int, int] | None:
    """Return the ROI for a template key, falling back to prefix matches."""

    if name in ROI_REGISTRY:
        return ROI_REGISTRY[name]

    prefix = name.split("/")[0]
    return ROI_REGISTRY.get(prefix)
