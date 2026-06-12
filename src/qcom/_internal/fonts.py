"""
Internal font helpers for publication-style Matplotlib plots.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable


def _default_times_new_roman_paths() -> tuple[Path, ...]:
    return (
        Path("/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/Times New Roman.ttf"),
        Path("/usr/share/fonts/truetype/msttcorefonts/times.ttf"),
        Path("/Library/Fonts/Times New Roman.ttf"),
        Path("/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        Path(r"C:\Windows\Fonts\Times New Roman.ttf"),
        Path(r"C:\Windows\Fonts\times.ttf"),
        Path(r"C:\Windows\Fonts\timesbd.ttf"),
        Path(r"C:\Windows\Fonts\timesi.ttf"),
        Path(r"C:\Windows\Fonts\timesbi.ttf"),
    )


@lru_cache(maxsize=None)
def _register_times_new_roman(candidate_paths: tuple[str, ...]) -> bool:
    from matplotlib import font_manager

    for candidate in candidate_paths:
        path = Path(candidate)
        if not path.is_file():
            continue

        font_manager.fontManager.addfont(str(path))
        return True

    return False


def publication_font_context(
    candidate_paths: Iterable[str | Path] | None = None,
) -> dict[Any, Any]:
    """
    Return a Matplotlib rcParams mapping for publication-style serif plots.

    The helper tries to register Times New Roman from common system locations.
    If the font is unavailable, it quietly falls back to other serif families.
    """
    if candidate_paths is None:
        candidate_source: Iterable[str | Path] = _default_times_new_roman_paths()
    else:
        candidate_source = candidate_paths

    paths = tuple(str(Path(path)) for path in candidate_source)
    has_times_new_roman = _register_times_new_roman(paths)

    serif_fonts = ["Times New Roman"] if has_times_new_roman else []
    serif_fonts.extend(["Times", "Liberation Serif", "DejaVu Serif"])

    return {
        "font.family": "serif",
        "font.serif": serif_fonts,
        "mathtext.fontset": "stix",
    }
