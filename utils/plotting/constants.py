"""
Plotting constants used across utils.plotting.

Keep this module free of Plotly/Pandas imports to avoid heavy import side effects.
"""

from __future__ import annotations

from typing import Final, Tuple, Dict

# -----------------------------------------------------------------------------
# Color constants (Plotly-compatible RGB strings)
# -----------------------------------------------------------------------------

BAR_COLORS: Final[Tuple[str, str, str, str]] = (
    "rgb(251, 180, 174)",  # soft red
    "rgb(179, 205, 227)",  # soft blue
    "rgb(204, 235, 197)",  # soft green
    "rgb(222, 203, 228)",  # soft purple
)

# Backwards-compatible names (optional; remove once all call sites updated)
BAR_COLOR_1: Final[str] = BAR_COLORS[0]
BAR_COLOR_2: Final[str] = BAR_COLORS[1]
BAR_COLOR_3: Final[str] = BAR_COLORS[2]
BAR_COLOR_4: Final[str] = BAR_COLORS[3]

# -----------------------------------------------------------------------------
# Layout and rendering defaults
# -----------------------------------------------------------------------------

BASE_HEIGHT_PER_ROW: Final[int] = 30
FLAG_SIZE: Final[int] = 22
TEXT_SIZE: Final[int] = 22

# Plotly export scale. Higher values can increase memory/CPU usage and may hang.
EXPORT_SCALE: Final[int] = 1

SCALE = 1

# Colour-blind safe continent palette (Okabe-Ito), shared by any figure that
# encodes continent so the same continent reads the same colour throughout.
CONTINENT_COLORS: Final[Dict[str, str]] = {
    "Africa": "#E69F00",
    "Asia": "#0072B2",
    "Europe": "#009E73",
    "North America": "#D55E00",
    "Oceania": "#CC79A7",
    "South America": "#56B4E9",
}
CONTINENT_FALLBACK_COLOR: Final[str] = "#666666"

__all__ = [
    "CONTINENT_COLORS",
    "CONTINENT_FALLBACK_COLOR",
    "BAR_COLORS",
    "BAR_COLOR_1",
    "BAR_COLOR_2",
    "BAR_COLOR_3",
    "BAR_COLOR_4",
    "BASE_HEIGHT_PER_ROW",
    "FLAG_SIZE",
    "TEXT_SIZE",
    "EXPORT_SCALE",
    "SCALE"
]
