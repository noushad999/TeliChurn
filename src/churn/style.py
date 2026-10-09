"""Shared visual language for every chart and README figure.

Cool slate neutrals with a teal accent, violet and amber as the second and third series, and one
sans-serif family throughout. The three categorical slots pass the colour-blind separation and
contrast checks on both the light and the dark surface. The bundled Source Sans 3 font is used
when the repo is checked out, with matplotlib defaults as the fallback.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager  # noqa: E402

LIGHT = {
    "surface": "#f8fafc",
    "card": "#eef2f6",
    "edge": "#d9e0e8",
    "ink": "#0f172a",
    "ink_2": "#475569",
    "muted": "#64748b",
    "grid": "#e5e9ef",
    "accent": "#00879e",
    "violet": "#7058d8",
    "amber": "#b07400",
    "accent_soft": "#d2ecf0",
}
DARK = {
    "surface": "#111827",
    "card": "#1b2433",
    "edge": "#2c3747",
    "ink": "#f1f5f9",
    "ink_2": "#cbd5e1",
    "muted": "#94a3b8",
    "grid": "#253041",
    "accent": "#22a6ba",
    "violet": "#8f7ff0",
    "amber": "#c2850f",
    "accent_soft": "#163e47",
}
SERIES = [LIGHT["accent"], LIGHT["violet"], LIGHT["amber"]]

# repo checkout first, then the working directory (installed package, e.g. the container's /app)
FONT_DIR = next(
    (
        d
        for d in (Path(__file__).resolve().parents[2] / "docs/assets/fonts", Path.cwd() / "docs/assets/fonts")
        if d.exists()
    ),
    None,
)
SANS = "DejaVu Sans"
if FONT_DIR is not None:
    for ttf in FONT_DIR.glob("*.ttf"):
        font_manager.fontManager.addfont(str(ttf))
    SANS = "Source Sans 3"

matplotlib.rcParams.update(
    {
        "font.family": SANS,
        "svg.fonttype": "path",
        "axes.titlepad": 12,
    }
)
