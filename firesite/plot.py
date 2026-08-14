# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Render a fire recurrence map.

Two variables, two channels: point size is how many detections a cell has
accumulated, colour is in how many distinct years it burned. Collapsing them into
one channel hides the case that decides a camera site, which is ground that burns
a little but burns every year.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.collections
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle

from .hotspots import rank_cells

# Sequential single-hue ramp, light to dark, validated for colour-vision
# deficiency against a light surface. Plus one reserved accent for the site.
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
ACCENT = "#d03b3b"
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"

# The one-year band is isolated on purpose: a cell that burned once in a decade
# is noise for siting, and folding it in with three-year cells produces a huge
# pale band that buries the signal.
BANDS = [(1, 1, "1 year"), (2, 4, "2-4 years"), (5, 7, "5-7 years"), (8, 999, "8+ years")]
ALPHAS = [0.28, 0.7, 0.9, 0.95]


def recurrence_map(
    detections: pd.DataFrame,
    output: Path,
    site: tuple[float, float] | None = None,
    radius_km: float = 15.0,
    cell_deg: float = 0.02,
    title: str = "Fire recurrence",
    margin_factor: float = 2.5,
) -> Path:
    """Render the map. With a site given, the view crops around it.

    Without the crop the site is a speck inside whatever box was downloaded,
    which is the wrong picture: the question is what this camera sees, not what
    the country did.
    """
    cells = rank_cells(detections, cell_deg=cell_deg)

    fig, ax = plt.subplots(figsize=(11, 10), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    if site is not None:
        lat, lon = site
        # Drawn under the data so it frames without hiding.
        dlon = 1 / (111.320 * math.cos(math.radians(lat)))
        ax.add_patch(
            Circle(
                (lon, lat),
                radius_km * dlon,
                facecolor=ACCENT,
                alpha=0.05,
                edgecolor=ACCENT,
                linewidth=1.5,
                linestyle=(0, (6, 4)),
                zorder=1,
            )
        )

    for (low, high, label), colour, alpha in zip(BANDS, RAMP, ALPHAS, strict=False):
        band = cells[(cells["years"] >= low) & (cells["years"] <= high)]
        if band.empty:
            continue
        ax.scatter(
            band["lon"],
            band["lat"],
            s=12 + 4.5 * band["detections"],
            c=colour,
            alpha=alpha,
            linewidths=0.6,
            edgecolors=SURFACE,
            label=f"{label}  (n={len(band)})",
            zorder=3 + (high > 4),
        )

    if site is not None:
        lat, lon = site
        ax.scatter(
            [lon],
            [lat],
            marker="X",
            s=260,
            c=ACCENT,
            edgecolors=SURFACE,
            linewidths=1.6,
            zorder=6,
        )
        # A plain text label over a dense scatter is unreadable however far it is
        # offset, so it sits on a plate the colour of the surface.
        ax.annotate(
            f"camera site\n{radius_km:g} km reach",
            (lon, lat),
            textcoords="offset points",
            xytext=(20, -42),
            fontsize=10,
            color=INK,
            weight="bold",
            zorder=7,
            bbox={
                "boxstyle": "round,pad=0.45",
                "facecolor": SURFACE,
                "edgecolor": ACCENT,
                "linewidth": 1.0,
                "alpha": 0.94,
            },
            arrowprops={
                "arrowstyle": "-",
                "color": ACCENT,
                "linewidth": 1.2,
                "shrinkA": 2,
                "shrinkB": 8,
            },
        )

    ax.set_title(title, fontsize=15, color=INK, weight="bold", loc="left", pad=16)
    ax.text(
        0,
        1.012,
        f"{len(detections)} detections. "
        "Size = accumulated detections, colour = distinct years with fire.",
        transform=ax.transAxes,
        fontsize=9.5,
        color=INK_2,
    )
    ax.set_xlabel("Longitude", fontsize=9.5, color=MUTED)
    ax.set_ylabel("Latitude", fontsize=9.5, color=MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.set_aspect("equal", adjustable="box")
    if site is not None:
        lat, lon = site
        dlat = margin_factor * radius_km / 110.574
        dlon = margin_factor * radius_km / (111.320 * math.cos(math.radians(lat)))
        ax.set_xlim(lon - dlon, lon + dlon)
        ax.set_ylim(lat - dlat, lat + dlat)

    # The legend sits over data, so it needs an opaque plate. frameon=False looks
    # cleaner on an empty corner and unreadable on a full one.
    legend = ax.legend(
        title="Distinct years with fire",
        loc="lower left",
        frameon=True,
        facecolor=SURFACE,
        edgecolor="#e1e0d9",
        framealpha=0.93,
        fontsize=9.5,
        labelspacing=0.8,
        borderpad=0.8,
        handletextpad=1.0,
    )
    legend.get_title().set_fontsize(9.5)
    legend.get_title().set_color(INK_2)
    for text in legend.get_texts():
        text.set_color(INK_2)
    # Legend markers encode a band, not a magnitude; equal sizes stop them being
    # read as a detection-count scale.
    for handle in legend.legend_handles:
        if isinstance(handle, matplotlib.collections.PathCollection):
            handle.set_sizes([70])
            handle.set_alpha(0.95)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return output
