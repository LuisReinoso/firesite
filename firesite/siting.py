# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Turn a fire history into a camera placement and a lens choice.

Two questions, in order:

1. Where does a camera see the most of what has actually burned?
2. From that spot, what lens and sensor actually resolve a plume at that range?

The second question is the one that sinks projects. A site can cover 60% of the
historical fires and still be useless because at 14 km an incipient plume lands
on four pixels and no detector can find it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .firms import BBox
from .hotspots import bearing_deg, rank_cells, visible_from

# Compass sectors used to summarize where the fire sits relative to a site.
SECTORS = [
    ("N", 337.5, 22.5),
    ("NE", 22.5, 67.5),
    ("E", 67.5, 112.5),
    ("SE", 112.5, 157.5),
    ("S", 157.5, 202.5),
    ("SW", 202.5, 247.5),
    ("W", 247.5, 292.5),
    ("NW", 292.5, 337.5),
]

# Width of an incipient smoke column, in metres. Early detection means catching
# the plume while it is still this small; by the time it is 200 m across the
# neighbours have already called it in.
INCIPIENT_PLUME_M = 30.0

# Detectors need roughly this many pixels across the plume to fire reliably.
# Below it, published smoke models degrade fast regardless of architecture.
MIN_PIXELS_ON_TARGET = 8.0


@dataclass(frozen=True)
class RankedSite:
    """A candidate after its line of sight has actually been checked."""

    site: Site
    visible_detections: int
    visible_share: float


@dataclass(frozen=True)
class Site:
    lat: float
    lon: float
    detections: int
    cells: int
    coverage: float  # fraction of the study area's detections


def pixels_on_target(
    distance_km: float, sensor_px: int, fov_deg: float, plume_m: float = INCIPIENT_PLUME_M
) -> float:
    """How many horizontal pixels a plume of `plume_m` spans at `distance_km`.

    This is the number that decides which camera to buy. It falls linearly with
    distance and with field of view, so a wider lens is not free: it buys arc at
    the cost of range.
    """
    if distance_km <= 0:
        return float("inf")
    angular = plume_m / (distance_km * 1000.0)  # small-angle, radians
    return sensor_px * angular / math.radians(fov_deg)


def max_useful_range_km(
    sensor_px: int,
    fov_deg: float,
    plume_m: float = INCIPIENT_PLUME_M,
    min_px: float = MIN_PIXELS_ON_TARGET,
) -> float:
    """Distance beyond which an incipient plume is too small to detect."""
    return sensor_px * plume_m / (min_px * math.radians(fov_deg) * 1000.0)


def search_sites(
    cells: pd.DataFrame,
    bbox: BBox,
    total_detections: int,
    radius_km: float = 15.0,
    step_deg: float = 0.02,
    top: int = 10,
) -> list[Site]:
    """Grid-search candidate positions and rank them by coverage.

    Every node of the grid is treated as a possible camera. This is deliberately
    naive: it ignores terrain, so a winning point may sit behind a ridge. Treat
    the output as a shortlist to check against a topographic map and against
    where you can actually get permission, not as a final answer.
    """
    found: list[tuple[int, float, Site]] = []
    for lat in np.arange(bbox.south, bbox.north, step_deg):
        for lon in np.arange(bbox.west, bbox.east, step_deg):
            seen = visible_from(float(lat), float(lon), cells, radius_km=radius_km)
            if seen.empty:
                continue
            detections = int(seen["detections"].sum())
            # Many grid nodes cover exactly the same cells, so coverage alone
            # leaves large ties and the winner becomes an artefact of iteration
            # order, often on the far edge of the cluster it covers. Breaking
            # ties by mean distance to what is covered picks the middle of the
            # cluster, which is both stable and the better place to stand.
            mean_distance = float(
                (seen["distance_km"] * seen["detections"]).sum() / max(detections, 1)
            )
            found.append(
                (
                    detections,
                    mean_distance,
                    Site(
                        lat=round(float(lat), 4),
                        lon=round(float(lon), 4),
                        detections=detections,
                        cells=len(seen),
                        coverage=detections / total_detections if total_detections else 0.0,
                    ),
                )
            )
    found.sort(key=lambda item: (-item[0], item[1]))
    return [site for _, _, site in found[:top]]


def shortlist_sites(
    cells: pd.DataFrame,
    bbox: BBox,
    radius_km: float = 15.0,
    step_deg: float = 0.02,
    keep: int = 20,
) -> list[Site]:
    """Candidates worth the cost of a visibility check.

    Checking line of sight to every cell from every node of a grid is far too
    expensive to run over the whole search space, so the range-only search picks
    a shortlist first and the terrain pass runs only on those. This mirrors the
    published practice of narrowing candidates before optimizing.
    """
    total = int(cells["detections"].sum()) if not cells.empty else 0
    return search_sites(
        cells, bbox, total, radius_km=radius_km, step_deg=step_deg, top=keep
    )


def rerank_by_visibility(sites: list[Site], cells, visible_detections) -> list[RankedSite]:
    """Re-order a shortlist by what each position can actually see.

    `visible_detections(site, cells) -> int` is injected rather than imported so
    the ordering policy can be tested without a DEM, and so a caller can swap in
    a cheaper or a more careful visibility model.

    Sites that see nothing are kept, not dropped: a candidate that scores well on
    range and sees none of it is the most useful thing this stage can report.
    """
    ranked = [
        RankedSite(
            site=site,
            visible_detections=(visible := int(visible_detections(site, cells))),
            visible_share=visible / site.detections if site.detections else 0.0,
        )
        for site in sites
    ]
    # sort is stable, so ties keep the shortlist order rather than shuffling.
    return sorted(ranked, key=lambda r: r.visible_detections, reverse=True)


def sector_breakdown(seen: pd.DataFrame, weight: str = "detections") -> pd.Series:
    """Weight of fire activity per compass sector, seen from one site.

    This is what tells you whether one fixed camera is enough. Activity packed
    into two adjacent sectors means a single lens will do; activity spread over
    six means a pan-tilt head or several cameras.
    """
    counts = {name: 0.0 for name, _, _ in SECTORS}
    for _, row in seen.iterrows():
        b = row["bearing"] % 360
        for name, lo, hi in SECTORS:
            inside = (lo <= b < hi) if lo < hi else (b >= lo or b < hi)
            if inside:
                counts[name] += float(row[weight])
                break
    return pd.Series(counts)


def recommend_optics(
    seen: pd.DataFrame,
    candidates: tuple[tuple[str, int, float], ...] = (
        ("1080p, 70° lens", 1920, 70.0),
        ("1080p, 45° lens", 1920, 45.0),
        ("4K, 70° lens", 3840, 70.0),
        ("4K, 55° lens", 3840, 55.0),
        ("4K, 35° lens", 3840, 35.0),
    ),
) -> pd.DataFrame:
    """Score sensor and lens combinations against the distances that matter.

    Weighted by detections, so a combination is judged on the fires it would
    actually have caught rather than on the farthest cell in the list.
    """
    if seen.empty:
        return pd.DataFrame()
    total = float(seen["detections"].sum())
    rows = []
    for label, sensor_px, fov in candidates:
        # Bind the loop variables as defaults. A bare closure over them is a
        # latent bug: it works while map() runs eagerly and breaks the moment the
        # call becomes lazy, at which point every row uses the last lens.
        px = seen["distance_km"].map(
            lambda d, px_=sensor_px, fov_=fov: pixels_on_target(float(d), px_, fov_)
        )
        resolved = float(seen.loc[px >= MIN_PIXELS_ON_TARGET, "detections"].sum())
        rows.append(
            {
                "optics": label,
                "useful_range_km": round(max_useful_range_km(sensor_px, fov), 1),
                "detections_resolved": int(resolved),
                "share_resolved": round(resolved / total, 3) if total else 0.0,
            }
        )
    return pd.DataFrame(rows).sort_values(by="share_resolved", ascending=False)


def evaluate_site(
    lat: float,
    lon: float,
    detections: pd.DataFrame,
    radius_km: float = 15.0,
    cell_deg: float = 0.02,
) -> dict:
    """Full report for one specific position, the one you can get permission for.

    This is the entry point that matters in practice: access beats optimality,
    so the usual workflow is to find a spot you can actually use and ask what it
    would have seen.
    """
    cells = rank_cells(detections, cell_deg=cell_deg)
    seen = visible_from(lat, lon, cells, radius_km=radius_km)
    total = int(cells["detections"].sum())
    covered = int(seen["detections"].sum()) if not seen.empty else 0
    return {
        "lat": lat,
        "lon": lon,
        "radius_km": radius_km,
        "cells_in_range": len(seen),
        "detections_in_range": covered,
        "coverage": round(covered / total, 3) if total else 0.0,
        "max_recurrence_years": int(seen["years"].max()) if not seen.empty else 0,
        "sectors": sector_breakdown(seen),
        "optics": recommend_optics(seen),
        "cells": seen,
    }


def bearing_to(lat: float, lon: float, target_lat: float, target_lon: float) -> int:
    """Re-exported so callers can aim at a named landmark without importing more."""
    return bearing_deg(lat, lon, target_lat, target_lon)
