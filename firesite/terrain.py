# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Terrain: does the camera actually see the ground that burns?

Everything else in firesite assumes a clear view and reports what lies within
range. That overstates a site, sometimes badly: on a real Andean site roughly
half the in-range fire history turned out to sit behind ridges, and raising the
mast from 2 m to 20 m recovered two percentage points, because the obstruction is
mountain-scale rather than tree-scale.

Elevation comes from the Copernicus DEM GLO-30, served as cloud-optimized GeoTIFF
from a public AWS bucket that needs no credentials.

The maths lives in `line_of_sight`, a pure function over two arrays. Reading a
DEM needs rasterio, which is an optional extra: `pip install firesite[terrain]`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Copernicus DEM GLO-30 on AWS Open Data. Public, anonymous, no request signing.
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"

# Light bends downward through the atmosphere, so the horizon sits slightly
# further away than geometry alone allows. The 7/6 factor is the standard
# approximation used in radio and surveying practice.
REFRACTION_FACTOR = 7 / 6
EARTH_RADIUS_M = 6_371_000.0
EFFECTIVE_EARTH_RADIUS_M = EARTH_RADIUS_M * REFRACTION_FACTOR

# A camera mounted on a roof, and a plume tall enough to clear the canopy. The
# target height matters: the question is not whether the ground is visible but
# whether a column of smoke rising from it would be.
DEFAULT_OBSERVER_HEIGHT_M = 5.0
DEFAULT_TARGET_HEIGHT_M = 50.0


class TerrainUnavailable(RuntimeError):
    """Raised when rasterio is missing or a DEM tile cannot be read."""


@dataclass(frozen=True)
class LineOfSight:
    visible: bool
    clearance_m: float  # smallest gap between sightline and terrain
    blocking_distance_m: float  # where that smallest gap occurs


def copernicus_tile_name(lat: float, lon: float) -> str:
    """Name of the one-degree tile containing a point.

    Tiles are named by their south-west corner, so the bound is floored rather
    than rounded: rounding puts points just south of a parallel in the tile to
    the north, which then does not contain them.
    """
    lat_floor = math.floor(lat)
    lon_floor = math.floor(lon)
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return (
        f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"
    )


def tile_url(tile_name: str) -> str:
    return f"{DEM_BUCKET}/{tile_name}/{tile_name}.tif"


def line_of_sight(
    distances_m: np.ndarray,
    elevations_m: np.ndarray,
    observer_height_m: float = DEFAULT_OBSERVER_HEIGHT_M,
    target_height_m: float = DEFAULT_TARGET_HEIGHT_M,
    earth_radius_m: float = EFFECTIVE_EARTH_RADIUS_M,
) -> LineOfSight:
    """Whether a target is visible along a sampled terrain profile.

    `distances_m` and `elevations_m` describe the ground from the observer at
    index 0 to the target at index -1. Curvature is applied by dropping terrain
    by d^2 / 2R, which is equivalent to bending the sightline and cheaper.

    Pure: no I/O, no DEM, no global state, which is what makes it testable.
    """
    distances_m = np.asarray(distances_m, dtype=float)
    elevations_m = np.asarray(elevations_m, dtype=float)
    if distances_m.shape != elevations_m.shape:
        raise ValueError("distances and elevations must have the same shape")
    total = float(distances_m[-1])
    if total <= 0:
        raise ValueError("the profile must span a positive distance")

    if math.isinf(earth_radius_m):
        drop = np.zeros_like(distances_m)
    else:
        drop = distances_m**2 / (2.0 * earth_radius_m)
    terrain = elevations_m - drop

    start = float(elevations_m[0]) + observer_height_m
    end = float(elevations_m[-1]) - float(drop[-1]) + target_height_m
    sightline = start + (end - start) * (distances_m / total)

    between = slice(1, -1)
    gaps = sightline[between] - terrain[between]
    if gaps.size == 0:
        # Nothing lies between observer and target.
        return LineOfSight(
            visible=True, clearance_m=float("inf"), blocking_distance_m=float("nan")
        )

    worst = int(np.argmin(gaps))
    clearance = float(gaps[worst])
    return LineOfSight(
        visible=clearance > 0,
        clearance_m=clearance,
        blocking_distance_m=float(distances_m[between][worst]),
    )


def _require_rasterio():
    try:
        import rasterio
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise TerrainUnavailable(
            "Reading a DEM needs rasterio. Install it with "
            "`pip install 'firesite[terrain]'`."
        ) from exc
    return rasterio


def download_tile(lat: float, lon: float, cache_dir: Path) -> Path:  # pragma: no cover
    """Fetch the DEM tile for a point, once, into `cache_dir`."""
    import requests

    name = copernicus_tile_name(lat, lon)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{name}.tif"
    if destination.exists():
        return destination

    response = requests.get(tile_url(name), timeout=600, stream=True)
    if response.status_code == 404:
        raise TerrainUnavailable(
            f"No Copernicus tile for {lat:.4f}, {lon:.4f} ({name}). "
            "Ocean and some polar areas have no coverage."
        )
    response.raise_for_status()
    destination.write_bytes(response.content)
    return destination


def sample_profile(
    dem, origin: tuple[float, float], target: tuple[float, float], samples: int = 256
) -> tuple[np.ndarray, np.ndarray]:  # pragma: no cover
    """Distances and ground elevations along a great-circle-ish straight line.

    Over the tens of kilometres a camera can see, interpolating linearly in
    latitude and longitude is well within the DEM's own error.
    """
    from .hotspots import haversine_km

    lats = np.linspace(origin[0], target[0], samples)
    lons = np.linspace(origin[1], target[1], samples)
    elevations = np.array(
        [v[0] for v in dem.sample(zip(lons, lats, strict=False))], dtype=float
    )
    total_m = haversine_km(*origin, *target) * 1000.0
    return np.linspace(0.0, total_m, samples), elevations


def visible_cells(
    cells,
    origin: tuple[float, float],
    dem_path: Path,
    observer_height_m: float = DEFAULT_OBSERVER_HEIGHT_M,
    target_height_m: float = DEFAULT_TARGET_HEIGHT_M,
    samples: int = 256,
):  # pragma: no cover
    """Add `visible` and `clearance_m` columns to a ranked cell table."""
    rasterio = _require_rasterio()
    import pandas as pd

    out = cells.copy()
    flags: list[bool | None] = []
    clearances: list[float] = []
    with rasterio.open(dem_path) as dem:
        nodata = dem.nodata
        for _, row in out.iterrows():
            distances, elevations = sample_profile(
                dem, origin, (row["lat"], row["lon"]), samples
            )
            outside = np.isnan(elevations)
            if nodata is not None:
                outside |= elevations == nodata
            if outside.any() or distances[-1] <= 0:
                # Off the downloaded tile, or the cell sits on the observer.
                flags.append(None)
                clearances.append(float("nan"))
                continue
            result = line_of_sight(
                distances, elevations, observer_height_m, target_height_m
            )
            flags.append(result.visible)
            clearances.append(result.clearance_m)
    out["visible"] = pd.array(flags, dtype="boolean")
    out["clearance_m"] = np.round(clearances, 1)
    return out
