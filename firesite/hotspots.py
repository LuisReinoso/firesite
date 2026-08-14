# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Aggregate fire detections into cells and rank the ground that keeps burning.

The question this answers is not "where were there fires" but "where is it worth
pointing a camera", and those have different answers. A single huge fire produces
hundreds of detections in one season and never returns; a gully that burns every
dry season produces few detections but is the one worth watching.
"""

from __future__ import annotations

import math

import pandas as pd

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def bearing_deg(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> int:
    """Compass bearing from one point to another, 0 = north, clockwise."""
    y = math.radians(to_lon - from_lon) * math.cos(math.radians(to_lat))
    x = math.radians(to_lat - from_lat)
    return round(math.degrees(math.atan2(y, x)) % 360)


def gridify(frame: pd.DataFrame, cell_deg: float = 0.02) -> pd.DataFrame:
    """Snap each detection to a regular cell.

    0.02 degrees is roughly 2.2 km, a few times the 375 m VIIRS footprint: small
    enough to separate neighbouring slopes, large enough that the same fire seen
    on consecutive passes lands in one cell instead of smearing across four.
    """
    out = frame.copy()
    out["cell_lat"] = (out["latitude"] / cell_deg).apply(math.floor) * cell_deg
    out["cell_lon"] = (out["longitude"] / cell_deg).apply(math.floor) * cell_deg
    out["cell"] = (
        out["cell_lat"].round(4).astype(str) + "," + out["cell_lon"].round(4).astype(str)
    )
    return out


def find_persistent_sources(
    frame: pd.DataFrame,
    cell_deg: float = 0.02,
    min_months: int = 10,
    max_mean_frp: float = 5.0,
) -> set[str]:
    """Identify cells that are fixed thermal sources rather than wildfires.

    Brick kilns, greenhouse boilers, quarries, flares and steel mills show up in
    VIIRS all year round at low, steady radiative power. Because the ranking
    rewards recurrence, they take the top places and quietly point the camera at
    a factory.

    The discriminant is seasonality combined with intensity: a wildfire clusters
    in the dry season and releases tens of megawatts, while a fixed source
    appears in nearly every month of the year at a few megawatts. This works in
    either hemisphere because it counts *distinct* months rather than assuming
    which ones are dry.

    Check the excluded cells against a satellite basemap before trusting them
    blindly: an area that genuinely burns year-round would be caught too.
    """
    gridded = gridify(frame, cell_deg)
    profile = gridded.groupby("cell").agg(
        months=("month", "nunique"),
        mean_frp=("frp", "mean"),
    )
    fixed = pd.DataFrame(
        profile[(profile["months"] >= min_months) & (profile["mean_frp"] < max_mean_frp)]
    )
    return set(fixed.index.astype(str))


def drop_persistent_sources(frame: pd.DataFrame, cell_deg: float = 0.02) -> pd.DataFrame:
    """Detections that do not fall on a fixed thermal source."""
    fixed = sorted(find_persistent_sources(frame, cell_deg))
    gridded = gridify(frame, cell_deg)
    return pd.DataFrame(gridded[~gridded["cell"].isin(fixed)])


def rank_cells(
    frame: pd.DataFrame, cell_deg: float = 0.02, top: int | None = None
) -> pd.DataFrame:
    """Rank cells by recurrence first, activity second.

    score = distinct_years * 10 + distinct_days

    The weighting is the whole point. One enormous fire contributes many days but
    a single year; ground that burns most years wins even when each event is
    small. For siting a fixed camera the second is what you want, because the
    camera will be there for a decade.
    """
    gridded = gridify(frame, cell_deg)
    gridded["date"] = gridded["ts_local"].dt.date

    agg = (
        gridded.groupby("cell")
        .agg(
            detections=("cell", "size"),
            days=("date", "nunique"),
            years=("year", "nunique"),
            first_seen=("ts_local", "min"),
            last_seen=("ts_local", "max"),
            max_frp=("frp", "max"),
            mean_frp=("frp", "mean"),
            lat=("latitude", "mean"),
            lon=("longitude", "mean"),
        )
        .reset_index()
    )

    agg["score"] = agg["years"] * 10 + agg["days"]
    agg["mean_frp"] = agg["mean_frp"].round(1)
    agg["first_seen"] = agg["first_seen"].dt.strftime("%Y-%m-%d")
    agg["last_seen"] = agg["last_seen"].dt.strftime("%Y-%m-%d")

    columns = [
        "lat",
        "lon",
        "detections",
        "days",
        "years",
        "max_frp",
        "mean_frp",
        "first_seen",
        "last_seen",
        "score",
    ]
    ranked = pd.DataFrame(agg.sort_values(by="score", ascending=False)[columns])
    ranked = ranked.reset_index(drop=True)
    # `if top` would treat 0 as "no limit" and return everything.
    return ranked if top is None else ranked.head(top)


def temporal_profile(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """When it burns, by year and month.

    Deliberately excludes an hour-of-day breakdown. Polar-orbiting satellites
    pass twice a day, so any hourly histogram of VIIRS data describes the orbit,
    not fire behaviour, and reading it as "fires start at 1pm" is wrong.
    """
    return {
        name: pd.Series(frame.groupby(column).size())
        for name, column in (("by_year", "year"), ("by_month", "month"))
    }


def visible_from(
    lat: float, lon: float, cells: pd.DataFrame, radius_km: float = 15.0
) -> pd.DataFrame:
    """Cells within a camera's useful range, with distance and bearing.

    The default 15 km is a reasonable reach for spotting an incipient plume with
    an ordinary surveillance camera in clean air. It is a planning figure, not a
    guarantee: haze, lens choice and sensor resolution all move it, and
    `pixels_on_target` in siting.py is the check that matters.
    """
    out = cells.copy()
    out["distance_km"] = out.apply(
        lambda r: round(haversine_km(lat, lon, r["lat"], r["lon"]), 1), axis=1
    )
    out["bearing"] = out.apply(lambda r: bearing_deg(lat, lon, r["lat"], r["lon"]), axis=1)
    within = out[out["distance_km"] <= radius_km]
    return pd.DataFrame(within).sort_values(by="score", ascending=False)
