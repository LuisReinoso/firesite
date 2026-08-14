# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Download active-fire detections from NASA FIRMS.

FIRMS exposes two access paths:

1. Public rolling CSVs (last 24h/48h/7d, per continent). MODIS C6.1 at 1 km only.
   No credentials.
2. The area API, which needs a free MAP_KEY. Gives VIIRS at 375 m plus the
   archive back to 2012.

For siting a camera, VIIRS is the one that matters: 375 m resolves the small
Andean or Mediterranean fires that MODIS at 1 km averages away.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

FIRMS_ROOT = "https://firms.modaps.eosdis.nasa.gov"
AREA_API = f"{FIRMS_ROOT}/api/area/csv"
AVAILABILITY_API = f"{FIRMS_ROOT}/api/data_availability/csv"

# Maximum window the area endpoint accepts. Asking for more returns HTTP 400
# with "Invalid day range. Expects [1..5]".
MAX_DAYS = 5

# Sustainable pace. Without a pause the quota (5000 transactions per 10 minutes,
# and a wide bounding box counts as several) runs out around request 480 and the
# server then rejects everything until the window rolls over.
REQUEST_PAUSE_S = 1.0

# Once the quota is gone, the useful wait is minutes, not seconds.
QUOTA_BACKOFF_S = (60, 180, 420, 660)

# Consolidated archive (SP) and near-real-time (NRT) sources. Availability
# windows shift over time; query `availability()` rather than hardcoding dates.
ARCHIVE_SOURCES = ("VIIRS_SNPP_SP", "VIIRS_NOAA20_SP", "MODIS_SP")
NRT_SOURCES = ("VIIRS_SNPP_NRT", "VIIRS_NOAA20_NRT", "VIIRS_NOAA21_NRT", "MODIS_NRT")


class FirmsError(RuntimeError):
    """Raised when FIRMS rejects a request in a way retrying will not fix."""


@dataclass(frozen=True)
class BBox:
    """Geographic bounding box in decimal degrees."""

    west: float
    south: float
    east: float
    north: float

    def __post_init__(self) -> None:
        if not -180 <= self.west < self.east <= 180:
            raise ValueError(f"bad longitudes: west={self.west} east={self.east}")
        if not -90 <= self.south < self.north <= 90:
            raise ValueError(f"bad latitudes: south={self.south} north={self.north}")

    @classmethod
    def parse(cls, text: str) -> BBox:
        """Parse 'west,south,east,north'."""
        parts = [float(p) for p in text.split(",")]
        if len(parts) != 4:
            raise ValueError("bbox must be 'west,south,east,north'")
        return cls(*parts)

    @classmethod
    def around(cls, lat: float, lon: float, radius_km: float) -> BBox:
        """Square box centred on a point, sized so the radius fits inside.

        Near the poles cos(latitude) collapses and the longitude span explodes
        past a full turn, so it is clamped to the whole world rather than
        allowed to produce an invalid box.
        """
        import math

        dlat = radius_km / 110.574
        cos_lat = math.cos(math.radians(lat))
        dlon = 180.0 if cos_lat < 1e-6 else radius_km / (111.320 * cos_lat)
        return cls(
            west=max(-180.0, lon - dlon),
            south=max(-90.0, lat - dlat),
            east=min(180.0, lon + dlon),
            north=min(90.0, lat + dlat),
        )

    def as_api(self) -> str:
        """Order the /api/area/ endpoint expects."""
        return f"{self.west},{self.south},{self.east},{self.north}"

    def contains(self, lat: pd.Series, lon: pd.Series) -> pd.Series:
        return lat.between(self.south, self.north) & lon.between(self.west, self.east)

    def area_km2(self) -> float:
        import math

        mid = math.radians((self.north + self.south) / 2)
        return (
            (self.east - self.west)
            * 111.320
            * math.cos(mid)
            * (self.north - self.south)
            * 110.574
        )


def _map_key(explicit: str | None = None) -> str:
    key = explicit or os.environ.get("FIRMS_MAP_KEY")
    if not key:
        raise FirmsError(
            "No FIRMS MAP_KEY. Get one free at "
            "https://firms.modaps.eosdis.nasa.gov/api/map_key/ and export it as "
            "FIRMS_MAP_KEY."
        )
    return key


def _get(url: str, timeout: int = 120) -> str:
    """GET with quota-aware retries.

    Two failures look almost identical over the wire: an exhausted quota and a
    malformed request both come back as HTTP 400. Retrying helps the first and
    never helps the second, so they are told apart by the response body.
    """
    last: Exception | None = None
    for attempt, wait in enumerate(QUOTA_BACKOFF_S):
        try:
            resp = requests.get(url, timeout=timeout)
            body = resp.text.strip()
            if resp.status_code in (400, 429, 500, 502, 503, 504):
                lowered = body.lower()
                if "invalid day range" in lowered or "invalid api call" in lowered:
                    raise FirmsError(f"FIRMS rejected the request: {body[:160]}")
                raise requests.HTTPError(f"HTTP {resp.status_code}: {body[:120]}")
            resp.raise_for_status()
            if body.lower().startswith(("invalid", "error", "you have exceeded")):
                raise FirmsError(f"FIRMS rejected the request: {body[:160]}")
            return resp.text
        except FirmsError:
            raise
        except Exception as exc:  # transient: network blip or spent quota
            last = exc
            if attempt < len(QUOTA_BACKOFF_S) - 1:
                print(f"    quota or transient failure, waiting {wait}s...", flush=True)
                time.sleep(wait)
    raise FirmsError(f"FIRMS failed after {len(QUOTA_BACKOFF_S)} attempts: {last}")


def availability(map_key: str | None = None) -> pd.DataFrame:
    """Date range each FIRMS source currently covers."""
    text = _get(f"{AVAILABILITY_API}/{_map_key(map_key)}/all")
    return pd.read_csv(io.StringIO(text))


def fetch_window(
    source: str,
    bbox: BBox,
    days: int = MAX_DAYS,
    start: str | None = None,
    map_key: str | None = None,
) -> pd.DataFrame:
    """One area query. `days` is 1..5; `start` is YYYY-MM-DD, counting forward."""
    if not 1 <= days <= MAX_DAYS:
        raise ValueError(f"FIRMS accepts 1..{MAX_DAYS} days per request")
    url = f"{AREA_API}/{_map_key(map_key)}/{source}/{bbox.as_api()}/{days}"
    if start:
        url = f"{url}/{start}"
    frame = pd.read_csv(io.StringIO(_get(url)))
    frame["source"] = source
    return frame


def fetch_range(
    source: str,
    bbox: BBox,
    start: str,
    end: str,
    map_key: str | None = None,
    cache_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Chain 5-day windows to cover an arbitrary span.

    A multi-year series is thousands of requests, so every window is cached on
    disk. Re-running resumes where it stopped instead of starting over, which
    matters because the quota will interrupt long pulls.
    """
    if cache_dir is None:
        cache_dir = Path("firms_cache") / source
    cache_dir.mkdir(parents=True, exist_ok=True)

    windows = list(pd.date_range(start, end, freq=f"{MAX_DAYS}D"))
    chunks: list[pd.DataFrame] = []
    for i, window_start in enumerate(windows, 1):
        remaining = (pd.Timestamp(end) - window_start).days + 1
        days = min(MAX_DAYS, remaining)
        if days < 1:
            break
        label = window_start.strftime("%Y-%m-%d")
        cached = cache_dir / f"{label}_{days}d.csv"
        if cached.exists():
            chunk = pd.read_csv(cached)
        else:
            chunk = fetch_window(source, bbox, days=days, start=label, map_key=map_key)
            chunk.to_csv(cached, index=False)
            time.sleep(REQUEST_PAUSE_S)
        if not chunk.empty:
            chunks.append(chunk)
        if verbose and (i % 20 == 0 or i == len(windows)):
            total = sum(len(c) for c in chunks)
            print(
                f"    {source} {i}/{len(windows)} windows | {total} detections", flush=True
            )
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True).drop_duplicates()


def normalize(frame: pd.DataFrame, timezone: str = "UTC") -> pd.DataFrame:
    """Unify the MODIS and VIIRS schemas and localize timestamps.

    Both carry latitude/longitude/acq_date/acq_time/confidence/frp, but MODIS
    reports confidence as 0-100 and VIIRS as l/n/h.
    """
    out = frame.copy()
    out["acq_time"] = out["acq_time"].astype(str).str.zfill(4)
    out["ts"] = pd.to_datetime(
        out["acq_date"].astype(str) + " " + out["acq_time"],
        format="%Y-%m-%d %H%M",
        utc=True,
    )
    local = out["ts"].dt.tz_convert(timezone)
    out["ts_local"] = local
    out["hour_local"] = local.dt.hour
    out["month"] = local.dt.month
    out["year"] = local.dt.year

    # Branch on whether the column is numeric, not on `dtype == object`: pandas 3
    # stores text in a str dtype, so the object check sends l/n/h down the
    # numeric path, where the letters coerce to NaN and every detection is
    # silently marked low confidence.
    confidence = out["confidence"]
    if pd.api.types.is_numeric_dtype(confidence):
        flag = pd.Series(pd.to_numeric(confidence, errors="coerce")).ge(50)
    else:
        text = confidence.astype("string").str.strip().str.lower()
        flag = text.isin(["h", "high", "n", "nominal"])
    # A missing confidence is not a high one, and leaving NA in place makes the
    # column unusable as a boolean mask.
    out["high_confidence"] = flag.fillna(False).astype(bool)
    return out
