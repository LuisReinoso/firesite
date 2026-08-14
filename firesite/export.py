# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Export an analysis to JSON for the web viewer.

Kept deliberately small and pure: it takes frames and returns a plain dict, so
the viewer's contract is testable without a browser and without writing a file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .hotspots import rank_cells, temporal_profile
from .siting import evaluate_site

SCHEMA_VERSION = 1


def _round_cells(cells: pd.DataFrame, precision: int = 4) -> list[dict]:
    """Trim coordinates to something a browser can hold comfortably.

    Four decimals is about 11 m, far finer than the 375 m footprint the numbers
    came from, and it roughly halves the payload.
    """
    trimmed = cells.copy()
    for column in ("lat", "lon"):
        trimmed[column] = trimmed[column].round(precision)
    keep = ["lat", "lon", "detections", "days", "years", "max_frp", "last_seen"]
    return trimmed[keep].to_dict(orient="records")


def build_payload(
    detections: pd.DataFrame,
    site: tuple[float, float] | None = None,
    radius_km: float = 15.0,
    cell_deg: float = 0.02,
    title: str = "Fire recurrence",
    max_cells: int = 4000,
) -> dict:
    """Everything the viewer needs, as plain JSON-serializable data."""
    cells = rank_cells(detections, cell_deg=cell_deg)
    truncated = len(cells) > max_cells
    profile = temporal_profile(detections)

    payload: dict = {
        "schema": SCHEMA_VERSION,
        "title": title,
        "generated_from": {
            "detections": len(detections),
            "first": str(detections["ts_local"].min())[:10] if len(detections) else None,
            "last": str(detections["ts_local"].max())[:10] if len(detections) else None,
            "cell_deg": cell_deg,
        },
        "cells": _round_cells(cells.head(max_cells)),
        "cells_truncated": truncated,
        "by_year": {int(k): int(v) for k, v in profile["by_year"].items()},
        "by_month": {int(k): int(v) for k, v in profile["by_month"].items()},
        "site": None,
    }

    if site is not None:
        lat, lon = site
        report = evaluate_site(lat, lon, detections, radius_km=radius_km, cell_deg=cell_deg)
        optics = report["optics"]
        payload["site"] = {
            "lat": lat,
            "lon": lon,
            "radius_km": radius_km,
            "cells_in_range": report["cells_in_range"],
            "detections_in_range": report["detections_in_range"],
            "coverage": report["coverage"],
            "max_recurrence_years": report["max_recurrence_years"],
            "sectors": {k: float(v) for k, v in report["sectors"].items()},
            "optics": optics.to_dict(orient="records") if not optics.empty else [],
        }
    return payload


def write_payload(payload: dict, output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False turns a stray NaN into a loud error here rather than into
    # invalid JSON that only fails once it reaches the browser.
    output.write_text(json.dumps(_clean(payload), allow_nan=False), encoding="utf-8")
    return output


def _clean(value):
    """Replace NaN and infinity, which JSON cannot represent, with null."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value
