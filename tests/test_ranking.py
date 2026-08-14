# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Ranking, the persistent-source filter, and schema normalization."""

from __future__ import annotations

import pandas as pd

from firesite.firms import normalize
from firesite.hotspots import (
    drop_persistent_sources,
    find_persistent_sources,
    rank_cells,
    temporal_profile,
    visible_from,
)


def detections(rows) -> pd.DataFrame:
    """Build a normalized frame from (lat, lon, 'YYYY-MM-DD', 'HHMM', frp)."""
    raw = pd.DataFrame(
        {
            "latitude": [r[0] for r in rows],
            "longitude": [r[1] for r in rows],
            "acq_date": [r[2] for r in rows],
            "acq_time": [r[3] for r in rows],
            "frp": [r[4] for r in rows],
            "confidence": ["h"] * len(rows),
        }
    )
    return normalize(raw)


class TestNormalize:
    def test_letter_confidence_maps_to_boolean(self):
        raw = pd.DataFrame(
            {
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [0.0, 0.0, 0.0],
                "acq_date": ["2024-08-01"] * 3,
                "acq_time": ["1200"] * 3,
                "frp": [10.0] * 3,
                "confidence": ["h", "n", "l"],
            }
        )
        assert normalize(raw)["high_confidence"].tolist() == [True, True, False]

    def test_string_dtype_confidence_maps_to_boolean(self):
        # pandas 3 stores text as a str dtype rather than object, so a check of
        # `dtype == object` silently routes letters through the numeric branch
        # and marks every detection low confidence. Caught by CI, not locally.
        raw = pd.DataFrame(
            {
                "latitude": [0.0, 0.0, 0.0],
                "longitude": [0.0, 0.0, 0.0],
                "acq_date": ["2024-08-01"] * 3,
                "acq_time": ["1200"] * 3,
                "frp": [10.0] * 3,
                "confidence": pd.array(["h", "n", "l"], dtype="string"),
            }
        )
        assert normalize(raw)["high_confidence"].tolist() == [True, True, False]

    def test_numeric_confidence_maps_to_boolean(self):
        raw = pd.DataFrame(
            {
                "latitude": [0.0, 0.0],
                "longitude": [0.0, 0.0],
                "acq_date": ["2024-08-01"] * 2,
                "acq_time": ["1200"] * 2,
                "frp": [10.0] * 2,
                "confidence": [80, 20],
            }
        )
        assert normalize(raw)["high_confidence"].tolist() == [True, False]

    def test_short_acq_time_is_zero_padded(self):
        raw = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "acq_date": ["2024-08-01"],
                "acq_time": [5],
                "frp": [10.0],
                "confidence": ["h"],
            }
        )
        assert normalize(raw)["ts"].iloc[0].hour == 0

    def test_timezone_shifts_local_fields(self):
        raw = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "acq_date": ["2024-08-01"],
                "acq_time": ["0200"],
                "frp": [10.0],
                "confidence": ["h"],
            }
        )
        out = normalize(raw, timezone="America/Guayaquil")  # UTC-5
        assert out["ts"].iloc[0].hour == 2
        assert out["hour_local"].iloc[0] == 21  # previous day, 21:00 local
        assert out["ts_local"].iloc[0].day == 31

    def test_does_not_mutate_input(self):
        raw = pd.DataFrame(
            {
                "latitude": [0.0],
                "longitude": [0.0],
                "acq_date": ["2024-08-01"],
                "acq_time": ["1200"],
                "frp": [10.0],
                "confidence": ["h"],
            }
        )
        before = raw.columns.tolist()
        normalize(raw)
        assert raw.columns.tolist() == before


class TestRankCells:
    def test_recurrence_outranks_raw_count(self):
        # One cell: a single huge fire, 6 detections over 3 days of one year.
        # Another: 3 detections spread across 3 different years.
        rows = [(0.10, -78.10, f"2024-08-0{d}", "1200", 100.0) for d in (1, 1, 2, 2, 3, 3)]
        rows += [(0.50, -78.50, f"{y}-08-01", "1200", 5.0) for y in (2022, 2023, 2024)]
        ranked = rank_cells(detections(rows))
        top = ranked.iloc[0]
        assert top["years"] == 3
        assert top["detections"] == 3  # fewer detections, still ranked first

    def test_score_formula(self):
        rows = [
            (0.5, -78.5, f"{y}-08-0{d}", "1200", 5.0) for y in (2022, 2023) for d in (1, 2)
        ]
        ranked = rank_cells(detections(rows))
        row = ranked.iloc[0]
        assert row["score"] == row["years"] * 10 + row["days"]

    def test_same_day_repeats_count_as_one_day(self):
        rows = [(0.5, -78.5, "2024-08-01", t, 5.0) for t in ("0130", "1330", "1400")]
        assert rank_cells(detections(rows)).iloc[0]["days"] == 1

    def test_top_limits_rows(self):
        rows = [(0.1 * i, -78.0 - 0.1 * i, "2024-08-01", "1200", 5.0) for i in range(6)]
        assert len(rank_cells(detections(rows), top=3)) == 3

    def test_top_zero_returns_nothing(self):
        rows = [(0.1 * i, -78.0 - 0.1 * i, "2024-08-01", "1200", 5.0) for i in range(6)]
        assert len(rank_cells(detections(rows), top=0)) == 0

    def test_empty_input_gives_empty_output(self):
        empty = detections([]).astype({"frp": float})
        assert rank_cells(empty).empty


class TestPersistentSources:
    def _kiln(self, lat=0.5, lon=-78.5):
        # Low power, present in all twelve months across two years: a fixed source.
        return [
            (lat, lon, f"{y}-{m:02d}-15", "1200", 1.5)
            for y in (2023, 2024)
            for m in range(1, 13)
        ]

    def _wildfire(self, lat=0.1, lon=-78.1):
        # High power, dry season only, several years: a real fire.
        return [
            (lat, lon, f"{y}-{m:02d}-15", "1200", 60.0)
            for y in (2022, 2023, 2024)
            for m in (8, 9)
        ]

    def test_flags_the_kiln(self):
        found = find_persistent_sources(detections(self._kiln() + self._wildfire()))
        assert len(found) == 1

    def test_does_not_flag_the_wildfire(self):
        frame = detections(self._kiln() + self._wildfire())
        kept = drop_persistent_sources(frame)
        assert len(kept) == len(self._wildfire())
        assert kept["frp"].min() > 50

    def test_works_in_the_southern_hemisphere(self):
        # Same shapes, mirrored season. The filter must not assume which months
        # are dry, only that a fixed source spans most of the year.
        kiln = [
            (-33.5, -70.5, f"{y}-{m:02d}-15", "1200", 1.5)
            for y in (2023, 2024)
            for m in range(1, 13)
        ]
        fire = [
            (-33.1, -70.1, f"{y}-{m:02d}-15", "1200", 60.0)
            for y in (2022, 2023, 2024)
            for m in (1, 2)
        ]
        kept = drop_persistent_sources(detections(kiln + fire))
        assert len(kept) == len(fire)

    def test_high_power_year_round_is_kept(self):
        # Burning all year at high power is unusual but real; only the low-power
        # combination should be excluded.
        hot = [
            (0.5, -78.5, f"{y}-{m:02d}-15", "1200", 80.0)
            for y in (2023, 2024)
            for m in range(1, 13)
        ]
        assert find_persistent_sources(detections(hot)) == set()


class TestTemporalProfile:
    def test_has_year_and_month_only(self):
        rows = [(0.5, -78.5, "2024-08-01", "1200", 5.0)]
        profile = temporal_profile(detections(rows))
        assert set(profile) == {"by_year", "by_month"}

    def test_no_hourly_breakdown_is_offered(self):
        # Polar orbiters pass twice a day; an hourly histogram would describe the
        # orbit, not the fires. Offering it at all invites the wrong conclusion.
        profile = temporal_profile(detections([(0.5, -78.5, "2024-08-01", "1200", 5.0)]))
        assert not any("hour" in key for key in profile)


class TestVisibleFrom:
    def _cells(self):
        rows = [
            (0.30, -78.22, "2024-08-01", "1200", 5.0),  # ~2 km away
            (0.50, -78.22, "2024-08-01", "1200", 5.0),
        ]  # ~24 km away
        return rank_cells(detections(rows))

    def test_excludes_cells_beyond_the_radius(self):
        seen = visible_from(0.2885, -78.2223, self._cells(), radius_km=15)
        assert len(seen) == 1

    def test_reports_distance_and_bearing(self):
        seen = visible_from(0.2885, -78.2223, self._cells(), radius_km=15)
        assert seen.iloc[0]["distance_km"] < 15
        assert 0 <= seen.iloc[0]["bearing"] < 360

    def test_empty_when_nothing_is_in_range(self):
        assert visible_from(10.0, 10.0, self._cells(), radius_km=15).empty

    def test_does_not_mutate_the_cells_frame(self):
        cells = self._cells()
        before = cells.columns.tolist()
        visible_from(0.2885, -78.2223, cells, radius_km=15)
        assert cells.columns.tolist() == before
