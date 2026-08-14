# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Site search, sector breakdown, optics, and CLI argument handling."""

from __future__ import annotations

import pandas as pd
import pytest

from firesite.cli import _glue_negative_values
from firesite.firms import BBox
from firesite.hotspots import rank_cells
from firesite.siting import (
    MIN_PIXELS_ON_TARGET,
    evaluate_site,
    max_useful_range_km,
    pixels_on_target,
    recommend_optics,
    search_sites,
    sector_breakdown,
)

from .test_ranking import detections


class TestPixelsOnTarget:
    def test_falls_off_with_distance(self):
        near = pixels_on_target(5, 3840, 55)
        far = pixels_on_target(15, 3840, 55)
        assert near > far

    def test_inversely_proportional_to_distance(self):
        assert pixels_on_target(10, 3840, 55) == pytest.approx(
            pixels_on_target(20, 3840, 55) * 2
        )

    def test_wider_lens_gives_fewer_pixels(self):
        assert pixels_on_target(10, 3840, 70) < pixels_on_target(10, 3840, 35)

    def test_more_sensor_pixels_give_more_pixels_on_target(self):
        assert pixels_on_target(10, 3840, 55) > pixels_on_target(10, 1920, 55)

    def test_agrees_with_max_useful_range(self):
        # At exactly the stated maximum range the plume must land on exactly the
        # minimum pixel count, otherwise the two functions disagree and the
        # recommendation table is inconsistent with the range column.
        limit = max_useful_range_km(3840, 55)
        assert pixels_on_target(limit, 3840, 55) == pytest.approx(MIN_PIXELS_ON_TARGET)

    def test_zero_distance_is_not_a_division_error(self):
        assert pixels_on_target(0, 3840, 55) == float("inf")

    def test_known_case_4k_55_at_15km(self):
        # A 30 m plume at 15 km through a 55 degree lens on a 4K sensor.
        assert pixels_on_target(15, 3840, 55) == pytest.approx(8.0, abs=0.3)


class TestRecommendOptics:
    def _seen(self, distances):
        return pd.DataFrame(
            {
                "distance_km": distances,
                "detections": [10] * len(distances),
                "bearing": [90] * len(distances),
                "years": [3] * len(distances),
                "score": [30] * len(distances),
            }
        )

    def test_better_optics_resolve_more(self):
        table = recommend_optics(self._seen([5.0, 10.0, 14.0])).set_index("optics")
        assert (
            table.loc["4K, 55° lens", "share_resolved"]
            >= table.loc["1080p, 70° lens", "share_resolved"]
        )

    def test_share_is_a_fraction(self):
        table = recommend_optics(self._seen([5.0, 10.0, 14.0]))
        assert table["share_resolved"].between(0, 1).all()

    def test_everything_close_is_fully_resolved(self):
        table = recommend_optics(self._seen([1.0, 2.0])).set_index("optics")
        assert table.loc["4K, 55° lens", "share_resolved"] == 1.0

    def test_everything_far_is_unresolved_by_a_wide_cheap_lens(self):
        table = recommend_optics(self._seen([40.0, 50.0])).set_index("optics")
        assert table.loc["1080p, 70° lens", "share_resolved"] == 0.0

    def test_empty_input_gives_empty_table(self):
        assert recommend_optics(pd.DataFrame()).empty


class TestSectorBreakdown:
    def _seen(self, bearings):
        return pd.DataFrame(
            {
                "bearing": bearings,
                "detections": [1] * len(bearings),
            }
        )

    def test_north_wraps_around_zero(self):
        # 350 and 10 degrees are both north; a naive range check puts one in NW.
        result = sector_breakdown(self._seen([350, 10]))
        assert result["N"] == 2

    def test_each_bearing_lands_in_exactly_one_sector(self):
        result = sector_breakdown(self._seen(list(range(0, 360, 7))))
        assert result.sum() == len(range(0, 360, 7))

    @pytest.mark.parametrize(
        "bearing,sector",
        [
            (0, "N"),
            (45, "NE"),
            (90, "E"),
            (135, "SE"),
            (180, "S"),
            (225, "SW"),
            (270, "W"),
            (315, "NW"),
        ],
    )
    def test_cardinal_and_intercardinal_bearings(self, bearing, sector):
        assert sector_breakdown(self._seen([bearing]))[sector] == 1

    def test_bearings_at_or_past_360_are_handled(self):
        assert sector_breakdown(self._seen([360, 361]))["N"] == 2


class TestSearchSites:
    def _cells(self):
        # A tight cluster of activity, plus one isolated cell far away.
        rows = [
            (0.30 + 0.01 * i, -78.22, f"{2020 + i}-08-01", "1200", 20.0) for i in range(4)
        ]
        rows += [(0.90, -78.90, "2024-08-01", "1200", 5.0)]
        return rank_cells(detections(rows))

    def test_picks_a_position_near_the_cluster(self):
        cells = self._cells()
        box = BBox(west=-79.0, south=0.2, east=-78.0, north=1.0)
        sites = search_sites(
            cells, box, int(cells["detections"].sum()), radius_km=15, step_deg=0.05, top=3
        )
        best = sites[0]
        assert 0.25 < best.lat < 0.45
        assert -78.4 < best.lon < -78.0

    def test_results_are_sorted_by_coverage(self):
        cells = self._cells()
        box = BBox(west=-79.0, south=0.2, east=-78.0, north=1.0)
        sites = search_sites(
            cells, box, int(cells["detections"].sum()), radius_km=15, step_deg=0.05, top=5
        )
        assert sites == sorted(sites, key=lambda s: s.detections, reverse=True)

    def test_coverage_never_exceeds_one(self):
        cells = self._cells()
        box = BBox(west=-79.0, south=0.2, east=-78.0, north=1.0)
        sites = search_sites(
            cells, box, int(cells["detections"].sum()), radius_km=50, step_deg=0.1, top=5
        )
        assert all(s.coverage <= 1.0 for s in sites)


class TestEvaluateSite:
    def test_reports_coverage_and_sectors(self):
        rows = [(0.30, -78.22, f"{2020 + i}-08-01", "1200", 20.0) for i in range(3)]
        report = evaluate_site(0.2885, -78.2223, detections(rows), radius_km=15)
        assert report["detections_in_range"] == 3
        assert report["coverage"] == pytest.approx(1.0)
        assert report["max_recurrence_years"] == 3

    def test_site_with_nothing_in_range_is_reported_not_crashed(self):
        rows = [(0.30, -78.22, "2024-08-01", "1200", 20.0)]
        report = evaluate_site(40.0, 10.0, detections(rows), radius_km=15)
        assert report["detections_in_range"] == 0
        assert report["coverage"] == 0.0
        assert report["optics"].empty


class TestNegativeArgumentGlue:
    def test_glues_a_negative_longitude(self):
        assert _glue_negative_values(["evaluate", "f.csv", "--lon", "-78.2"]) == [
            "evaluate",
            "f.csv",
            "--lon=-78.2",
        ]

    def test_glues_a_negative_bbox(self):
        assert _glue_negative_values(["--bbox", "-78.5,0.1,-78.1,0.4"]) == [
            "--bbox=-78.5,0.1,-78.1,0.4"
        ]

    def test_leaves_positive_values_alone(self):
        assert _glue_negative_values(["--lat", "0.28"]) == ["--lat", "0.28"]

    def test_leaves_unrelated_flags_alone(self):
        argv = ["rank", "f.csv", "--top", "5", "--keep-persistent"]
        assert _glue_negative_values(argv) == argv

    def test_does_not_swallow_a_following_flag(self):
        # `--lat --lon 1` is a user error; gluing it into `--lat=--lon` produces a
        # baffling float parse error instead of argparse's clear message.
        argv = ["evaluate", "f.csv", "--lat", "--lon", "1.0"]
        assert _glue_negative_values(argv) == argv

    def test_handles_a_trailing_flag_with_no_value(self):
        assert _glue_negative_values(["evaluate", "--lat"]) == ["evaluate", "--lat"]
