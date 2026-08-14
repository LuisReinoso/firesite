# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Two-stage siting: shortlist on range, then re-rank on what is really visible.

The re-ranking takes a visibility function as an argument, so the whole policy is
testable with a stub and never needs a DEM.
"""

from __future__ import annotations

import pytest

from firesite.firms import BBox
from firesite.hotspots import rank_cells
from firesite.siting import Site, rerank_by_visibility, shortlist_sites

from .test_ranking import detections


def cells_with_two_clusters():
    """A west cluster and an east cluster, both about the same size."""
    rows = []
    for i in range(4):
        rows += [(0.30, -78.40 + 0.001 * i, f"{2020 + i}-08-01", "1200", 20.0)]
    for i in range(4):
        rows += [(0.30, -78.10 + 0.001 * i, f"{2020 + i}-08-01", "1200", 20.0)]
    return rank_cells(detections(rows))


class TestShortlist:
    def test_returns_at_most_the_requested_number(self):
        cells = cells_with_two_clusters()
        box = BBox(west=-78.5, south=0.2, east=-78.0, north=0.4)
        sites = shortlist_sites(cells, box, radius_km=20, step_deg=0.05, keep=3)
        assert len(sites) <= 3

    def test_is_ordered_by_naive_coverage(self):
        cells = cells_with_two_clusters()
        box = BBox(west=-78.5, south=0.2, east=-78.0, north=0.4)
        sites = shortlist_sites(cells, box, radius_km=20, step_deg=0.05, keep=5)
        assert [s.detections for s in sites] == sorted(
            (s.detections for s in sites), reverse=True
        )

    def test_empty_when_nothing_is_in_range_anywhere(self):
        cells = cells_with_two_clusters()
        box = BBox(west=10.0, south=10.0, east=10.5, north=10.5)
        assert shortlist_sites(cells, box, radius_km=5, step_deg=0.1, keep=5) == []


class TestRerankByVisibility:
    def _sites(self):
        return [
            Site(lat=0.30, lon=-78.40, detections=100, cells=4, coverage=0.5),
            Site(lat=0.30, lon=-78.10, detections=90, cells=4, coverage=0.45),
        ]

    def test_a_blocked_leader_loses_to_a_clear_runner_up(self):
        # The whole point of the second stage: the naive winner sits behind a
        # ridge and the runner-up, which sees everything, should take first place.
        def visibility(site, _cells):
            return 10 if site.lon == -78.40 else 90

        ranked = rerank_by_visibility(self._sites(), None, visibility)
        assert ranked[0].site.lon == -78.10
        assert ranked[0].visible_detections == 90

    def test_order_is_kept_when_nothing_is_blocked(self):
        ranked = rerank_by_visibility(
            self._sites(), None, lambda site, _cells: site.detections
        )
        assert ranked[0].site.lon == -78.40

    def test_reports_the_fraction_that_survives(self):
        ranked = rerank_by_visibility(
            self._sites(), None, lambda site, _cells: site.detections // 2
        )
        assert ranked[0].visible_share == pytest.approx(0.5)

    def test_a_site_that_sees_nothing_is_reported_not_dropped(self):
        # Dropping it would hide the finding. A site with zero visibility is the
        # most useful thing the second stage can tell you.
        ranked = rerank_by_visibility(self._sites(), None, lambda site, _cells: 0)
        assert len(ranked) == 2
        assert all(r.visible_detections == 0 for r in ranked)
        assert all(r.visible_share == 0.0 for r in ranked)

    def test_a_site_with_no_detections_does_not_divide_by_zero(self):
        empty = [Site(lat=0.0, lon=0.0, detections=0, cells=0, coverage=0.0)]
        ranked = rerank_by_visibility(empty, None, lambda site, _cells: 0)
        assert ranked[0].visible_share == 0.0

    def test_ties_keep_the_shortlist_order(self):
        ranked = rerank_by_visibility(self._sites(), None, lambda site, _cells: 50)
        assert ranked[0].site.lon == -78.40
