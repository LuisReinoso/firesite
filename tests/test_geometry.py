# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Pure geometry: distances, bearings, boxes, grid cells."""

from __future__ import annotations

import pandas as pd
import pytest

from firesite.firms import BBox
from firesite.hotspots import bearing_deg, gridify, haversine_km


class TestHaversine:
    def test_zero_distance(self):
        assert haversine_km(0.0, 0.0, 0.0, 0.0) == pytest.approx(0.0)

    def test_one_degree_of_latitude_is_about_111_km(self):
        assert haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(111.19, abs=0.1)

    def test_symmetric(self):
        a = haversine_km(0.29, -78.22, 0.36, -78.35)
        b = haversine_km(0.36, -78.35, 0.29, -78.22)
        assert a == pytest.approx(b)

    def test_known_pair_quito_to_cotacachi(self):
        # Quito centre to Cotacachi volcano, roughly 75 km.
        d = haversine_km(-0.2200, -78.5000, 0.3581, -78.3511)
        assert 60 < d < 90

    def test_antimeridian_is_short_not_long(self):
        # Two points either side of 180 are close, not most of the way round.
        assert haversine_km(0.0, 179.9, 0.0, -179.9) == pytest.approx(22.24, abs=0.5)


class TestBearing:
    @pytest.mark.parametrize(
        "dlat,dlon,expected",
        [
            (1.0, 0.0, 0),  # north
            (0.0, 1.0, 90),  # east
            (-1.0, 0.0, 180),  # south
            (0.0, -1.0, 270),  # west
        ],
    )
    def test_cardinals(self, dlat, dlon, expected):
        assert bearing_deg(0.0, 0.0, dlat, dlon) == expected

    def test_always_in_range(self):
        for dlat in (-1.0, -0.5, 0.0, 0.5, 1.0):
            for dlon in (-1.0, -0.5, 0.0, 0.5, 1.0):
                assert 0 <= bearing_deg(0.0, 0.0, dlat, dlon) < 360

    def test_opposite_directions_differ_by_180(self):
        there = bearing_deg(0.0, 0.0, 1.0, 1.0)
        back = bearing_deg(1.0, 1.0, 0.0, 0.0)
        assert abs((there - back) % 360) == pytest.approx(180, abs=1)


class TestBBox:
    def test_parse_negative_coordinates(self):
        box = BBox.parse("-78.55,0.15,-78.10,0.45")
        assert box.west == -78.55
        assert box.north == 0.45

    def test_parse_rejects_wrong_arity(self):
        with pytest.raises(ValueError):
            BBox.parse("-78.55,0.15,-78.10")

    def test_rejects_inverted_box(self):
        with pytest.raises(ValueError):
            BBox(west=10.0, south=0.0, east=-10.0, north=1.0)

    def test_rejects_out_of_range_latitude(self):
        with pytest.raises(ValueError):
            BBox(west=-1.0, south=-91.0, east=1.0, north=1.0)

    def test_around_contains_its_centre(self):
        box = BBox.around(0.2885, -78.2223, 40)
        inside = box.contains(pd.Series([0.2885]), pd.Series([-78.2223]))
        assert bool(inside.iloc[0])

    def test_around_reaches_the_requested_radius(self):
        lat, lon, radius = 0.2885, -78.2223, 40.0
        box = BBox.around(lat, lon, radius)
        assert haversine_km(lat, lon, box.north, lon) == pytest.approx(radius, rel=0.02)
        assert haversine_km(lat, lon, lat, box.east) == pytest.approx(radius, rel=0.02)

    def test_around_survives_the_poles(self):
        # cos(latitude) goes to zero at the pole; this must not divide by zero
        # nor produce longitudes outside [-180, 180].
        box = BBox.around(89.9, 0.0, 50)
        assert -180 <= box.west < box.east <= 180

    def test_area_is_positive_and_sane(self):
        box = BBox(west=-78.5, south=0.0, east=-78.0, north=0.5)
        # half a degree square near the equator is about 55 x 55 km
        assert 2500 < box.area_km2() < 3500

    def test_as_api_order(self):
        box = BBox(west=-1.0, south=-2.0, east=3.0, north=4.0)
        assert box.as_api() == "-1.0,-2.0,3.0,4.0"


class TestGridify:
    def _frame(self, points):
        return pd.DataFrame(
            {"latitude": [p[0] for p in points], "longitude": [p[1] for p in points]}
        )

    def test_nearby_points_share_a_cell(self):
        out = gridify(self._frame([(0.2885, -78.2223), (0.2891, -78.2229)]), 0.02)
        assert out["cell"].nunique() == 1

    def test_distant_points_do_not(self):
        out = gridify(self._frame([(0.2885, -78.2223), (0.3885, -78.3223)]), 0.02)
        assert out["cell"].nunique() == 2

    def test_negative_coordinates_floor_downward(self):
        # -0.201 must land in the cell starting at -0.22, not -0.20: rounding
        # toward zero here would shift every southern-hemisphere cell.
        out = gridify(self._frame([(-0.201, -78.201)]), 0.02)
        assert out["cell_lat"].iloc[0] == pytest.approx(-0.22)
        assert out["cell_lon"].iloc[0] == pytest.approx(-78.22)

    def test_cell_label_is_stable_across_calls(self):
        frame = self._frame([(0.2885, -78.2223)])
        assert gridify(frame, 0.02)["cell"].iloc[0] == gridify(frame, 0.02)["cell"].iloc[0]

    def test_does_not_mutate_input(self):
        frame = self._frame([(0.2885, -78.2223)])
        before = frame.columns.tolist()
        gridify(frame, 0.02)
        assert frame.columns.tolist() == before
