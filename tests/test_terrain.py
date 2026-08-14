# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Terrain: tile naming and the line-of-sight maths.

The maths is a pure function over two arrays, so it is tested here without any
DEM, any download, or rasterio installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from firesite.terrain import (
    EFFECTIVE_EARTH_RADIUS_M,
    copernicus_tile_name,
    line_of_sight,
    tile_url,
)


class TestTileNaming:
    @pytest.mark.parametrize(
        "lat,lon,expected",
        [
            (0.2885, -78.2223, "Copernicus_DSM_COG_10_N00_00_W079_00_DEM"),
            (0.0, 0.0, "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"),
            (43.7, 7.3, "Copernicus_DSM_COG_10_N43_00_E007_00_DEM"),
            (-33.9, -70.6, "Copernicus_DSM_COG_10_S34_00_W071_00_DEM"),
            (-0.5, -78.5, "Copernicus_DSM_COG_10_S01_00_W079_00_DEM"),
        ],
    )
    def test_names_the_tile_that_contains_the_point(self, lat, lon, expected):
        assert copernicus_tile_name(lat, lon) == expected

    def test_tiles_are_named_by_their_south_west_corner(self):
        # A point just north of the equator and one just south sit in different
        # tiles. Rounding instead of flooring would put both in N00.
        assert copernicus_tile_name(0.01, 10.0) != copernicus_tile_name(-0.01, 10.0)

    def test_url_is_the_public_bucket_and_needs_no_credentials(self):
        url = tile_url(copernicus_tile_name(0.2885, -78.2223))
        assert url.startswith("https://copernicus-dem-30m.s3.amazonaws.com/")
        assert url.endswith(".tif")
        assert "?" not in url  # no signature, no token


def flat(distance_m: float, height: float = 1000.0, n: int = 200):
    d = np.linspace(0, distance_m, n)
    return d, np.full(n, height)


class TestLineOfSight:
    def test_flat_ground_is_visible(self):
        d, z = flat(5000)
        assert line_of_sight(d, z).visible

    def test_a_ridge_taller_than_the_sightline_blocks(self):
        d, z = flat(10000)
        z[100] = 1400  # a 400 m ridge halfway
        result = line_of_sight(d, z, observer_height_m=5, target_height_m=50)
        assert not result.visible
        assert result.clearance_m < 0
        assert 4000 < result.blocking_distance_m < 6000

    def test_a_ridge_below_the_sightline_does_not_block(self):
        d, z = flat(10000)
        z[100] = 1010  # a bump well under the line
        assert line_of_sight(d, z, observer_height_m=5, target_height_m=50).visible

    def test_target_height_can_rescue_a_marginal_case(self):
        # A plume rising above the ground is seen where the ground itself is not.
        d, z = flat(10000)
        z[100] = 1030
        assert not line_of_sight(d, z, observer_height_m=2, target_height_m=0).visible
        assert line_of_sight(d, z, observer_height_m=2, target_height_m=100).visible

    def test_observer_height_can_rescue_a_marginal_case(self):
        d, z = flat(10000)
        z[100] = 1030
        assert not line_of_sight(d, z, observer_height_m=2, target_height_m=0).visible
        assert line_of_sight(d, z, observer_height_m=120, target_height_m=0).visible

    def test_earth_curvature_hides_a_distant_flat_target(self):
        # Over 60 km of perfectly flat ground the bulge is tens of metres, so a
        # low observer cannot see a low target. Ignoring curvature would call
        # this visible and overstate every long-range site.
        d, z = flat(60000)
        assert not line_of_sight(d, z, observer_height_m=2, target_height_m=2).visible

    def test_the_same_case_is_visible_on_a_flat_earth(self):
        d, z = flat(60000)
        result = line_of_sight(
            d, z, observer_height_m=2, target_height_m=2, earth_radius_m=float("inf")
        )
        assert result.visible

    def test_clearance_is_positive_when_visible(self):
        d, z = flat(5000)
        assert line_of_sight(d, z).clearance_m > 0

    def test_a_two_point_profile_is_trivially_visible(self):
        # Nothing lies between the observer and the target.
        assert line_of_sight(np.array([0.0, 1000.0]), np.array([100.0, 100.0])).visible

    def test_rejects_mismatched_arrays(self):
        with pytest.raises(ValueError):
            line_of_sight(np.array([0.0, 1.0]), np.array([1.0]))

    def test_rejects_a_zero_length_profile(self):
        with pytest.raises(ValueError):
            line_of_sight(np.array([0.0, 0.0]), np.array([1.0, 1.0]))

    def test_effective_radius_exceeds_the_geometric_one(self):
        # Standard refraction bends light downward, which lets you see slightly
        # further than geometry alone; the usual 7/6 factor encodes that.
        assert EFFECTIVE_EARTH_RADIUS_M > 6_371_000
