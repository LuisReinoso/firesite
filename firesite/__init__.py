# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""firesite: decide where to put a wildfire detection camera.

Uses NASA FIRMS satellite fire history to answer two questions that usually get
answered by intuition and later regretted: which ground actually keeps burning,
and what a camera at a given spot would have seen.
"""

from . import firms, hotspots, plot, siting
from .firms import BBox
from .hotspots import rank_cells, temporal_profile, visible_from
from .siting import evaluate_site, max_useful_range_km, pixels_on_target, search_sites

__version__ = "0.1.0"

__all__ = [
    "BBox",
    "__version__",
    "evaluate_site",
    "firms",
    "hotspots",
    "max_useful_range_km",
    "pixels_on_target",
    "plot",
    "rank_cells",
    "search_sites",
    "siting",
    "temporal_profile",
    "visible_from",
]
