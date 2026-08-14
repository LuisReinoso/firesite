# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Choosing several cameras that complement each other.

The best pair of sites is not the two best individual sites. Two positions that
each see 40% of the fire history may jointly see 70% or 45%, depending entirely
on whether the same ridge blocks both, and ranking them separately cannot tell
the difference.

This is the maximal covering location problem. Picking the optimal subset is
NP-hard, but the objective is submodular and monotone, so the greedy rule of
taking the largest marginal gain each round is guaranteed to land within
1 - 1/e (about 63%) of the optimum. For a handful of cameras out of a shortlist
of candidates that is more than good enough, and it is fast.

Everything here is pure set arithmetic over cell identifiers, deliberately
knowing nothing about coordinates, terrain or satellites.
"""

from __future__ import annotations

from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass

Key = Hashable
CellId = Hashable


@dataclass(frozen=True)
class CoverageChoice:
    """One camera in the chosen set, with what it added when it was picked."""

    key: Key
    added: float  # weight this camera contributed that nothing else had
    cumulative: float  # weight covered by every camera chosen so far


def _weight_of(cells: Iterable[CellId], weights: Mapping[CellId, float]) -> float:
    # Unknown ids contribute nothing rather than raising: one stale identifier
    # should not take down a whole run.
    return float(sum(weights.get(cell, 0.0) for cell in cells))


def joint_coverage(
    keys: Iterable[Key], options: Mapping[Key, set[CellId]], weights: Mapping[CellId, float]
) -> float:
    """Weight covered by a set of cameras together, counting each cell once."""
    covered: set[CellId] = set()
    for key in keys:
        covered |= options.get(key, set())
    return _weight_of(covered, weights)


def greedy_cover(
    options: Mapping[Key, set[CellId]], weights: Mapping[CellId, float], cameras: int
) -> list[CoverageChoice]:
    """Pick cameras one at a time by largest marginal gain.

    `options` maps a candidate to the cells it can see; `weights` gives each cell
    its value, usually detections. Stops early once no candidate adds anything,
    so asking for five cameras where two cover everything returns two.

    Ties go to whichever candidate appears first in `options`. That is not an
    arbitrary rule: candidates arrive already ranked by range, so on equal
    visible coverage the better-placed one wins.
    """
    if cameras <= 0 or not options:
        return []

    covered: set[CellId] = set()
    remaining = dict(options)
    chosen: list[CoverageChoice] = []
    total = 0.0

    for _ in range(cameras):
        best_key: Key | None = None
        best_gain = 0.0
        for key, cells in remaining.items():
            gain = _weight_of(cells - covered, weights)
            if gain > best_gain:
                best_key, best_gain = key, gain
        if best_key is None:
            break  # nothing left adds anything
        covered |= remaining.pop(best_key)
        total += best_gain
        chosen.append(CoverageChoice(key=best_key, added=best_gain, cumulative=total))

    return chosen
