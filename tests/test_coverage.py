# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Choosing several cameras that complement each other.

Pure set arithmetic over cell identifiers and weights, so none of this needs a
DEM, a network, or a coordinate.
"""

from __future__ import annotations

import pytest

from firesite.coverage import CoverageChoice, greedy_cover, joint_coverage


def weights(**pairs) -> dict[str, float]:
    return dict(pairs)


class TestGreedyCover:
    def test_picks_the_biggest_first(self):
        options = {"a": {"1", "2", "3"}, "b": {"4"}}
        w = weights(**{str(i): 1.0 for i in range(1, 5)})
        chosen = greedy_cover(options, w, cameras=1)
        assert [c.key for c in chosen] == ["a"]

    def test_second_pick_maximises_marginal_gain_not_size(self):
        # `b` is larger than `c` but overlaps `a` completely, so it adds nothing.
        # A greedy that sorted by absolute size would pick it and cover less.
        options = {
            "a": {"1", "2", "3"},
            "b": {"1", "2", "3"},
            "c": {"4", "5"},
        }
        w = weights(**{str(i): 1.0 for i in range(1, 6)})
        chosen = greedy_cover(options, w, cameras=2)
        assert [c.key for c in chosen] == ["a", "c"]
        assert chosen[1].added == pytest.approx(2.0)

    def test_reports_the_marginal_gain_of_each_pick(self):
        options = {"a": {"1", "2"}, "b": {"2", "3"}}
        w = weights(**{"1": 1.0, "2": 1.0, "3": 1.0})
        chosen = greedy_cover(options, w, cameras=2)
        assert chosen[0].added == pytest.approx(2.0)
        assert chosen[1].added == pytest.approx(1.0)  # cell 2 was already covered

    def test_running_total_never_decreases(self):
        options = {"a": {"1", "2"}, "b": {"3"}, "c": {"4"}}
        w = weights(**{str(i): 1.0 for i in range(1, 5)})
        chosen = greedy_cover(options, w, cameras=3)
        totals = [c.cumulative for c in chosen]
        assert totals == sorted(totals)
        assert totals[-1] == pytest.approx(4.0)

    def test_weights_beat_counts(self):
        # One heavy cell outweighs three light ones; greedy must follow weight.
        options = {"a": {"heavy"}, "b": {"l1", "l2", "l3"}}
        w = weights(heavy=100.0, l1=1.0, l2=1.0, l3=1.0)
        assert greedy_cover(options, w, cameras=1)[0].key == "a"

    def test_stops_when_nothing_new_can_be_added(self):
        # Asking for three cameras when two cover everything returns two.
        options = {"a": {"1"}, "b": {"2"}, "c": {"1", "2"}}
        w = weights(**{"1": 1.0, "2": 1.0})
        chosen = greedy_cover(options, w, cameras=3)
        assert len(chosen) <= 2
        assert chosen[-1].cumulative == pytest.approx(2.0)

    def test_more_cameras_than_candidates_is_not_an_error(self):
        options = {"a": {"1"}}
        assert len(greedy_cover(options, {"1": 1.0}, cameras=5)) == 1

    def test_zero_cameras_returns_nothing(self):
        assert greedy_cover({"a": {"1"}}, {"1": 1.0}, cameras=0) == []

    def test_no_candidates_returns_nothing(self):
        assert greedy_cover({}, {"1": 1.0}, cameras=2) == []

    def test_a_candidate_that_covers_nothing_is_never_chosen(self):
        options = {"blind": set(), "a": {"1"}}
        assert [c.key for c in greedy_cover(options, {"1": 1.0}, cameras=2)] == ["a"]

    def test_unknown_cells_are_ignored_rather_than_crashing(self):
        # A cell present in a candidate but missing from the weights contributes
        # nothing; raising here would make one stale id break a whole run.
        options = {"a": {"1", "ghost"}}
        chosen = greedy_cover(options, {"1": 2.0}, cameras=1)
        assert chosen[0].added == pytest.approx(2.0)

    def test_the_same_input_always_gives_the_same_answer(self):
        options = {"b": {"1"}, "a": {"2"}}
        w = weights(**{"1": 1.0, "2": 1.0})
        runs = {tuple(c.key for c in greedy_cover(options, w, cameras=2)) for _ in range(5)}
        assert len(runs) == 1

    def test_a_tie_goes_to_whichever_came_first(self):
        # Candidates arrive already ranked by range, so input order carries the
        # tie-break: on equal visible coverage, the better-placed one wins.
        w = weights(**{"1": 1.0, "2": 1.0})
        assert greedy_cover({"b": {"1"}, "a": {"2"}}, w, cameras=1)[0].key == "b"
        assert greedy_cover({"a": {"2"}, "b": {"1"}}, w, cameras=1)[0].key == "a"

    def test_result_is_a_coverage_choice(self):
        chosen = greedy_cover({"a": {"1"}}, {"1": 1.0}, cameras=1)
        assert isinstance(chosen[0], CoverageChoice)


class TestJointCoverage:
    def test_union_not_sum(self):
        # Two cameras behind the same ridge see the same cells. Adding their
        # individual scores would double-count and promise coverage that is not
        # there; this is the whole reason the best pair is not the two best.
        options = {"a": {"1", "2"}, "b": {"2", "3"}}
        w = {"1": 1.0, "2": 1.0, "3": 1.0}
        assert joint_coverage(["a", "b"], options, w) == pytest.approx(3.0)

    def test_identical_cameras_add_nothing(self):
        options = {"a": {"1", "2"}, "b": {"1", "2"}}
        w = {"1": 1.0, "2": 1.0}
        assert joint_coverage(["a", "b"], options, w) == pytest.approx(2.0)

    def test_empty_selection_covers_nothing(self):
        assert joint_coverage([], {"a": {"1"}}, {"1": 1.0}) == 0.0
