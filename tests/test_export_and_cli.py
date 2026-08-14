# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""The viewer payload, the argument parser, and the map writer.

None of these touch the network. The payload builder is pure, the parser is
pure, and the map is exercised against a temporary path.
"""

from __future__ import annotations

import json
import math

import pandas as pd
import pytest

from firesite import plot
from firesite.cli import build_parser, main
from firesite.export import SCHEMA_VERSION, _clean, build_payload, write_payload

from .test_ranking import detections


def sample(rows: int = 6) -> pd.DataFrame:
    data = [
        (
            0.30 + 0.01 * i,
            -78.22 - 0.01 * i,
            f"{2020 + (i % 4)}-08-0{1 + i % 5}",
            "1200",
            20.0,
        )
        for i in range(rows)
    ]
    return detections(data)


class TestBuildPayload:
    def test_carries_a_schema_version(self):
        assert build_payload(sample())["schema"] == SCHEMA_VERSION

    def test_reports_the_source_range(self):
        payload = build_payload(sample())
        assert payload["generated_from"]["detections"] == 6
        assert payload["generated_from"]["first"] <= payload["generated_from"]["last"]

    def test_cells_carry_what_the_viewer_draws(self):
        cell = build_payload(sample())["cells"][0]
        assert set(cell) == {
            "lat",
            "lon",
            "detections",
            "days",
            "years",
            "max_frp",
            "last_seen",
        }

    def test_no_site_means_no_site_block(self):
        assert build_payload(sample())["site"] is None

    def test_site_block_has_sectors_and_optics(self):
        payload = build_payload(sample(), site=(0.30, -78.22), radius_km=15)
        site = payload["site"]
        assert site["detections_in_range"] > 0
        assert len(site["sectors"]) == 8
        assert site["optics"]

    def test_truncation_is_flagged_not_silent(self):
        payload = build_payload(sample(), max_cells=2)
        assert payload["cells_truncated"] is True
        assert len(payload["cells"]) == 2

    def test_not_truncated_when_it_fits(self):
        assert build_payload(sample(), max_cells=999)["cells_truncated"] is False

    def test_year_and_month_keys_are_integers(self):
        payload = build_payload(sample())
        assert all(isinstance(k, int) for k in payload["by_year"])
        assert all(isinstance(k, int) for k in payload["by_month"])

    def test_round_trips_through_json(self, tmp_path):
        payload = build_payload(sample(), site=(0.30, -78.22))
        path = write_payload(payload, tmp_path / "out.json")
        assert json.loads(path.read_text())["schema"] == SCHEMA_VERSION


class TestClean:
    def test_replaces_nan_with_null(self):
        assert _clean({"a": float("nan")}) == {"a": None}

    def test_replaces_infinity_with_null(self):
        assert _clean([float("inf"), float("-inf")]) == [None, None]

    def test_leaves_finite_numbers_alone(self):
        assert _clean({"a": 1.5, "b": [2, 3]}) == {"a": 1.5, "b": [2, 3]}

    def test_write_payload_rejects_nothing_it_cleaned(self, tmp_path):
        # allow_nan=False would raise if a NaN survived _clean.
        path = write_payload({"a": float("nan"), "b": [math.inf]}, tmp_path / "x.json")
        assert json.loads(path.read_text()) == {"a": None, "b": [None]}


SUBCOMMANDS = [
    ["availability"],
    ["fetch", "--bbox=-1,-1,1,1", "--start", "2024-01-01", "--end", "2024-01-05"],
    ["rank", "x.csv"],
    ["search", "x.csv"],
    ["evaluate", "x.csv", "--lat=0.3", "--lon=-78.2"],
    ["map", "x.csv"],
    ["export", "x.csv"],
]


class TestParser:
    @pytest.mark.parametrize("argv", SUBCOMMANDS, ids=lambda a: a[0])
    def test_every_subcommand_parses(self, argv):
        assert build_parser().parse_args(argv).func is not None

    def test_a_missing_subcommand_is_an_error(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args([])

    def test_evaluate_requires_coordinates(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(["evaluate", "x.csv"])

    def test_negative_coordinates_survive_the_parser(self):
        parsed = build_parser().parse_args(
            ["evaluate", "x.csv", "--lat=-0.2", "--lon=-78.3"]
        )
        assert parsed.lat == -0.2
        assert parsed.lon == -78.3

    def test_fetch_rejects_bbox_and_centre_together(self):
        with pytest.raises(SystemExit):
            build_parser().parse_args(
                [
                    "fetch",
                    "--bbox=-1,-1,1,1",
                    "--lat=0",
                    "--start",
                    "2024-01-01",
                    "--end",
                    "2024-01-02",
                ]
            )


class TestCommandsEndToEnd:
    """The commands that only read a CSV, run through main() as a user would."""

    @pytest.fixture
    def csv(self, tmp_path):
        raw = pd.DataFrame(
            {
                "latitude": [0.30 + 0.01 * i for i in range(8)],
                "longitude": [-78.22 - 0.01 * i for i in range(8)],
                "acq_date": [f"{2020 + (i % 4)}-08-0{1 + i % 5}" for i in range(8)],
                "acq_time": ["1200"] * 8,
                "frp": [25.0] * 8,
                "confidence": ["h"] * 8,
            }
        )
        path = tmp_path / "fires.csv"
        raw.to_csv(path, index=False)
        return path

    def test_rank(self, csv, capsys):
        main(["rank", str(csv), "--top", "3"])
        assert "detections" in capsys.readouterr().out

    def test_evaluate(self, csv, capsys):
        main(["evaluate", str(csv), "--lat=0.30", "--lon=-78.22"])
        out = capsys.readouterr().out
        assert "detections in" in out
        assert "compass sector" in out

    def test_search(self, csv, capsys):
        main(["search", str(csv), "--radius", "20", "--step", "0.05", "--top", "2"])
        assert "best positions" in capsys.readouterr().out

    def test_export(self, csv, tmp_path, capsys):
        out = tmp_path / "payload.json"
        main(["export", str(csv), "--lat=0.30", "--lon=-78.22", "--output", str(out)])
        assert json.loads(out.read_text())["site"]["radius_km"] == 15.0
        assert "cells ->" in capsys.readouterr().out

    def test_map(self, csv, tmp_path, capsys):
        out = tmp_path / "map.png"
        main(["map", str(csv), "--lat=0.30", "--lon=-78.22", "--output", str(out)])
        assert out.exists() and out.stat().st_size > 5000
        assert "map ->" in capsys.readouterr().out


class TestPlot:
    def test_writes_a_png_without_a_site(self, tmp_path):
        out = plot.recurrence_map(sample(), output=tmp_path / "a.png")
        assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"

    def test_writes_a_png_with_a_site(self, tmp_path):
        out = plot.recurrence_map(
            sample(), output=tmp_path / "b.png", site=(0.30, -78.22), radius_km=10
        )
        assert out.exists() and out.read_bytes()[:4] == b"\x89PNG"

    def test_creates_missing_directories(self, tmp_path):
        out = plot.recurrence_map(sample(), output=tmp_path / "deep" / "c.png")
        assert out.exists()


class TestFlagsAndGuards:
    """Paths that need no network but would otherwise go unexercised."""

    @pytest.fixture
    def csv(self, tmp_path):
        raw = pd.DataFrame(
            {
                "latitude": [0.30 + 0.01 * i for i in range(6)],
                "longitude": [-78.22 - 0.01 * i for i in range(6)],
                "acq_date": [f"{2020 + (i % 3)}-0{1 + i % 9}-15" for i in range(6)],
                "acq_time": ["1200"] * 6,
                "frp": [2.0 if i < 3 else 60.0 for i in range(6)],
                "confidence": ["h", "h", "h", "h", "l", "h"],
            }
        )
        path = tmp_path / "fires.csv"
        raw.to_csv(path, index=False)
        return path

    def test_export_viewshed_without_coordinates_is_refused(self, csv, tmp_path):
        # Failing early beats downloading a DEM tile for a site that is not set.
        with pytest.raises(SystemExit):
            main(["export", str(csv), "--viewshed", "--output", str(tmp_path / "x.json")])

    def test_search_without_viewshed_says_so(self, csv, capsys):
        main(["search", str(csv), "--radius", "25", "--step", "0.1", "--top", "2"])
        assert "--viewshed" in capsys.readouterr().out

    def test_keep_low_confidence_admits_more_detections(self, csv, capsys):
        main(["rank", str(csv)])
        strict = capsys.readouterr().out
        main(["rank", str(csv), "--keep-low-confidence"])
        loose = capsys.readouterr().out
        assert strict.split(" detections")[0] < loose.split(" detections")[0]

    def test_keep_persistent_reports_nothing_dropped(self, csv, capsys):
        main(["rank", str(csv), "--keep-persistent"])
        assert "fixed thermal sources" not in capsys.readouterr().err

    def test_evaluate_writes_the_cells_csv_when_asked(self, csv, tmp_path, capsys):
        out = tmp_path / "cells.csv"
        main(["evaluate", str(csv), "--lat=0.30", "--lon=-78.22", "--csv", str(out)])
        assert out.exists()
        assert "distance_km" in out.read_text()

    def test_export_truncation_warns_on_stderr(self, csv, tmp_path, capsys):
        main(["export", str(csv), "--max-cells", "1", "--output", str(tmp_path / "y.json")])
        assert "max-cells" in capsys.readouterr().err
