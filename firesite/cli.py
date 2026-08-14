# Copyright 2026 the firesite authors
# SPDX-License-Identifier: Apache-2.0
"""Command line entry point for firesite."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import __version__, firms, hotspots, siting


def _load(
    path: Path, timezone: str, keep_low_confidence: bool, keep_persistent: bool
) -> pd.DataFrame:
    raw = pd.read_csv(path)
    frame = firms.normalize(raw, timezone=timezone)
    if not keep_low_confidence:
        frame = pd.DataFrame(frame[frame["high_confidence"]])
    if not keep_persistent:
        before = len(frame)
        frame = hotspots.drop_persistent_sources(frame)
        dropped = before - len(frame)
        if dropped:
            print(
                f"dropped {dropped} detections on fixed thermal sources "
                f"({dropped / before:.0%}); pass --keep-persistent to keep them",
                file=sys.stderr,
            )
    return frame


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--timezone",
        default="UTC",
        help="IANA zone for local times, e.g. America/Guayaquil",
    )
    parser.add_argument(
        "--keep-low-confidence",
        action="store_true",
        help="keep detections FIRMS flags as low confidence",
    )
    parser.add_argument(
        "--keep-persistent",
        action="store_true",
        help="keep fixed thermal sources (kilns, flares, quarries)",
    )
    parser.add_argument(
        "--cell-deg",
        type=float,
        default=0.02,
        help="grid cell size in degrees (default 0.02, ~2.2 km)",
    )


def cmd_availability(args: argparse.Namespace) -> None:
    print(firms.availability().to_string(index=False))


def cmd_fetch(args: argparse.Namespace) -> None:
    bbox = (
        firms.BBox.parse(args.bbox)
        if args.bbox
        else firms.BBox.around(args.lat, args.lon, args.radius)
    )
    print(f"area {bbox.as_api()} ({bbox.area_km2():.0f} km2)", file=sys.stderr)

    frames = []
    for source in args.sources:
        print(f"== {source} {args.start} -> {args.end}", file=sys.stderr)
        try:
            got = firms.fetch_range(
                source, bbox, args.start, args.end, cache_dir=Path(args.cache) / source
            )
            print(f"   {source}: {len(got)} detections", file=sys.stderr)
            if not got.empty:
                frames.append(got)
        except firms.FirmsError as exc:
            print(f"   {source} failed: {exc}", file=sys.stderr)
    if not frames:
        raise SystemExit("nothing downloaded")

    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["latitude", "longitude", "acq_date", "acq_time"]
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out, index=False)
    print(f"{len(combined)} unique detections -> {out}")


def cmd_rank(args: argparse.Namespace) -> None:
    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    cells = hotspots.rank_cells(frame, cell_deg=args.cell_deg, top=args.top)
    print(
        f"{len(frame)} detections, "
        f"{frame['ts_local'].min():%Y-%m-%d} -> {frame['ts_local'].max():%Y-%m-%d}\n"
    )
    print(cells.to_string())
    profile = hotspots.temporal_profile(frame)
    print("\nby year:\n" + profile["by_year"].to_string())
    print("\nby month:\n" + profile["by_month"].to_string())


def cmd_search(args: argparse.Namespace) -> None:
    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    cells = hotspots.rank_cells(frame, cell_deg=args.cell_deg)
    bbox = (
        firms.BBox.parse(args.bbox)
        if args.bbox
        else firms.BBox(
            float(cells["lon"].min()),
            float(cells["lat"].min()),
            float(cells["lon"].max()),
            float(cells["lat"].max()),
        )
    )
    total = int(cells["detections"].sum())
    sites = siting.search_sites(
        cells, bbox, total, radius_km=args.radius, step_deg=args.step, top=args.top
    )
    print(f"best positions within {bbox.as_api()} (radius {args.radius:g} km)\n")
    print(
        pd.DataFrame(
            [
                {
                    "lat": s.lat,
                    "lon": s.lon,
                    "detections": s.detections,
                    "cells": s.cells,
                    "coverage": f"{s.coverage:.0%}",
                }
                for s in sites
            ]
        ).to_string(index=False)
    )
    print(
        "\nTerrain is not modelled: check the winners against a topographic map "
        "and against where you can get permission."
    )


def cmd_evaluate(args: argparse.Namespace) -> None:
    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    report = siting.evaluate_site(
        args.lat, args.lon, frame, radius_km=args.radius, cell_deg=args.cell_deg
    )
    print(f"site {report['lat']}, {report['lon']} within {report['radius_km']:g} km\n")
    print(
        f"  {report['detections_in_range']} detections in "
        f"{report['cells_in_range']} cells "
        f"({report['coverage']:.0%} of everything in the input file)"
    )
    print(f"  deepest recurrence in range: {report['max_recurrence_years']} years\n")

    sectors = report["sectors"]
    if sectors.sum():
        print("where the fire sits, by compass sector:")
        peak = sectors.max()
        for name, value in sectors.items():
            bar = "#" * round(28 * value / peak) if peak else ""
            print(f"  {name:3} {bar:<28} {value / sectors.sum():.0%}")

    optics = report["optics"]
    if not optics.empty:
        print("\nwhich camera actually resolves those distances:")
        print(optics.to_string(index=False))
        print(
            f"\n(an incipient plume is taken as {siting.INCIPIENT_PLUME_M:.0f} m wide "
            f"and needs {siting.MIN_PIXELS_ON_TARGET:.0f} px to be detectable)"
        )

    if args.csv:
        report["cells"].to_csv(args.csv, index=False)
        print(f"\ncells in range -> {args.csv}")


def cmd_map(args: argparse.Namespace) -> None:
    from . import plot

    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    site = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
    out = plot.recurrence_map(
        frame,
        output=Path(args.output),
        site=site,
        radius_km=args.radius,
        cell_deg=args.cell_deg,
        title=args.title,
    )
    print(f"map -> {out}")


def cmd_export(args: argparse.Namespace) -> None:
    from . import export

    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    site = (args.lat, args.lon) if args.lat is not None and args.lon is not None else None
    payload = export.build_payload(
        frame,
        site=site,
        radius_km=args.radius,
        cell_deg=args.cell_deg,
        title=args.title,
        max_cells=args.max_cells,
    )
    out = export.write_payload(payload, Path(args.output))
    print(f"{len(payload['cells'])} cells -> {out}")
    if payload["cells_truncated"]:
        print(
            f"  note: only the top {args.max_cells} cells were exported; "
            "raise --max-cells to include the rest",
            file=sys.stderr,
        )


def cmd_viewshed(args: argparse.Namespace) -> None:
    from . import terrain

    frame = _load(
        Path(args.input), args.timezone, args.keep_low_confidence, args.keep_persistent
    )
    cells = hotspots.rank_cells(frame, cell_deg=args.cell_deg)
    seen = hotspots.visible_from(args.lat, args.lon, cells, radius_km=args.radius)
    if seen.empty:
        raise SystemExit("nothing within range of that position")

    tile = terrain.download_tile(args.lat, args.lon, Path(args.cache))
    print(f"DEM tile: {tile.name}", file=sys.stderr)
    checked = terrain.visible_cells(
        seen,
        (args.lat, args.lon),
        tile,
        observer_height_m=args.observer_height,
        target_height_m=args.target_height,
    )

    known = checked[checked["visible"].notna()]
    offtile = len(checked) - len(known)
    visible = known[known["visible"].astype(bool)]
    total_det = int(known["detections"].sum())
    seen_det = int(visible["detections"].sum())

    print(f"site {args.lat}, {args.lon} within {args.radius:g} km\n")
    print(f"  {len(visible)}/{len(known)} cells have a clear line of sight")
    print(
        f"  {seen_det}/{total_det} detections "
        f"({seen_det / total_det:.0%}) are actually visible"
    )
    if offtile:
        print(f"  {offtile} cells fall outside the downloaded tile and were skipped")
    print(
        f"\n  observer {args.observer_height:g} m above ground, "
        f"target {args.target_height:g} m (a rising plume, not the ground)"
    )
    print(
        "  Copernicus GLO-30 is a surface model, so canopy and buildings count "
        "as terrain\n  and it blocks slightly more than bare earth would."
    )

    if args.csv:
        checked.to_csv(args.csv, index=False)
        print(f"\n  per-cell results -> {args.csv}")

    hidden = known[~known["visible"].astype(bool)].nlargest(5, "detections")
    if not hidden.empty:
        print("\nbusiest cells that are blocked:")
        print(
            hidden[
                ["distance_km", "bearing", "detections", "years", "clearance_m"]
            ].to_string(index=False)
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="firesite",
        description="Decide where to put a wildfire detection camera, "
        "using NASA FIRMS fire history.",
    )
    parser.add_argument("--version", action="version", version=f"firesite {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("availability", help="date range each FIRMS source covers")
    p.set_defaults(func=cmd_availability)

    p = sub.add_parser("fetch", help="download fire history for an area")
    area = p.add_mutually_exclusive_group(required=True)
    area.add_argument("--bbox", help="west,south,east,north in decimal degrees")
    area.add_argument("--lat", type=float, help="centre latitude (use with --lon)")
    p.add_argument("--lon", type=float, help="centre longitude")
    p.add_argument(
        "--radius", type=float, default=30.0, help="half-width in km when using --lat/--lon"
    )
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(firms.ARCHIVE_SOURCES[:1]),
        help=f"FIRMS sources; archive: {', '.join(firms.ARCHIVE_SOURCES)}; "
        f"near real time: {', '.join(firms.NRT_SOURCES)}",
    )
    p.add_argument("--cache", default="firms_cache", help="directory for window cache")
    p.add_argument("--output", default="fires.csv")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("rank", help="rank cells by recurrence")
    p.add_argument("input")
    p.add_argument("--top", type=int, default=20)
    _add_common(p)
    p.set_defaults(func=cmd_rank)

    p = sub.add_parser("search", help="grid-search the best camera positions")
    p.add_argument("input")
    p.add_argument("--bbox", help="restrict the search area")
    p.add_argument("--radius", type=float, default=15.0, help="camera reach in km")
    p.add_argument("--step", type=float, default=0.02, help="search grid step in degrees")
    p.add_argument("--top", type=int, default=10)
    _add_common(p)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("evaluate", help="report on one specific position")
    p.add_argument("input")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument("--csv", help="write the cells in range to this file")
    _add_common(p)
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("map", help="render a recurrence map")
    p.add_argument("input")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument("--title", default="Fire recurrence")
    p.add_argument("--output", default="recurrence.png")
    _add_common(p)
    p.set_defaults(func=cmd_map)

    p = sub.add_parser("export", help="write JSON for the web viewer")
    p.add_argument("input")
    p.add_argument("--lat", type=float)
    p.add_argument("--lon", type=float)
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument("--title", default="Fire recurrence")
    p.add_argument("--max-cells", type=int, default=4000)
    p.add_argument("--output", default="docs/data/analysis.json")
    _add_common(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser(
        "viewshed", help="check which cells the terrain actually lets you see"
    )
    p.add_argument("input")
    p.add_argument("--lat", type=float, required=True)
    p.add_argument("--lon", type=float, required=True)
    p.add_argument("--radius", type=float, default=15.0)
    p.add_argument(
        "--observer-height", type=float, default=5.0, help="camera height above ground"
    )
    p.add_argument(
        "--target-height",
        type=float,
        default=50.0,
        help="height of the smoke column to be seen, not of the ground",
    )
    p.add_argument("--cache", default="dem_cache", help="where to keep DEM tiles")
    p.add_argument("--csv", help="write per-cell visibility to this file")
    _add_common(p)
    p.set_defaults(func=cmd_viewshed)

    return parser


#: Options whose value legitimately starts with a minus sign.
_NEGATIVE_OK = ("--bbox", "--lat", "--lon")


def _looks_numeric(token: str) -> bool:
    """True for a number or a comma-separated list of numbers."""
    parts = token.split(",")
    if not all(parts):
        return False
    try:
        for part in parts:
            float(part)
    except ValueError:
        return False
    return True


def _glue_negative_values(argv: list[str]) -> list[str]:
    """Rewrite `--lon -78.2` as `--lon=-78.2`.

    Every western-hemisphere longitude and every southern latitude starts with a
    minus, which argparse reads as the start of another option. Without this the
    tool rejects a correct command line with a confusing "expected one argument".

    Only numeric-looking values are glued. Otherwise a genuine mistake such as
    `--lat --lon 1.0` would fuse into `--lat=--lon` and surface as a baffling
    float parse error instead of argparse's clear complaint.
    """
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        following = argv[i + 1] if i + 1 < len(argv) else None
        if (
            token in _NEGATIVE_OK
            and following is not None
            and following.startswith("-")
            and _looks_numeric(following)
        ):
            out.append(f"{token}={following}")
            i += 2
            continue
        out.append(token)
        i += 1
    return out


def main(argv: list[str] | None = None) -> None:
    raw = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(_glue_negative_values(raw))
    try:
        args.func(args)
    except firms.FirmsError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
