
"""Command-line interface for exact relative isoperimetric profile evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, List, Sequence, Tuple

from .core import (
    PI,
    candidate_at_area,
    candidate_to_dict,
    center_and_scale_to_area,
    center_polygon,
    max_relative_ratio,
    polygon_area,
    plot_isoperimetric_profile,
    precompute_profile,
    profile_on_areas,
    profile_on_uniform_grid,
    profile_values_to_dict,
    relative_profile_maximum_to_dict,
)

Point = Tuple[float, float]


def _parse_json_polygon(data: Any) -> List[Point]:
    if not isinstance(data, list):
        raise ValueError("JSON polygon must be a list")
    points: List[Point] = []
    for item in data:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("each JSON vertex must be a pair [x, y]")
        points.append((float(item[0]), float(item[1])))
    return points


def load_polygon(path: str | Path) -> List[Point]:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError("polygon file is empty")

    if file_path.suffix.lower() == ".json" or text[0] == "[":
        return _parse_json_polygon(json.loads(text))

    points: List[Point] = []
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        line = line.replace(",", " ")
        parts = [part for part in line.split() if part]
        if len(parts) < 2:
            raise ValueError(f"could not parse line: {raw_line!r}")
        points.append((float(parts[0]), float(parts[1])))

    if not points:
        raise ValueError("no vertices found")
    return points


def write_profile_csv(path: str | Path, profile_payload: dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = zip(
        profile_payload["query_areas"],
        profile_payload["reduced_areas"],
        profile_payload["profile"],
        profile_payload["disk_profile"],
        profile_payload["ratio"],
        profile_payload["families"],
    )
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["query_area", "reduced_area", "profile", "disk_profile", "ratio", "family"])
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute the exact relative isoperimetric profile of a strictly convex polygon."
    )
    parser.add_argument(
        "polygon",
        help="path to a polygon file (.json or plain text with one vertex 'x y' per line)",
    )
    parser.add_argument(
        "--normalize-area",
        type=float,
        default=None,
        help="optionally rescale and recenter the polygon to this total area before evaluation",
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="center the polygon at its centroid before evaluation",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=256,
        help="number of area samples for the reported profile grid on [eps, area/2 - eps]",
    )
    parser.add_argument(
        "--area-epsilon",
        type=float,
        default=None,
        help="area cutoff near 0 and area/2 for the uniform grid; defaults to 2e-3 * area / pi",
    )
    parser.add_argument(
        "--query-area",
        type=float,
        default=None,
        help="also evaluate the exact minimizer at one specific area",
    )
    parser.add_argument(
        "--coarse-points",
        type=int,
        default=256,
        help="coarse grid size for the maximal relative-ratio search",
    )
    parser.add_argument(
        "--fine-points",
        type=int,
        default=1024,
        help="fine grid size for the maximal relative-ratio search",
    )
    parser.add_argument(
        "--local-window",
        type=int,
        default=3,
        help="half-width of the fine-search window around the best coarse point",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="optional JSON output path",
    )
    parser.add_argument(
        "--profile-csv",
        type=str,
        default=None,
        help="optional CSV output path for the sampled profile grid",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="optional output image path for plotting I_P(A) on the sampled grid",
    )
    parser.add_argument(
        "--plot-show",
        action="store_true",
        help="display the profile plot in an interactive window",
    )
    parser.add_argument(
        "--plot-title",
        type=str,
        default=None,
        help="optional custom title for the profile plot",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indentation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    polygon = load_polygon(args.polygon)
    if args.normalize_area is not None:
        polygon = center_and_scale_to_area(polygon, args.normalize_area)
    elif args.center:
        polygon = center_polygon(polygon)

    pre = precompute_profile(polygon)
    profile = profile_on_uniform_grid(
        pre,
        num_points=args.num_points,
        area_epsilon=args.area_epsilon,
    )
    best = max_relative_ratio(
        pre,
        num_points_coarse=args.coarse_points,
        num_points_fine=args.fine_points,
        area_epsilon=args.area_epsilon,
        local_window=args.local_window,
    )

    payload: dict[str, Any] = {
        "polygon": [[float(x), float(y)] for x, y in pre.polygon],
        "num_vertices": len(pre.polygon),
        "total_area": float(pre.total_area),
        "profile": profile_values_to_dict(profile),
        "best_relative_ratio": relative_profile_maximum_to_dict(best),
    }

    if args.query_area is not None:
        query_values = profile_on_areas(pre, [args.query_area])
        query_candidate = candidate_at_area(pre, args.query_area)
        payload["query"] = {
            "requested_area": float(args.query_area),
            "reduced_area": float(query_values.reduced_areas[0]),
            "profile_value": float(query_values.profile[0]),
            "disk_profile_value": float(query_values.disk_profile[0]),
            "ratio": float(query_values.ratio[0]),
            "family": query_values.families[0],
            "candidate": candidate_to_dict(query_candidate),
        }

    if args.profile_csv is not None:
        write_profile_csv(args.profile_csv, profile_values_to_dict(profile))
    if args.plot is not None or args.plot_show:
        plot_isoperimetric_profile(
            profile,
            path=args.plot,
            show=bool(args.plot_show),
            title=args.plot_title,
        )

    text = json.dumps(payload, indent=args.indent)
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text + "\n", encoding="utf-8")

    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
