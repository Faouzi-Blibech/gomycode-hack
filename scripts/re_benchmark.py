"""Score the multi-view pipeline against the reverse-engineering dataset (clean renders, true size given).

  uv run python scripts/re_benchmark.py --dataset "C:/Users/moham/Desktop/GoMyCode/Reverce engineering" --jobs 4
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from s2c.multiview.benchmark import _row, load_part, score_part, summarize, summary_markdown
from s2c.multiview.pipeline import MvPipeline


def _find_parts(dataset: Path, categories: list[str] | None) -> list[Path]:
    cats = categories or sorted(p.name for p in dataset.iterdir() if p.is_dir())
    out = []
    for cat in cats:
        cat_dir = dataset / cat
        if not cat_dir.is_dir():
            continue
        out += sorted(p for p in cat_dir.iterdir() if p.is_dir() and (p / "model.stl").exists())
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("RE_DATASET"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--category", action="append", default=None, help="repeat for several")
    ap.add_argument("--parts", default=None, help="comma list of <category>/<dir>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    args = ap.parse_args(argv)
    if not args.dataset:
        ap.error("--dataset or RE_DATASET is required")
    dataset = Path(args.dataset)

    if args.parts:
        part_dirs = [dataset / p for p in args.parts.split(",")]
    else:
        part_dirs = _find_parts(dataset, args.category)
    if args.limit is not None:
        part_dirs = part_dirs[: args.limit]

    out = Path(args.out) if args.out else Path("tmp/re_bench") / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out.mkdir(parents=True, exist_ok=True)
    parts_root = out / "parts"

    refs = [load_part(d) for d in part_dirs]
    pipe = MvPipeline()

    rows: list[dict] = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as f, \
         ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(score_part, ref, pipe, parts_root, args.timeout): ref for ref in refs}
        for future in as_completed(futures):
            ref = futures[future]
            try:
                row = future.result()
            except Exception as e:  # noqa: BLE001 - a worker crash must not stop the whole run
                row = _row(ref, f"error {type(e).__name__}")
            rows.append(row)
            f.write(json.dumps(row) + "\n")
            f.flush()

    summary = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text(summary_markdown(summary))
    overall = summary["overall"]
    print(f"{overall['built']}/{overall['n']} built, median abs(vol_err) {overall['median_abs_vol_err']}, "
         f"median voxel IoU {overall['median_voxel_iou']}, out={out}")


if __name__ == "__main__":
    main()
