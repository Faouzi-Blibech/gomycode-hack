"""Score the multi-view pipeline against the reverse-engineering dataset (clean renders, true size given).

  uv run python scripts/re_benchmark.py --dataset "C:/Users/moham/Desktop/GoMyCode/Reverce engineering" --jobs 4

Each part is scored in its own subprocess (the hidden --one mode below), so a hang, a native crash (an
OCCT segfault) or a bad metadata.json/STL can only ever take down that one part, never the whole run.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from s2c.multiview.benchmark import bare_row, load_part, score_part, summarize, summary_markdown
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


def _name(part_dir: Path) -> tuple[str, str]:
    return f"{part_dir.parent.name}/{part_dir.name}", part_dir.parent.name


def _score_one(part_dir: Path, root: Path, timeout_s: float) -> dict:
    """Runs inside the --one subprocess: whatever goes wrong, including in load_part, becomes one row."""
    name, category = _name(part_dir)
    try:
        ref = load_part(part_dir)
        return score_part(ref, MvPipeline(), root, timeout_s)
    except Exception as e:  # noqa: BLE001 - this process's only job is to always print exactly one row
        return bare_row(name, category, f"error {type(e).__name__}")


def _run_one(part_dir: Path, parts_root: Path, timeout_s: float) -> dict:
    """Scores one part in its own process. A parent-side timeout or a non-zero exit never touches the run."""
    name, category = _name(part_dir)
    try:
        proc = subprocess.run(
            [sys.executable, __file__, "--one", str(part_dir), "--out", str(parts_root), "--timeout", str(timeout_s)],
            capture_output=True, text=True, timeout=timeout_s + 30, check=False)
    except subprocess.TimeoutExpired:
        return bare_row(name, category, "skipped timeout")
    row = None
    if proc.returncode == 0:
        try:
            row = json.loads(proc.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            row = None
    if row is None:
        row = bare_row(name, category, "error crash")
        row["detail"] = proc.stderr[-300:]
    return row


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default=os.environ.get("RE_DATASET"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--category", action="append", default=None, help="repeat for several")
    ap.add_argument("--parts", default=None, help="comma list of <category>/<dir>")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)  # internal: score exactly one part
    args = ap.parse_args(argv)

    if args.one:
        print(json.dumps(_score_one(Path(args.one), Path(args.out), args.timeout)))
        return

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

    rows: list[dict] = []
    with (out / "results.jsonl").open("w", encoding="utf-8") as f, \
         ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_run_one, d, parts_root, args.timeout): d for d in part_dirs}
        for future in as_completed(futures):
            d = futures[future]
            try:
                row = future.result()
            except Exception as e:  # noqa: BLE001 - a dispatch-side failure must not stop the whole run
                name, category = _name(d)
                row = bare_row(name, category, f"error {type(e).__name__}")
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
