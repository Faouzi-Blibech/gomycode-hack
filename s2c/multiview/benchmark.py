"""Score the multi-view pipeline against reverse-engineered parts with known STLs and clean renders.
The true envelope is passed as user values (CLAUDE.md rule 2): this measures modeling accuracy given size,
not size recovery. Never uses trimesh.Trimesh.contains (rtree is not installed); voxel IoU uses voxelize+fill."""
from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image

from s2c.multiview import spec as S
from s2c.multiview.artifacts import build_part
from s2c.multiview.pipeline import ImageInput, MvPipeline
from s2c.multiview.raster import Mesh, face_mask, iou, normalize_mask, solid_mesh

MAX_TRIANGLES = 200_000
ROW_FIELDS = ("frame_iou", "vol_true", "vol_built", "vol_err", "voxel_iou", "view_iou", "features")


@dataclass
class RefPart:
    name: str          # "<category>/<dir>"
    category: str
    mesh: Mesh         # our frame, min corner at 0
    envelope: S.Envelope
    renders: dict[str, Path]
    watertight: bool
    triangles: int


def _to_ours(v: np.ndarray) -> np.ndarray:
    """Dataset is Z-up, ours is Y-up. The proper rotation only; the reflection flips the volume sign."""
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], axis=1)


def load_part(part_dir: Path) -> RefPart:
    part_dir = Path(part_dir)
    category = part_dir.parent.name
    meta = json.loads((part_dir / "metadata.json").read_text())
    stl = trimesh.load(part_dir / "model.stl", force="mesh", process=False)
    v = _to_ours(np.asarray(stl.vertices, dtype=np.float64))
    v = v - v.min(axis=0)
    mesh = Mesh(v, np.asarray(stl.faces))
    env = S.Envelope(x_mm=float(v[:, 0].max()), y_mm=float(v[:, 1].max()), z_mm=float(v[:, 2].max()))
    renders = {f: part_dir / f"{f}.png" for f in S.FACES}
    return RefPart(f"{category}/{part_dir.name}", category, mesh, env, renders, bool(meta["watertight"]),
                   len(mesh.faces))


def render_mask(png: Path) -> np.ndarray:
    """255 where a pixel differs from white by more than 30 in the sum of |RGB - 255|."""
    arr = np.array(Image.open(png).convert("RGB")).astype(int)
    return np.where(np.abs(arr - 255).sum(-1) > 30, 255, 0).astype(np.uint8)


def _cells(mesh: trimesh.Trimesh, pitch: float) -> set[tuple[int, int, int]]:
    vox = mesh.voxelized(pitch).fill()
    return set(map(tuple, np.round(vox.points / pitch).astype(int)))


def voxel_iou(a: trimesh.Trimesh, b: trimesh.Trimesh, n: int = 64) -> float:
    pitch = float(max(b.extents)) / n
    ca, cb = _cells(a, pitch), _cells(b, pitch)
    union = len(ca | cb)
    return len(ca & cb) / union if union else 0.0


def _row(ref: RefPart, result: str, secs: float = 0.0, **extra) -> dict:
    row = {"part": ref.name, "category": ref.category, "result": result, "secs": round(secs, 1)}
    row.update(dict.fromkeys(ROW_FIELDS))
    row.update(extra)
    return row


def _score_built(ref: RefPart, pipe: MvPipeline, root: Path, t0: float) -> dict:
    env = ref.envelope
    truth = trimesh.Trimesh(ref.mesh.vertices, ref.mesh.faces, process=False)
    true_vol = abs(float(truth.volume))
    frame_iou = float(np.mean([iou(normalize_mask(face_mask(ref.mesh, f, env)[0]), normalize_mask(render_mask(ref.renders[f])))
                               for f in S.FACES]))
    row = _row(ref, "built", frame_iou=round(frame_iou, 3), vol_true=round(true_vol, 3))
    imgs = [ImageInput(ref.renders[f].read_bytes(), f, "drawing") for f in S.FACES]
    observed = pipe.observe(imgs)
    if isinstance(observed, S.MvAbstain):
        row["result"] = f"abstain {observed.stage}:{observed.reason}"
        row["secs"] = round(time.time() - t0, 1)
        return row
    user_values = {"envelope.x_mm": env.x_mm, "envelope.y_mm": env.y_mm, "envelope.z_mm": env.z_mm}
    fused = pipe.fuse(observed, user_values)
    if isinstance(fused, S.MvAbstain):
        row["result"] = f"abstain {fused.stage}:{fused.reason}"
        row["secs"] = round(time.time() - t0, 1)
        return row
    part = build_part(fused, root=Path(root))
    if isinstance(part, S.MvAbstain):
        row["result"] = f"abstain {part.stage}:{part.reason}"
        row["secs"] = round(time.time() - t0, 1)
        return row
    built_mesh = solid_mesh(part.solid)
    rebuilt = trimesh.Trimesh(built_mesh.vertices, built_mesh.faces, process=False)
    view_iou = {f: round(iou(normalize_mask(face_mask(built_mesh, f, env)[0]),
                             normalize_mask(face_mask(ref.mesh, f, env)[0])), 3) for f in S.FACES}
    built_vol = float(part.volume_mm3)
    row.update(vol_built=round(built_vol, 3), vol_err=round(built_vol / true_vol - 1, 4),
              voxel_iou=round(voxel_iou(rebuilt, truth), 3), view_iou=view_iou, features=len(fused.features))
    row["secs"] = round(time.time() - t0, 1)
    return row


def _run(ref: RefPart, pipe: MvPipeline, root: Path) -> dict:
    t0 = time.time()
    try:
        return _score_built(ref, pipe, root, t0)
    except Exception as e:  # noqa: BLE001 - one part's failure must not stop the whole run (review focus 1)
        return _row(ref, f"error {type(e).__name__}", secs=time.time() - t0)


def score_part(ref: RefPart, pipe: MvPipeline, root: Path, timeout_s: float = 120) -> dict:
    if not ref.watertight:
        return _row(ref, "skipped not_watertight")
    if ref.triangles > MAX_TRIANGLES:
        return _row(ref, "skipped too_large")
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(_run, ref, pipe, root)
        try:
            return future.result(timeout=timeout_s)
        except FutureTimeoutError:
            return _row(ref, "skipped timeout", secs=timeout_s)


def _reason(result: str) -> tuple[str, str | None]:
    kind, _, rest = result.partition(" ")
    return kind, rest or None


def _stats(rows: list[dict]) -> dict:
    built = [r for r in rows if r["result"] == "built"]
    abstained: dict[str, int] = {}
    skipped = errors = 0
    for r in rows:
        kind, rest = _reason(r["result"])
        if kind == "abstain":
            abstained[rest] = abstained.get(rest, 0) + 1
        elif kind == "skipped":
            skipped += 1
        elif kind == "error":
            errors += 1
    abs_vol_err = [abs(r["vol_err"]) for r in built if r.get("vol_err") is not None]
    voxel_ious = [r["voxel_iou"] for r in built if r.get("voxel_iou") is not None]
    mean_view_ious = [statistics.mean(r["view_iou"].values()) for r in built if r.get("view_iou")]
    return {
        "n": len(rows),
        "built": len(built),
        "abstained": abstained,
        "skipped": skipped,
        "errors": errors,
        "median_abs_vol_err": statistics.median(abs_vol_err) if abs_vol_err else None,
        "median_voxel_iou": statistics.median(voxel_ious) if voxel_ious else None,
        "median_view_iou": statistics.median(mean_view_ious) if mean_view_ious else None,
        "within_5pct_vol_err": (sum(1 for e in abs_vol_err if e <= 0.05) / len(built)) if built else None,
    }


def summarize(rows: list[dict]) -> dict:
    categories = {cat: _stats([r for r in rows if r["category"] == cat])
                 for cat in sorted({r["category"] for r in rows})}
    return {"overall": _stats(rows), "categories": categories}


def _fmt(v, pct: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v * 100:.1f}%" if pct else f"{v:.3f}"


def _md_row(name: str, s: dict) -> str:
    return (f"| {name} | {s['n']} | {s['built']} | {sum(s['abstained'].values())} | {s['skipped']} | "
           f"{s['errors']} | {_fmt(s['median_abs_vol_err'])} | {_fmt(s['median_voxel_iou'])} | "
           f"{_fmt(s['median_view_iou'])} | {_fmt(s['within_5pct_vol_err'], pct=True)} |")


def summary_markdown(summary: dict) -> str:
    header = ("| category | n | built | abstained | skipped | errors | median abs(vol_err) | median voxel IoU | "
             "median view IoU | built within 5% |")
    lines = [
        "Reverse-engineering benchmark: clean renders, true envelope given as user values.",
        "",
        header,
        "|---|---|---|---|---|---|---|---|---|---|",
        _md_row("overall", summary["overall"]),
    ]
    lines += [_md_row(cat, s) for cat, s in summary["categories"].items()]
    return "\n".join(lines)
