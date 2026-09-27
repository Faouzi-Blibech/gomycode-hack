"""Debug overlays for the lab screen: what each stage saw. Off unless SKETCH_DEBUG_DIR is set."""
from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

COLOURS = {"visible": (0, 160, 0), "hidden": (200, 80, 0), "centre": (180, 0, 180),
           "dimension": (0, 140, 255), "extension": (0, 200, 200), "leader": (200, 200, 0)}
BADGE = {"written": (0, 160, 0), "uncertain": (0, 165, 255), "predicted": (0, 165, 255),
         "conflict": (0, 0, 230), "derived": (0, 160, 0), "edited": (160, 0, 0)}


def _pt(p):
    return round(p[0]), round(p[1])


def write_overlays(trace, out_dir: Path) -> list[Path]:
    if trace.captured is None:
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    img = trace.captured.sheet.copy()
    for c in (trace.classified or {}).values():
        for kind in ("visible", "hidden", "centre"):
            for p in getattr(c, kind):
                cv2.polylines(img, [np.int32(p.pts)], p.kind == "circle", COLOURS[kind], 2)
        for d in c.dimlines:
            cv2.line(img, _pt(d.p0), _pt(d.p1), COLOURS["dimension"], 2)
            for a in d.arrows:
                cv2.circle(img, _pt(a.tip), 5, (0, 0, 255), -1)
        for e in c.extensions.values():
            cv2.line(img, _pt(e.p0), _pt(e.p1), COLOURS["extension"], 2)
        for lead in c.leaders:
            cv2.line(img, _pt(lead.tail), _pt(lead.tip), COLOURS["leader"], 2)
    records = []
    dims = trace.solved.dims if trace.solved else {}
    for t in trace.texts or []:
        x, y, w, h = t.box
        badge = dims[t.id].badge if t.id in dims else (t.badge or "uncertain")
        cv2.rectangle(img, (x, y), (x + w, y + h), BADGE.get(badge, (128, 128, 128)), 2)
        crop_path = out_dir / f"{stamp}_{t.id}.png"
        cv2.imwrite(str(crop_path), trace.captured.sheet[y:y + h, x:x + w])
        records.append({"id": t.id, "role": t.role, "badge": badge, "crop": crop_path.name,
                        "readings": [r.model_dump() for r in t.readings]})
    for v in trace.views or []:
        x, y, w, h = v.bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (90, 90, 90), 1)
        cv2.putText(img, v.name, (x, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 2)
    overlay = out_dir / f"{stamp}_overlay.png"
    cv2.imwrite(str(overlay), img)
    log = out_dir / f"{stamp}_readings.jsonl"
    log.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8")
    return [overlay, log]
