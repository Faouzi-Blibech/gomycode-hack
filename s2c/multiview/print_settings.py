"""PrintSettings -> PrusaSlicer 2.9 command-line overrides, from a whitelist; values are validated by pydantic before
they get here, because a bad enum makes the slicer print its help and write no G-code. Spec 2026-09-23-studio §4."""
from __future__ import annotations

from s2c.multiview.settings import MATERIALS, PrintSettings


def slicer_flags(p: PrintSettings) -> list[str]:
    nozzle, first_nozzle, bed, first_bed = MATERIALS[p.material]
    first_layer = min(max(p.layer_mm, 0.2), 0.75 * p.nozzle_mm)
    flags = [
        "--filament-type", p.material,
        "--temperature", str(nozzle), "--first-layer-temperature", str(first_nozzle),
        "--bed-temperature", str(bed), "--first-layer-bed-temperature", str(first_bed),
        "--nozzle-diameter", f"{p.nozzle_mm:g}",
        "--layer-height", f"{p.layer_mm:g}", "--first-layer-height", f"{first_layer:g}",
        "--fill-density", f"{p.infill_pct}%", "--fill-pattern", p.infill_pattern,
        "--perimeters", str(p.perimeters), "--brim-width", f"{p.brim_mm:g}",
    ]
    if p.supports == "off":
        flags.append("--no-support-material")
    else:
        flags += ["--support-material", "--support-material-auto"]
        flags.append("--support-material-buildplate-only" if p.supports == "buildplate"
                     else "--no-support-material-buildplate-only")
    return flags
