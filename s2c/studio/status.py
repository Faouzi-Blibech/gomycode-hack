"""Which providers this Studio can use, shown as chips in the header. Computed once at start."""
from __future__ import annotations

from s2c.multiview.blend import blender_runner
from s2c.multiview.pipeline import MvPipeline
from s2c.multiview.slice import find_slicer
from s2c.studio.theme import chip


def provider_status(pipe: MvPipeline) -> dict[str, bool]:
    return {
        "Qwen-VL": pipe.batch_reader is not None or pipe.chat is not None,
        "Qwen-Image": pipe.image_gen is not None,
        "TripoSR": pipe.mesh_provider is not None,
        "Solaria": pipe.depth is not None,
        "PrusaSlicer": find_slicer() is not None,
        "Blender": blender_runner() is not None,
    }


def header_html(status: dict[str, bool]) -> str:
    chips = "".join(chip(f"{'●' if ok else '○'} {name}", "ok" if ok else "info") for name, ok in status.items())
    return ('<div class="studio-header"><div><h1>Sketch-to-CAD Studio</h1>'
            "<p>Sketches or photos of a part, one or more per face → an editable CAD part, drawings and G-code.</p>"
            f"</div><div>{chips}{chip('images deleted after 1 h', 'info')}</div></div>")
