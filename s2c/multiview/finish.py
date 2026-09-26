"""Studio geometry settings applied to a MultiViewSpec. Spec 2026-09-23-studio section 4.
spec.py is unchanged: the finish uses the existing Fillet and Chamfer models, and the size the user chose is
user_edited. "all_vertical" means edges along the depth axis, as the multi-view spec section 5 defines it."""
from __future__ import annotations

from s2c.multiview.settings import GeometrySettings
from s2c.multiview.spec import MultiViewSpec

FINISH_LIMIT = 0.45  # of the smallest envelope side


def max_finish_mm(spec: MultiViewSpec) -> float:
    env = spec.envelope
    return round(FINISH_LIMIT * min(env.x_mm, env.y_mm, env.z_mm), 2)


def apply_geometry(spec: MultiViewSpec, geometry: GeometrySettings) -> tuple[MultiViewSpec, list[str]]:
    if geometry.finish == "none":
        return spec, []
    size = min(geometry.finish_mm, max_finish_mm(spec))
    if size == geometry.finish_mm:
        warnings = []
    else:
        msg = f"{geometry.finish.capitalize()} reduced to {size:g} mm to fit the part"
        warnings = [msg]
    data = spec.model_dump()
    k = len(data["finishes"])
    data["finishes"].append({"type": geometry.finish, "edges": geometry.finish_edges, "radius_mm": size})
    data["provenance"][f"finishes[{k}].radius_mm"] = "user_edited"
    data["warnings"] = [*data["warnings"], *warnings]
    return MultiViewSpec.model_validate(data), warnings
