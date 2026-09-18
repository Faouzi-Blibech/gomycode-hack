# Role brief: Geometry owner (3D design teammate)

Paste this whole file into your AI assistant at the start of a session, together with `CLAUDE.md`. Provisional until the team meeting confirms your tools.

## What the product does

Phone photo of a sketch or a real part becomes an editable parametric CAD model. Your work is everything from a validated `PartSpec` JSON to a printable file and back to a silhouette for checking.

## You own

- `builder.py`: `build(partspec) -> cadquery.Workplane` and `export(solid, out_dir) -> (step_path, stl_path)`.
- `views.py`: `silhouettes(solid, px=512) -> dict[str, ndarray]` for front, back, left, right, top, bottom. Each mask is normalised with `s2c.silhouette.normalize_mask`, and `s2c.silhouette.iou` (integrator) compares masks.
- `tests/golden/`: the ground-truth parts. You model at least 10 parts by hand (plates, brackets, flanges, spacers, one profile extrusion), record their exact dimensions in `expected.json`, then draw a sketch of each by hand with the dimensions written on it and photograph it. For 5 of them, produce or find a physical version and photograph it top-down next to a coin.
- The Three.js STL viewer inside `web/` and the slider-to-geometry feel.
- The demo parts and, if a printer is found, the printed part for the video.

## Your contract

Input is a `PartSpec` from `partspec/`. You never parse images and never call a model. If a spec is valid but cannot be built (a hole outside the plate, a bolt circle larger than the flange), raise `BuildError(reason, remedy)` and the integrator maps it to an `Abstain` with `stage: build`.

The grammar is frozen: `plate`, `l_bracket`, `flange`, `spacer`, `profile_extrusion`, with `hole`, `slot`, `fillet`, `chamfer`. Do not add types. If a demo part does not fit, choose a different demo part.

Coordinate convention: front view is the XY plane, extrusion along +Z. Origin at the bottom-left of the front-view bounding box, x right, y up. L-bracket leg `a` lies in XY, leg `b` rises from the far edge. Positions of features are in that frame.

## Order of work

1. `build()` for `plate` and `spacer` with volume tests (volume of a plate equals width times height times thickness minus hole volumes, within 0.5 percent).
2. `export()` to STEP and STL. Open both in FreeCAD or your CAD tool to confirm.
3. `flange`, `l_bracket`, `profile_extrusion`.
4. Features: `hole` (through and blind), `slot`, `fillet`, `chamfer` with the edge selectors from the grammar.
5. `views.py`: render each face to a binary mask. Simplest approach: tessellate the solid, project every triangle along the view axis, and rasterise with OpenCV `fillPoly`. Test: `s2c.silhouette.iou` of a built plate against its own front silhouette above 0.98.
6. The golden set. This is the most valuable thing you can deliver in the first three days because the other two cannot test without it.
7. Three.js viewer: load an STL blob, orbit controls, auto-fit camera, re-load on every slider change.

## Tips

- Use `uv run pytest tests/test_builder.py -x` after every change.
- Keep every builder function under 40 lines. One function per part type, one per feature.
- CadQuery fillet selectors fail on degenerate edges. Catch and raise `BuildError("fillet_failed", "Reduce the fillet radius.")`.
- Export STL with a fine tolerance (`tolerance=0.01, angularTolerance=0.1`) so prints look clean.

## Definition of done

- All five types and all four features build and export.
- Volume tests pass for every type.
- Self round-trip IoU above 0.98 for every golden part.
- 10 golden sketches and 5 golden photos in `tests/golden/` with `expected.json`.
