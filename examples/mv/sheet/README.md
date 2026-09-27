# Example drawing sheet

`sheet.png` is the Studio's "Try a drawing sheet" example. `scripts/make_example_sheet.py` draws it from a flanged angle bracket built in CadQuery. It is a first-angle ISO sheet with the front, top and left side views, view labels, centre lines on the holes and the first-angle projection symbol.

The sheet gives proportions only, never millimetres (rule 2). To build the part, type its true envelope in Review:

| Size | Value |
|---|---|
| Width (X) | 80 mm |
| Height (Y) | 60 mm |
| Depth (Z) | 40 mm |

The part:
- a base flange 80 x 40 x 8 mm with two 9 mm through holes, 12 mm in from each end and 16 mm from the front edge;
- a web 50 mm wide and 8 mm thick, standing at the back edge, 60 mm tall, top corners R10, with a 16 mm through hole 36 mm up.

Regenerate: `uv run python -m scripts.make_example_sheet`
