# Golden multi-view cases

One folder per capture: the face images and `expected.json`. Five parts, each captured twice: once with a single face (`<name>_1view`) and once with three faces (`<name>_3view`).

Parts: plate with two holes, L-bracket with a hole on each leg, flange, spacer, wedge (a part whose side view is a triangle).

Capture rules:
- Sketches: dark pen on plain white paper, one face per sheet, dimensions in mm written outside the outline, with a gap between dimension lines and the part. Diameters with ⌀ or D.
- Photos: part flat on a plain background, phone parallel to the table, one reference object (a coin, a card or an A4 sheet under the part) fully in frame.
- Measure the real part with calipers; those values go in `expected.json`.

expected.json:

    {
      "images": [{"file": "front.jpg", "face": "front", "kind": "sketch"},
                 {"file": "top.jpg", "face": "top", "kind": "sketch"}],
      "reference": null,
      "user_values": {"envelope.x_mm": 60.0, "envelope.y_mm": 40.0, "envelope.z_mm": 5.0},
      "envelope": {"x_mm": 60.0, "y_mm": 40.0, "z_mm": 5.0},
      "hole_count": 2
    }

`user_values` is what a user would type when OCR is off. Without `GOLDEN_AI=1` the test uses it; with `GOLDEN_AI=1` the values must come from OCR or the reference object.

Run: `uv run pytest tests/test_mv_golden.py -v`, or `GOLDEN_AI=1 uv run pytest tests/test_mv_golden.py -v` with a `.env` and the ai extra.
