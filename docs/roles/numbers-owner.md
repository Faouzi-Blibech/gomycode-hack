# Role brief: Numbers owner (handwriting recognition teammate)

Paste this whole file into your AI assistant at the start of a session, together with `CLAUDE.md`. Provisional until the team meeting confirms your approach.

## What the product does

Phone photo of a sketch or a real part becomes an editable parametric CAD model. You own every millimetre value that enters the system. Nothing else in the pipeline is allowed to produce a number, so your accuracy is the accuracy of the product.

## You own

- `ocr.py`: `read_annotations(image, topology) -> Annotations`. Finds handwritten or printed dimension text on a sketch or drawing, reads the value, and links it to a field: width, height, thickness, hole diameter, and so on.
- `metrology.py`: `measure(image) -> Measurements | Abstain`. Finds the coin, checks it is flat, computes mm per pixel, extracts the outer contour and the hole circles in millimetres.
- `tests/test_ocr.py` and `tests/test_metrology.py` with a labelled set and an accuracy number the team can put on a slide.

## Your contract

Both functions return models from `partspec/`. Read them before writing code.

`Annotation` has `value_mm`, `kind` (linear, diameter, radius), `bbox_px`, `linked_to` (a fixed list of field names, or `unknown`), `hole_index`, `confidence`. If you cannot link a value, return it with `linked_to: unknown` and let `merge.py` apply heuristics. Never drop a value you read.

`Measurements` has `mm_per_px`, the coin record with `eccentricity`, `outer_contour_mm`, `bbox_mm`, `circles_mm`, `confidence`. If the coin ellipse eccentricity is above 0.30 or no coin is found, return `Abstain(stage="metrology", reason="coin_tilted" or "coin_not_found", remedy="Lay the coin flat, shoot top-down, retake.")`.

You may use the vision model as a helper for reading digits if a trained model is not ready, through `vision/client.py` only, and only to read text the user wrote. The model must never be asked to estimate a size. The `Topology` you receive tells you how many annotations to expect and where the holes roughly are; use it to link values.

## Two possible approaches, pick one at the team meeting

Trained-model path: a handwritten digit and symbol recogniser (TrOCR, a small CRNN, or a fine-tuned PaddleOCR) on crops found by classical CV. Strong story for the "Quality of AI use" score. Needs a small labelled set, which the golden sketches provide.

VLM-assisted path: classical CV finds text regions and arrows, the vision model reads each crop and returns `{value, kind}` as JSON. Faster to build. Still respects the rule because the model reads what the user wrote.

Either way the linking step is yours: which edge or hole does a value belong to. Heuristics that work on a sketch: a value between two arrowheads links to the edge parallel to the arrow; a value with a diameter symbol or a leader line to a circle links to that hole; the value nearest the short side of the bounding box is thickness if a side view exists.

## Order of work

1. `metrology.py` coin detection with Hough circles on a blurred grayscale, then `fitEllipse` on the contour for the eccentricity gate. Coin table: 1 TND is 25.0 mm, 1 EUR is 23.25 mm, 2 EUR is 25.75 mm, add the coins you actually have. Test on 3 photos at 3 distances, error under 2 percent.
2. `metrology.py` outer contour and hole circles in mm on the 5 golden photos.
3. `ocr.py` text region detection on the 10 golden sketches.
4. `ocr.py` value reading with the chosen approach. Report accuracy.
5. `ocr.py` linking. Report how many golden sketches get every field linked correctly.
6. Abstention tests: tilted coin, no coin, blank page, sketch with no numbers.

## Tips

- Work at a fixed longest side of 1600 px. Resize on input, scale boxes back.
- Handwritten "0" and "O", "1" and "l", "5" and "S" are the usual errors. A value on a dimension line is always a number, so restrict the character set.
- Log every read with its crop so the team can debug failures in the lab UI.

## Definition of done

- mm-per-pixel error under 2 percent on the coin photos.
- Value accuracy and linking accuracy reported as numbers in the README.
- All abstention tests pass with the right `reason`.
