# Drawing Sheets: One Image, Several Orthographic Views

Date 2026-09-27. Approved in chat the same day: labels first, then layout; hidden lines decide holes.

## 1. Goal

A user drops one image that holds several orthographic views of one part, such as a sheet with Front, Top, Right Side, Left Side and Rear views, labelled or not. The Studio:
- splits the sheet into views and names the face each one shows;
- lets the user check the names in Capture;
- builds the part through the existing multi-view pipeline.

Today such an image stops with "Face unknown".

Success:
- A first-angle labelled sheet and a two-view turned part both build in one upload.
- A wrong label is caught and overruled.
- Sheet mode measured on reference parts is close to per-face uploads.
- A sheet never stalls: a view that cannot be named stays "auto" for the user to pick.

## 2. Drafting rules the reader follows (ISO 128, ISO 5456-2, ASME Y14.3)

- **Projection.**
  - First-angle (ISO, the default): the top view is drawn below the front, the view from the left on the right of the front, the view from the right on its left, and the bottom view above.
  - Third-angle (ASME): the top view is above the front, and the right view is on the right.
  - The rear view goes at the far end of the row, beside a side view.
- **Projection symbol.** A truncated cone drawn as a trapezoid, next to its end view (two concentric circles).
  - Circles next to the large end of the trapezoid: first-angle.
  - Circles next to the small end: third-angle.
  - A sheet that holds the symbol and another drawing uses the symbol only to set the projection, never as the part.
- **Line types.**
  - Continuous lines are visible edges.
  - Dashed lines (short, regular dashes) are hidden edges.
  - Chain lines (long and short dashes) are centre lines; they extend beyond the outline and are never part of it.
- **Alignment.**
  - Views in one row share the front's height.
  - Views in one column share the front's width.
  - Top and bottom views are width × depth; left and right views are depth × height; the rear view is width × height.
  - All views share one scale.

## 3. Components

### 3.1 `s2c/multiview/sheet.py`: split

`split_sheet(image_bgr) -> Sheet` finds the views of one image.

1. Build an ink mask: pixels that differ from the median border grey by more than 40.
2. Remove the sheet border: a component that touches or nearly touches all four sides and fills less than 5 % of its box.
3. Cut the image at separator bands: runs of rows (or columns) with more than 90 % ink.
4. Dilate by `max(3, round(0.008 × long side))` px and take connected components. Drop specks under 0.02 % of the image.
5. Sort the components:
   - **View:** a component that is not a label.
   - **Label:** its height is under 6 % of the image, its width is at least 2.5 × its height, and it lies within 1.5 × its height of a view, directly above or below it and horizontally overlapping it.
   - Components whose box lies inside a view's box join that view. This covers centre-line pieces and inner circles.
6. Group views into drawings separated by separator bands, or by gaps wider than 3 × the median view size.

Result:
- `Sheet.drawings: list[Drawing]`.
- `Drawing.views: list[View(box, label_box | None)]`.
- `Sheet.warnings`.

An image with fewer than 2 views in every drawing is not a sheet: `split_sheet` returns a `Sheet` with no drawing that holds 2 or more views, and the caller treats the image as a single view, as today.

### 3.2 `s2c/multiview/sheet.py`: name

`name_views(sheet, image_bgr, projection, reader=None) -> Naming`

- **Projection symbol.** A drawing of exactly 2 views that are a 4-vertex trapezoid and a circle with one concentric inner circle, aligned on one centre line, with lengths within 8 %: the tall side matches the outer diameter and the short side the inner one.
  - It sets the projection per section 2.
  - If another drawing with 2 or more views exists, the symbol drawing is dropped from the part.
- **Part drawing.** The part drawing is the remaining drawing with the most views. If more than one is left, a warning says which one was used.
- **Labels.** When a reader is given, each label box is read.
  - Keywords, matched case-insensitively and allowing one wrong character: front; rear or back; top or plan; bottom; left; right; "side" plus left or right; end.
  - "Elevation" alone means front.
- **Layout.**
  - The front is the view with the most neighbours aligned with it: same row (centres within 10 % of the view's height) or same column (within 10 % of its width). A tie goes to the largest area, then to the top-left.
  - The other views are named by their position (section 2).
  - A view two steps along the row is the rear.
  - With two views, the left (or upper) one is the front.
- **Scale check.** Each name implies a size in the sheet's shared scale. With the front measuring W × H:
  - top or bottom must be W × D;
  - right or left must be D × H;
  - rear must be W × H.
  - D comes from any view that shows it, and the tolerance is 8 %.
  - A label that fails the check is overruled by the layout name, with the warning "<label> does not match its size; used as <face>".
  - When neither passes, the view stays "auto".
- **Result.** `Naming.faces: list[str]` (one per view, a face name or "auto"), `projection`, `projection_source` ("symbol" | "setting"), `warnings`.

`crop_views(sheet, image_bgr, naming) -> list[tuple[bytes, str]]` returns PNG crops of the part's views. Each crop:
- is the view box plus a 4 % margin;
- is on white;
- keeps its scale.

### 3.3 Line drawings in the outline stage

A view is **line art** when its ink fills under 35 % of its filled outer outline. Sketches, and renders or photos of solid parts, are unaffected by this section: it applies only to images of kind "drawing" that are line art.

- **Spur removal.** Centre lines, extension lines and arrowheads sticking out of the closed outline are removed:
  - Open the filled outer region with a square kernel of `max(5, 0.006 × long side)` px.
  - Keep the largest component, and take the outline from it.
- **Openings.** Non-circular regions enclosed by visible lines are edges, never openings. Only circles are candidate holes.
- **Hidden lines.** Dashed segments are detected and returned as axis-parallel hidden lines, in the view's normalised 0..1 box coordinates: runs of at least 3 collinear short ink pieces with regular gaps.

`Observation` gains two fields: `line_art: bool = False` and `hidden: list[tuple[str, float, float, float]]`. Each entry is ("h" | "v", position, start, end), normalised to the outline bbox.

### 3.4 Holes in line drawings: which circles are holes

For a line-art observation, each circle of diameter d, at position p in face F, is classified using the other line-art views that share an axis with F.

1. **Hole:** a neighbour shows a pair of hidden lines parallel to F's viewing axis, at p ± d/2, within 3 % of the view.
2. **Edge:** a neighbour's outer silhouette, measured across that shared axis, has a width of d (within 4 %) centred on p at some station along F's viewing axis. That is a step, a shoulder or a taper end, which explains the circle; this is the cone's small end.
3. **Hole:** otherwise. A visible circle must be some edge, and a circle that no silhouette explains is a hole. With no neighbouring view, the circle is also a hole.

Edges are dropped from the features. Each verdict of type 2 or 3 adds a Review warning:
- "Circle Ø<d> on <face>: read as a hole, no other view explains it";
- "... read as an edge (step in <face>)".

This refines the chat choice ("hidden lines decide"). Drawings often leave hidden lines out, and a bracket's lug holes would otherwise vanish.

### 3.5 Missing side view of a turned part

If a canonical view is round (the roundness conditions 1-2 of `turned.turned_axis`) and exactly one of its two side views is observed, the other side view is that one, copied across the axis:
- source "inferred", provenance "inferred";
- added before completion, so the turned-part build applies.

### 3.6 Studio

- **Upload.** An uploaded image with face "auto" is run through `split_sheet`. If its part drawing has 2 or more views, the item is replaced by one item per view:
  - the crop is saved under the session folder;
  - the face is the detected name ("auto" if unnamed);
  - the kind is "drawing".
  - A Capture card shows the sheet warnings and the projection used.
- **Projection switch.** A radio in Capture: "First-angle (ISO)" / "Third-angle (US)", default first-angle.
  - A symbol on the sheet overrides it, and the card says so.
  - Changing it renames the views of every sheet item not edited by hand. The session keeps the Sheet and Naming for this.
- **Example.** A "Try a drawing sheet" button loads a first-angle sheet generated by our own code: `examples/mv/sheet/sheet.png`, drawn from a known part with labels and a projection symbol.
- **Sizes.** Unchanged: the user types sizes. "Use suggested sizes" fills the others from each view's proportions, and the sheet's shared scale keeps them consistent.

## 4. Rules that stay

- Rule 1: no code from any model.
- Rule 2: every millimetre is typed, measured or accepted by the user. The sheet gives proportions only, never millimetres.
- `spec.py` is frozen.
- No new dependencies.
- The OCR reader is the existing TrOCR reader when available. With no reader, naming is by layout only.

## 5. Out of scope

- Title blocks and tables.
- Section views, auxiliary or inclined views, detail views, broken views and isometric views on the sheet. An isometric view is skipped when it does not align with any view.
- Reading dimension values from a sheet beyond what the OCR stage already does.
- Hand-drawn sheets are best effort.

## 6. Testing

- **Synthetic sheets** drawn by a test helper from known solids (`face_mask` outlines as lines):
  - first-angle and third-angle;
  - with and without labels;
  - with centre lines, dashed hidden lines, a projection symbol, a border frame and two drawings in one image.
- **Named cases:**
  - the mislabel ("Rear View" drawn with the top view's size gets overruled to bottom);
  - the two-view truncated cone (builds as a cone, inner circle not a hole);
  - a washer (bore is a hole);
  - a bracket lug (hole kept without hidden lines).
- **Sheet benchmark:**
  - Compose first-angle line-art sheets from about 40 reference parts: Canny edges of the front, top and right renders, laid out with labels on half of them.
  - Run them through split, name and the pipeline.
  - Compare with per-face uploads of the same parts. Reported: named correctly %, built %, median 3D IoU.
