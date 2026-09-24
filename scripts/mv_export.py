"""A MultiViewSpec JSON -> every chosen format and a zip, with the Studio's settings.

  uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format gcode \
      --quality fine --material PETG --layer 0.2 --infill 30 --supports off --scale 100 --out tmp/export"""
import argparse
import shutil
from pathlib import Path

from s2c.multiview.artifacts import build_part, bundle, export_part
from s2c.multiview.settings import FORMATS, ExportSettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--format", action="append", choices=list(FORMATS), help="repeat for several")
    ap.add_argument("--quality", choices=["draft", "normal", "fine"], default="normal")
    ap.add_argument("--material", default="PLA")
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--infill", type=int, default=20)
    ap.add_argument("--supports", choices=["off", "buildplate", "everywhere"], default="buildplate")
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--out", default="tmp/export")
    args = ap.parse_args()
    spec = MultiViewSpec.model_validate_json(Path(args.spec).read_text())
    formats = ExportSettings(formats=args.format or ExportSettings().formats).formats
    printing = PrintSettings(material=args.material, layer_mm=args.layer, infill_pct=args.infill,
                              supports=args.supports, scale_pct=args.scale)
    part = build_part(spec)
    if isinstance(part, MvAbstain):
        raise SystemExit(f"{part.stage}: {part.reason}. {part.remedy}")
    res = export_part(part, formats, MeshSettings(quality=args.quality), printing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in res.files.values():
        shutil.copy2(path, out / path.name)
    zip_path = shutil.copy2(bundle(part, res), out)
    for fmt, path in res.files.items():
        print(f"{fmt:<6} {out / path.name}")
    print(f"zip    {zip_path}")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
