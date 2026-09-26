"""A MultiViewSpec JSON -> every chosen format and a zip, with the Studio's settings.

  uv run python scripts/mv_export.py examples/mv/l_bracket.json --format stl --format step --format gcode \
      --quality fine --material PETG --layer 0.2 --infill 30 --supports off --scale 100 --out tmp/export"""
import argparse
import shutil
from pathlib import Path

from pydantic import ValidationError

from s2c.multiview.artifacts import build_part, bundle, export_part, sweep
from s2c.multiview.settings import FORMATS, MATERIALS, ExportSettings, MeshSettings, PrintSettings
from s2c.multiview.spec import MultiViewSpec, MvAbstain


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec")
    ap.add_argument("--format", action="append", choices=list(FORMATS), help="repeat for several")
    ap.add_argument("--quality", choices=["draft", "normal", "fine"], default="normal")
    ap.add_argument("--material", choices=list(MATERIALS), default="PLA")
    ap.add_argument("--layer", type=float, default=0.2)
    ap.add_argument("--infill", type=int, default=20)
    ap.add_argument("--supports", choices=["off", "buildplate", "everywhere"], default="buildplate")
    ap.add_argument("--scale", type=float, default=100.0)
    ap.add_argument("--out", default="tmp/export")
    args = ap.parse_args(argv)
    spec = MultiViewSpec.model_validate_json(Path(args.spec).read_text())
    formats = ExportSettings(formats=args.format or ExportSettings().formats).formats
    mesh = MeshSettings(quality=args.quality)
    try:
        printing = PrintSettings(material=args.material, layer_mm=args.layer, infill_pct=args.infill,
                                  supports=args.supports, scale_pct=args.scale)
    except ValidationError as e:
        ap.error(str(e))
    sweep()  # after parse_args AND validation: --help or any bad argument must not touch old builds
    part = build_part(spec)
    if isinstance(part, MvAbstain):
        raise SystemExit(f"{part.stage}: {part.reason}. {part.remedy}")
    res = export_part(part, formats, mesh, printing)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for path in res.files.values():
        shutil.copy2(path, out / path.name)
    settings = {"mesh": mesh.model_dump(), "printing": printing.model_dump(), "formats": formats}
    zip_path = shutil.copy2(bundle(part, res, settings), out)
    for fmt, path in res.files.items():
        print(f"{fmt:<6} {out / path.name}")
    print(f"zip    {zip_path}")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
