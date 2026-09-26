"""Hand-written MultiViewSpec -> STEP, STL and G-code.
Usage: uv run python scripts/mv_build.py examples/mv/l_bracket.json --out tmp/mv_demo"""
import argparse
from pathlib import Path

from s2c.multiview.build import BuildError, build, export
from s2c.multiview.slice import slice_solid
from s2c.multiview.spec import MultiViewSpec, MvAbstain


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("spec")
    ap.add_argument("--out", default="tmp/mv_demo")
    args = ap.parse_args()
    spec = MultiViewSpec.model_validate_json(Path(args.spec).read_text())
    try:
        solid = build(spec)
    except BuildError as e:
        raise SystemExit(f"build: {e.reason}. {e.remedy}")
    step, stl = export(solid, args.out)
    print(f"STEP    {step}\nSTL     {stl}")
    res = slice_solid(solid, args.out)
    if isinstance(res, MvAbstain):
        raise SystemExit(f"slice: {res.reason}. {res.remedy}")
    print(f"print   {res.print_stl}\nG-code  {res.gcode or 'none'}")
    if res.print_time_s is not None:
        print(f"time    {res.print_time_s / 60:.0f} min, filament {res.filament_g or 0:.1f} g")
    for w in res.warnings:
        print("warning:", w)


if __name__ == "__main__":
    main()
