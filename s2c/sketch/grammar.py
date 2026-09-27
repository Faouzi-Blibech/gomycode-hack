"""Pure parsing of what a reader transcribed: dimension grammar (spec 5.3) and view labels."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Literal

from s2c.sketch.models import ViewName

Kind = Literal["linear", "diameter", "radius", "angle", "chamfer", "thread"]
MAX_VALUE = 5000.0


@dataclass(frozen=True)
class Parsed:
    kind: Kind
    value: float
    count: int = 1
    tolerance: str | None = None
    through: bool = False
    depth: float | None = None
    angle: float | None = None


_NUM = r"(?:\d+(?:\.\d+)?|\.\d+)"
_TOL = rf"(?P<tol>±{_NUM}|\+{_NUM}/-{_NUM})"
_MAIN = re.compile(rf"^(?P<prefix>Ø|D|R|C|M)?(?P<num>{_NUM})(?P<deg>°)?(?:MM)?{_TOL}?$")
_CHAMFER = re.compile(rf"^(?P<size>{_NUM})X(?P<ang>{_NUM})°$")
_COUNT = re.compile(r"^(?P<n>\d+)X(?=[ØRCDM])")
_DEPTH = re.compile(rf"(?:↧|DP|DEEP)(?P<d>{_NUM})$")
_DIGIT_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "L": "1", "I": "1", "|": "1"})


def _normalise(raw: str) -> str:
    t = raw.strip()
    for sign in "⌀∅øΦφ":
        t = t.replace(sign, "Ø")
    t = t.replace("×", "x").replace(",", ".").replace(" ", "")
    if len(t) > 1 and t[0] in "oO" and (t[1].isdigit() or t[1] == "."):
        t = "Ø" + t[1:]
    return t


def _fix_digits(t: str) -> str:
    """Map look-alike letters to digits, but only after the first character (the prefix slot)."""
    if not t:
        return t
    head, tail = t[0], t[1:]
    if head in "OolLI|":
        head = head.translate(_DIGIT_FIXES)
    return head + tail.translate(_DIGIT_FIXES)


def parse_text(raw: str) -> Parsed | None:
    t = _normalise(raw)
    if not t:
        return None
    upper = t.upper()
    through = upper.endswith("THRU")
    if through:
        upper = upper[: -len("THRU")]
    upper = upper.removesuffix("TYP")
    depth = None
    m = _DEPTH.search(upper)
    if m:
        depth = float(m.group("d"))
        upper = upper[: m.start()]
    m = _CHAMFER.match(_fix_digits(upper))
    if m:
        return Parsed("chamfer", float(m.group("size")), angle=float(m.group("ang")))
    count = 1
    m = _COUNT.match(upper)
    if m:
        count, upper = int(m.group("n")), upper[m.end():]
    m = _MAIN.match(_fix_digits(upper))
    if not m:
        return None
    value = float(m.group("num"))
    prefix, deg = m.group("prefix"), m.group("deg")
    kind: Kind = ("angle" if deg else {"Ø": "diameter", "D": "diameter", "R": "radius",
                                       "C": "chamfer", "M": "thread"}.get(prefix or "", "linear"))
    limit = 360.0 if kind == "angle" else MAX_VALUE
    if not 0 < value < limit:
        return None
    return Parsed(kind, value, count=count, tolerance=m.group("tol"), through=through, depth=depth)


# order matters: the more specific words first ("COTE GAUCHE" is left, not right)
_LABELS: list[tuple[str, ViewName]] = [
    ("GAUCHE", "left"), ("LEFT", "left"),
    ("DESSOUS", "bottom"), ("BOTTOM", "bottom"),
    ("ARRIERE", "back"), ("BACK", "back"),
    ("DESSUS", "top"), ("TOP", "top"),
    ("FACE", "front"), ("FRONT", "front"),
    ("DROITE", "right"), ("RIGHT", "right"), ("SIDE", "right"), ("COTE", "right"),
    ("PROFIL", "right"),
]


def match_label(raw: str) -> ViewName | None:
    t = unicodedata.normalize("NFD", raw.upper())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    words = set(re.findall(r"[A-Z]+", t))
    for word, name in _LABELS:
        if word in words:
            return name
    return None
