import pytest

from s2c.sketch.grammar import match_label, parse_text


@pytest.mark.parametrize("raw, kind, value", [
    ("60", "linear", 60.0),
    ("4.00", "linear", 4.0),
    (".50", "linear", 0.5),
    ("12,5", "linear", 12.5),
    ("1.5O", "linear", 1.5),        # O read in a digit position is 0
    ("5O", "linear", 50.0),
    ("Ø12", "diameter", 12.0),
    ("⌀.50", "diameter", 0.5),
    ("∅6", "diameter", 6.0),
    ("o6", "diameter", 6.0),        # a handwritten Ø often reads as o
    ("D 8", "diameter", 8.0),
    ("R5", "radius", 5.0),
    ("r 2.5", "radius", 2.5),
    ("45°", "angle", 45.0),
    ("C2", "chamfer", 2.0),
    ("M6", "thread", 6.0),
    ("40 mm", "linear", 40.0),
])
def test_values_and_kinds(raw, kind, value):
    p = parse_text(raw)
    assert p is not None, raw
    assert p.kind == kind and p.value == pytest.approx(value)


def test_count_prefix():
    p = parse_text("2xØ6")
    assert p.kind == "diameter" and p.value == 6.0 and p.count == 2
    assert parse_text("4 × Ø 5").count == 4


def test_chamfer_size_by_angle():
    p = parse_text("2x45°")
    assert p.kind == "chamfer" and p.value == 2.0 and p.angle == 45.0


def test_tolerance_is_kept_as_text():
    p = parse_text("40±0.1")
    assert p.value == 40.0 and p.tolerance == "±0.1"
    assert parse_text("25 +0.2/-0.1").tolerance == "+0.2/-0.1"


def test_hole_callouts():
    assert parse_text("Ø6 THRU").through is True
    p = parse_text("Ø6 ↧10")
    assert p.depth == 10.0 and p.through is False
    assert parse_text("Ø6 DEEP 12").depth == 12.0


@pytest.mark.parametrize("raw", ["", "hello", "FRONT", "0", "-5", "..", "Ø", "12.5.3", "99999"])
def test_rejects_non_dimensions(raw):
    assert parse_text(raw) is None


@pytest.mark.parametrize("raw, name", [
    ("TOP", "top"), ("Top view", "top"), ("DESSUS", "top"), ("vue de dessus", "top"),
    ("FRONT", "front"), ("FACE", "front"), ("Vue de face", "front"),
    ("SIDE", "right"), ("RIGHT", "right"), ("CÔTÉ", "right"), ("droite", "right"),
    ("PROFIL", "right"), ("LEFT", "left"), ("gauche", "left"), ("côté gauche", "left"),
    ("BOTTOM", "bottom"), ("DESSOUS", "bottom"), ("BACK", "back"), ("ARRIÈRE", "back"),
])
def test_labels(raw, name):
    assert match_label(raw) == name


def test_label_rejects_numbers_and_noise():
    assert match_label("4.00") is None and match_label("stop") is None
