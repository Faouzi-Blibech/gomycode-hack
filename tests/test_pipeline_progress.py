from pathlib import Path

from s2c.multiview.pipeline import ImageInput, MvPipeline

SK = Path(__file__).resolve().parents[1] / "examples" / "mv" / "sketches"


def images():
    return [ImageInput((SK / "front.png").read_bytes(), "front", "sketch"),
            ImageInput((SK / "top.png").read_bytes(), "top", "sketch")]


def test_observe_and_fuse_emit_stage_events_in_order():
    events = []
    pipe = MvPipeline()
    observed = pipe.observe(images(), None, progress=lambda n, d: events.append((n, d)))
    pipe.fuse(observed, {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20},
              progress=lambda n, d: events.append((n, d)))
    keys = [(d["key"], d["state"]) for n, d in events if n == "stage"]
    assert keys[:2] == [("label", "running"), ("label", "done")]
    assert ("outline", "done") in keys and ("draw", "done") in keys and keys[-1] == ("fuse", "done")
    outline = next(d for n, d in events if d.get("key") == "outline" and d["state"] == "done")
    assert len(outline["outline"]) >= 3 and all(len(p) == 2 for p in outline["outline"])
    label = next(d for n, d in events if d.get("key") == "label" and d["state"] == "done")
    assert label["face"] == "front" and label["width"] > 0 and label["height"] > 0


def test_no_progress_callback_changes_nothing():
    pipe = MvPipeline()
    a = pipe.fuse(pipe.observe(images()), {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20})
    b = pipe.fuse(pipe.observe(images(), progress=lambda n, d: None),
                  {"envelope.x_mm": 50, "envelope.y_mm": 30, "envelope.z_mm": 20}, progress=lambda n, d: None)
    assert a.model_dump() == b.model_dump()


class Stop(Exception):
    pass


def test_a_raising_callback_stops_the_pipeline():
    def cb(n, d):
        if d.get("key") == "outline":
            raise Stop()
    try:
        MvPipeline().observe(images(), progress=cb)
    except Stop:
        return
    raise AssertionError("progress exception was swallowed")
