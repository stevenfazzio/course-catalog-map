import importlib

viz = importlib.import_module("05_visualize")


def test_primary_format_uses_priority_order():
    assert viz.primary_format(["DIS", "LEC"]) == "Lecture"
    assert viz.primary_format(["INS"]) == "Independent study"
    assert viz.primary_format(["T/D", "INS"]) == "Thesis / dissertation"
    assert viz.primary_format([]) == "No sections this year"
    assert viz.primary_format(["ZZZ"]) == "ZZZ"  # unknown codes pass through


def test_primary_ger_prefers_ways_then_ger_then_none():
    assert viz.primary_ger(["GER:DB-SocSci", "WAY-SI"]) == "WAY-SI"
    assert viz.primary_ger(["GER:DB-SocSci"]) == "GER:DB-SocSci"
    assert viz.primary_ger([]) == "None"


def test_truncate_breaks_on_word_boundary():
    text = "alpha beta gamma delta"
    assert viz.truncate(text, 100) == text
    out = viz.truncate(text, 12)
    assert out.endswith("…") and len(out) <= 12 and " " not in out[-2:]


def test_categorical_pins_neutral_to_grey():
    import numpy as np

    meta, vals = viz.categorical("f", "F", np.array(["a", "b", "None", "a"]), neutral="None")
    assert meta["kind"] == "categorical" and meta["show_legend"] is True
    assert set(meta["color_mapping"]) == {"a", "b", "None"}
    assert meta["color_mapping"]["None"] == "#bdbdbd"
    assert len({meta["color_mapping"]["a"], meta["color_mapping"]["b"]}) == 2
