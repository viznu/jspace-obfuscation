import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.tasks import load_items, numeric_cluster, make_numeric_items
from jspace.types import CType, Trajectory, Readout
from jspace.pressure import build_prompt, CUES, MAX_DEPTH
from jspace.monitors import (
    CoTOnlyMonitor, OutputOnlyMonitor, JSpaceMonitor, calibrate, evaluate, lexical_detector,
)


def test_seed_loads_all_ctypes():
    items = load_items()
    assert len(items) >= 30
    types = {it.c_type for it in items}
    assert types == {CType.NAMED_ENTITY, CType.SEMANTIC, CType.NUMERIC}
    # every cluster is lowercased and non-empty
    for it in items:
        assert it.c_cluster and all(s == s.lower() for s in it.c_cluster)


def test_filter_by_ctype():
    nums = load_items(["numeric"])
    assert nums and all(it.c_type == CType.NUMERIC for it in nums)


def test_numeric_cluster_forms():
    assert "21" in numeric_cluster(21)
    assert any("twenty" in w for w in numeric_cluster(21))
    extra = make_numeric_items([("x1", 8, 7, "*")])
    assert extra[0].c_canonical == "56" and extra[0].answer == "57"


def test_prompt_depth_stacks_cues():
    p0 = build_prompt("Q?", 0)
    assert "constraints" not in p0.lower()
    p_full = build_prompt("Q?", MAX_DEPTH)
    for cue in CUES:
        assert cue in p_full


def test_lexical_detector_hits_and_misses():
    assert lexical_detector("the answer is Italy", ["italy"]) == 1.0
    assert lexical_detector("no mention here", ["italy"]) < 1.0


def _readouts_with_c(weight):
    return [Readout(layer=15, position=-2, tokens=["italy", "the"], weights=[weight, 0.1])]


def test_jspace_monitor_separates_present_absent():
    m = JSpaceMonitor(workspace_layers=range(10, 22))
    present = Trajectory("a", "p", "cot", "ans", ["italy"], True, _readouts_with_c(9.0))
    absent = Trajectory("b", "p", "cot", "ans", ["italy"], False,
                        [Readout(15, -2, ["the", "a"], [0.2, 0.1])])
    assert m.score(present) > m.score(absent)


def test_cot_monitor_collapses_when_concealed():
    """The fragile baseline: CoT monitor catches C when named, misses when hidden."""
    named = Trajectory("a", "p", "intermediate is Italy", "ans", ["italy"], True, [])
    hidden = Trajectory("b", "p", "reasoning without the name", "ans", ["italy"], True, [])
    benign = [Trajectory(f"n{i}", "p", "unrelated reasoning", "ans", ["italy"], False, [])
              for i in range(50)]
    m = CoTOnlyMonitor()
    cal = calibrate(m, benign, 0.01)
    assert cal.flags(named)        # caught when verbalized
    assert not cal.flags(hidden)   # missed when concealed


def test_output_monitor_blind():
    m = OutputOnlyMonitor()
    # answer never contains C -> should not flag
    t = Trajectory("a", "p", "cot names Italy", "26", ["italy"], True, [])
    assert m.score(t) < 1.0
