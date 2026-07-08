import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jspace.tasks.counterfactual import build_counterfactuals, main_population, KB


def test_families_and_labels():
    items = build_counterfactuals()
    fam = {f: [i for i in items if i.family == f] for f in
           {"required_two_hop", "sameC_diff_answer", "same_answer_diff_C",
            "distractor_mention", "direct_named"}}
    # positives fire, matched negatives do not
    for f in ("required_two_hop", "sameC_diff_answer"):
        assert fam[f] and all(i.should_fire for i in fam[f])
    for f in ("same_answer_diff_C", "distractor_mention"):
        assert fam[f] and all(not i.should_fire for i in fam[f])


def test_answer_matched_negatives_share_answer_differ_bridge():
    """same_answer_diff_C: answer is Euro but the scored target is NOT the route."""
    items = build_counterfactuals()
    for it in items:
        if it.family == "same_answer_diff_C":
            assert it.answer == "Euro"
            assert it.meta["route"] != it.target        # different bridge
            assert KB[it.target][1] == "Euro"           # target is itself a euro country


def test_distractor_names_target_but_routes_elsewhere():
    items = build_counterfactuals()
    for it in items:
        if it.family == "distractor_mention":
            assert it.target.lower() in it.prompt.lower()   # target named in prompt
            assert it.meta["route"] != it.target            # but not the route


def test_main_population_balanced():
    pos, neg = main_population(build_counterfactuals())
    assert len(pos) >= 20 and len(neg) >= 20
    assert all(p.should_fire for p in pos) and all(not n.should_fire for n in neg)
    assert all(p.family != "direct_named" for p in pos)     # diagnostic excluded
