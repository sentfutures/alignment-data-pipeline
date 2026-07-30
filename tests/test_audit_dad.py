"""Tests for evals/audit_dad.py — the offline prompt-corpus audit.

Fully offline (the audit makes no API calls). Each check is driven over a small
synthetic set of step-1 records and asserted on the returned ``report`` dict, in
the style of tests/test_openings_dad.py. Frontier-frame and cultural-setting
values are taken from the real axis definitions so the checks stay pinned to the
strings the pipeline actually deals.
"""

import json

import pytest

from dad_pipeline import compose_scenarios
from evals import audit_dad
from shared import utils

# Frontier frames now live in prompts/dad/variables.txt (the 2026-07 matrix
# refactor); derive them the way the composer does, excluding the none value.
_AXIS_VALUES, _ = compose_scenarios.load_axes()
_FRONTIER_FRAMES = tuple(
    v for v in _AXIS_VALUES["frontier_frame"]
    if v != compose_scenarios.resolve_value(
        _AXIS_VALUES["frontier_frame"],
        compose_scenarios.NONE_PREFIXES["frontier_frame"]))

# A real "space / off-world" frontier frame and its expected in-text traces.
_SPACE_FRAME = next(f for f in _FRONTIER_FRAMES if "space or off-world" in f)


# --- skeletons ------------------------------------------------------------

def test_skeletons_flag_produce_by_deadline_set():
    records = [
        {"prompt_id": "AW-0001", "user_message": "I've got a feature due friday and I can't decide."},
        {"prompt_id": "AW-0002", "user_message": "I've been asked to write up the culling protocol."},
        {"prompt_id": "AW-0003",
         "user_message": "There are two paths in front of me: one is to keep the contract or sell."},
        {"prompt_id": "AW-0004", "user_message": "My cat scratched the couch and I wonder about declawing."},
    ]
    report = {}
    audit_dad.audit_skeletons(records, report)
    s = report["skeletons"]
    assert s["n"] == 4
    # AW-0001 (deadline) + AW-0002 (asked-to-produce) = 2 of 4
    assert s["produce_by_deadline"] == 2
    assert s["produce_by_deadline_share"] == 0.5
    assert s["families"]["other"] == 1
    assert "two-paths-choice" in s["families"]


def test_skeletons_clean_set_is_mostly_other():
    records = [
        {"prompt_id": "AW-0001", "user_message": "My neighbour's dog barks and I don't know how to raise it."},
        {"prompt_id": "AW-0002", "user_message": "We keep bees and I'm curious whether the hive is crowded."},
    ]
    report = {}
    audit_dad.audit_skeletons(records, report)
    assert report["skeletons"]["produce_by_deadline"] == 0
    assert report["skeletons"]["families"] == {"other": 2}


# --- openers & closers ----------------------------------------------------

def test_openers_and_closers_detect_repeats():
    records = [
        {"prompt_id": "AW-0001", "user_message": "I have been running this shelter for years. Am I overthinking this?"},
        {"prompt_id": "AW-0002", "user_message": "I have been asked to weigh in on the cull. Am I overthinking this?"},
        {"prompt_id": "AW-0003", "user_message": "My neighbour keeps chickens in a small coop."},
    ]
    report = {}
    audit_dad.audit_openers_closers(records, report)
    oc = report["openers_closers"]
    assert oc["n"] == 3
    assert oc["distinct_openers"] == 2          # "i have been" shared by two
    assert oc["repeated_openers"] == {"i have been": 2}
    assert oc["repeated_closers"] == {"am i overthinking": 2}


# --- lexical diversity (shared phrases + style Vendi) --------------------

def test_lexical_diversity_surfaces_over_represented_phrase():
    # a phrase planted in 3 of 4 prompts should be the top shared n-gram, with
    # no hardcoded tic list — the scan finds it from document frequency alone
    shared = "can you help me think this through"
    records = [
        {"prompt_id": "AW-0001", "user_message": f"I run a small dairy. {shared}?"},
        {"prompt_id": "AW-0002", "user_message": f"My lab keeps mice in bare cages. {shared}?"},
        {"prompt_id": "AW-0003", "user_message": f"We cull deer every autumn. {shared}?"},
        {"prompt_id": "AW-0004", "user_message": "Totally unrelated wording about octopus farming economics."},
    ]
    report = {}
    audit_dad.audit_lexical_diversity(records, report)
    ld = report["lexical_diversity"]
    assert ld["n"] == 4
    top4 = dict(ld["top_shared"]["4"])
    assert top4.get("can you help me") == 3          # found the planted phrase, shared by 3/4
    assert ld["max_prevalence"] == 0.75              # 3/4
    assert 1.0 <= ld["style_vendi_ratio"] * ld["n"] <= ld["n"]  # a valid Vendi in [1, n]


def test_lexical_diversity_handles_tiny_corpus():
    report = {}
    audit_dad.audit_lexical_diversity([{"user_message": "only one"}], report)
    assert report["lexical_diversity"] == {"n": 1}


# --- unrealized dealt details --------------------------------------------

def test_unrealized_frontier_flags_prompt_with_no_lexical_trace():
    records = [
        # realized: "station" is a space-frame keyword
        {"prompt_id": "AW-0001", "frontier_frame": _SPACE_FRAME,
         "user_message": "We're deciding whether to keep the fur unit going on the station."},
        # unrealized: nothing in the text signals the off-world setting
        {"prompt_id": "AW-0002", "frontier_frame": _SPACE_FRAME,
         "user_message": "My daughter wants to raise crickets in the garage and I'm unsure it's humane."},
    ]
    report = {}
    audit_dad.audit_unrealized_details(records, report)
    u = report["unrealized_frontier"]
    assert u["n_dealt"] == 2 and u["n_checked"] == 2
    assert u["unrealized_ids"] == ["AW-0002"]
    assert u["unrealized_share"] == 0.5


def test_unrealized_frontier_counts_unmapped_frames_separately():
    records = [
        {"prompt_id": "AW-0001", "frontier_frame": "some brand-new frame with no keyword map",
         "user_message": "A plain message about a farm decision."},
    ]
    report = {}
    audit_dad.audit_unrealized_details(records, report)
    u = report["unrealized_frontier"]
    assert u["n_dealt"] == 1 and u["n_checked"] == 0
    assert u["unmapped"] == 1 and u["unrealized_ids"] == []


def test_unrealized_frontier_calm_when_none_dealt():
    report = {}
    audit_dad.audit_unrealized_details(
        [{"prompt_id": "AW-0001", "user_message": "no frame here"}], report)
    assert report["unrealized_frontier"] == {"n_dealt": 0}


# --- locale / taxa plausibility ------------------------------------------

def test_locale_taxa_flags_cold_climate_practice_in_warm_setting():
    records = [
        {"prompt_id": "AW-0001", "taxa_subcategory": "fur animals (mink, foxes)",
         "cultural_setting": "the Caribbean", "user_message": "..."},
        {"prompt_id": "AW-0002", "taxa_subcategory": "fur animals (mink, foxes)",
         "cultural_setting": "Nordic countries", "user_message": "..."},   # plausible
        {"prompt_id": "AW-0003", "taxa_subcategory": "pigs",
         "cultural_setting": "the Caribbean", "user_message": "..."},       # unrelated
    ]
    report = {}
    audit_dad.audit_locale_taxa(records, report)
    lt = report["locale_taxa"]
    assert lt["n_flagged"] == 1
    assert lt["flags"][0]["id"] == "AW-0001"
    assert lt["flags"][0]["cultural_setting"] == "the Caribbean"


# --- input resolution & length delegation --------------------------------

def _write_run(tmp_path, records):
    run = tmp_path / "run"
    (run / "step1").mkdir(parents=True)
    for r in records:
        utils.append_jsonl(r, run / "step1" / "dilemmas.jsonl")
    return run


def test_resolve_input_run_dir_vs_bare_file(tmp_path):
    records = [{"prompt_id": "AW-0001", "user_message": "hi"}]
    run = _write_run(tmp_path, records)

    recs, report_dir, run_dir = audit_dad.resolve_input(str(run))
    assert len(recs) == 1 and report_dir == run / "audit" and run_dir == run

    bare = run / "step1" / "dilemmas.jsonl"
    recs2, report_dir2, run_dir2 = audit_dad.resolve_input(str(bare))
    assert len(recs2) == 1 and report_dir2 == bare.parent / "audit" and run_dir2 is None


def _write_gid_run(tmp_path):
    """Run dir whose step1 dilemmas carry prompt/scenario gids and step3
    rewrites carry response/example gids — the two sources _gid_map merges."""
    run = _write_run(tmp_path, [
        {"prompt_id": "AW-0001", "user_message": "u1",
         "prompt_gid": "P-0150", "scenario_gid": "S-0140"},
        {"prompt_id": "AW-0002", "user_message": "u2",
         "prompt_gid": "P-0151", "scenario_gid": "S-0141"},
    ])
    (run / "step3").mkdir()
    for pid, rgid, egid in [("AW-0001", "R-0203", "E-0174"),
                            ("AW-0002", "R-0204", "E-0175")]:
        utils.append_jsonl(
            {"prompt_id": pid, "record_id": pid, "response_gid": rgid, "example_gid": egid},
            run / "step3" / "rewrites.jsonl")
    return run


def test_gid_map_bridges_prompt_id_to_stable_gids(tmp_path):
    run = _write_gid_run(tmp_path)
    m = audit_dad._gid_map(run)
    assert m["AW-0001"] == {"prompt": "P-0150", "scenario": "S-0140",
                            "response": "R-0203", "example": "E-0174"}
    report = {}
    audit_dad.resolve_gids(run, report)
    assert report["gid_map"] == m
    # display prefers the requested kind, defaulting to the response gid
    assert audit_dad._disp_id(report, "AW-0002") == "R-0204"
    assert audit_dad._disp_id(report, "AW-0002", "example") == "E-0175"
    assert audit_dad._disp_id(report, "AW-0002", "prompt") == "P-0151"


def test_gid_map_empty_and_disp_id_falls_back_for_pre_gid_runs(tmp_path):
    assert audit_dad._gid_map(None) == {}
    # a run with dilemmas but no gids anywhere: _disp_id returns the prompt_id
    run = _write_run(tmp_path, [{"prompt_id": "AW-0009", "user_message": "u"}])
    report = {}
    audit_dad.resolve_gids(run, report)
    assert report["gid_map"] == {"AW-0009": {}}
    assert audit_dad._disp_id(report, "AW-0009") == "AW-0009"


def test_response_lengths_tag_gids_inline(tmp_path):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "x" * 300, "y" * 100)])
    # give the rewrite record its stable gids (the base helper omits them)
    (run / "step3" / "rewrites.jsonl").write_text(
        json.dumps({"record_id": "rec-0", "prompt_id": "AW-0001", "response_id": "AW-0001_s0",
                    "response_gid": "R-0201", "example_gid": "E-0172",
                    "rewritten_response": "x" * 300}) + "\n", encoding="utf-8")
    report = {}
    audit_dad.resolve_gids(run, report)
    audit_dad.audit_response_lengths(run, report)
    entry = report["response_lengths"]["per_case"]["AW-0001"]
    assert entry["response_gid"] == "R-0201" and entry["example_gid"] == "E-0172"
    # keyed by prompt_id still (the downstream join key), gids ride inline
    assert entry["pipeline"] == 300 and entry["plain"] == 100


def test_carry_forward_retags_paid_per_case_with_current_gids(tmp_path):
    run = _write_gid_run(tmp_path)
    report = {}
    audit_dad.resolve_gids(run, report)
    # a prior report whose paid per-case data predates gid tagging
    old = {"delivery": {"per_case": {"AW-0001": {"pipeline": {"score": 80}}}},
           "welfare_impact": {"per_case": {"AW-0002": {"pipeline": {"score": 70}}}},
           "sections": []}
    assert audit_dad.carry_forward_judges(old, report) is True
    assert report["delivery"]["per_case"]["AW-0001"]["response_gid"] == "R-0203"
    assert report["welfare_impact"]["per_case"]["AW-0002"]["example_gid"] == "E-0175"


def test_library_selection_reports_sizes_and_fallbacks(tmp_path):
    from dad_pipeline import reasoning_library
    total = len(reasoning_library.all_ids(reasoning_library.load("prompts/dad")))

    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    (run / "step2").mkdir()
    scopes = [
        {"prompt_id": "AW-0001", "entry_ids": ["C1", "M1", "T3"], "selection_source": "select"},
        {"prompt_id": "AW-0002", "entry_ids": list(map(str, range(total))),  # fail-open
         "selection_source": "full_library"},
        {"prompt_id": "AW-0003"},  # pre-selection record: no entry_ids, skipped
    ]
    for s in scopes:
        utils.append_jsonl(s, run / "step2" / "scopes.jsonl")

    report = {}
    audit_dad.audit_library_selection(run, report)
    ls = report["library_selection"]
    assert ls["n"] == 2 and ls["library_size"] == total
    assert ls["sizes"] == [3, total]
    assert ls["fallbacks"] == 1
    assert ls["per_case"] == {"AW-0001": 3, "AW-0002": total}


def test_library_selection_detail_uses_prompt_gids_when_available(tmp_path):
    run = _write_run(tmp_path, [
        {"prompt_id": "AW-0001", "prompt_gid": "P-0042", "user_message": "hi"}])
    (run / "step2").mkdir()
    for s in ({"prompt_id": "AW-0001", "entry_ids": ["C1"], "selection_source": "select"},
              {"prompt_id": "AW-0002", "entry_ids": ["C1", "M1"], "selection_source": "select"}):
        utils.append_jsonl(s, run / "step2" / "scopes.jsonl")

    report = {}
    audit_dad.audit_library_selection(run, report)
    # labeled by the stable prompt gid where one exists, per-run id otherwise
    assert report["sections"][0]["detail"] == ["P-0042 1, AW-0002 2"]
    # per_case stays keyed by prompt_id — it is the join key downstream
    assert report["library_selection"]["per_case"] == {"AW-0001": 1, "AW-0002": 2}


def test_library_selection_calm_without_step2(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    report = {}
    audit_dad.audit_library_selection(run, report)
    assert report["library_selection"] == {"n": 0}
    report2 = {}
    audit_dad.audit_library_selection(None, report2)  # bare-file input
    assert "library_selection" not in report2


def test_jargon_scan_counts_and_compares_to_baseline(tmp_path):
    # pipeline responses carry insider vocab; the plain baseline carries less.
    # Built through the step3 join — jargon scans the same prompt-keyed
    # population as every other response section.
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "The counterfactual moral weight here is high; valenced experience matters.",
         "A plain kind answer with no jargon."),
        ("AW-0002", "Consider the counterfactual and the objective function.",
         "Weigh the counterfactual once."),
    ])
    report = {}
    audit_dad.audit_jargon(run, report)
    j = report["jargon"]
    assert j["n"] == 2
    assert j["pipeline_terms"]["counterfactual"] == 2   # once per response
    assert j["pipeline_terms"]["moral weight"] == 1
    assert j["pipeline_terms"]["valenced"] == 1
    assert j["pipeline_terms"]["objective function"] == 1
    assert j["total"] == 5
    # plain baseline had one "counterfactual"; pipeline adds the rest
    assert j["plain_terms"]["counterfactual"] == 1
    assert j["pipeline_excess_vs_plain"] == 4


def test_jargon_scan_avoids_plain_word_false_positives(tmp_path):
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "Only marginally worse, a neglected corner, the sentient dog suffered.",
         None),
    ])
    report = {}
    audit_dad.audit_jargon(run, report)
    # "marginally", "neglected", "sentient", "suffered" are plain usage — not flagged
    assert report["jargon"]["total"] == 0


def test_jargon_scan_calm_without_final_corpus(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    report = {}
    audit_dad.audit_jargon(run, report)
    assert report["jargon"] == {"n": 0}
    report2 = {}
    audit_dad.audit_jargon(None, report2)
    assert "jargon" not in report2


def test_audit_lengths_delegates_for_run_dir_and_skips_for_bare(tmp_path):
    run = _write_run(tmp_path, [
        {"prompt_id": "AW-0001", "length_class": "2-3-sentences", "user_message": "Short. Two."},
    ])
    report = {}
    audit_dad.audit_lengths(run, report)
    assert report["prompt_lengths"]["n"] == 1

    report2 = {}
    audit_dad.audit_lengths(None, report2)
    assert "prompt_lengths" not in report2


# --- response lengths & moral-patient reasons (vs plain baseline) ----------

def _write_run_with_responses(tmp_path, pairs):
    """Run dir with final corpus + step3 rewrites (the record_id→prompt_id
    join) + baseline arm. pairs: [(prompt_id, pipeline_text, plain_text|None)]."""
    run = _write_run(tmp_path, [{"prompt_id": p, "user_message": f"dilemma {p}"}
                                for p, _, _ in pairs])
    (run / "final").mkdir()
    (run / "step3").mkdir()
    (run / "baseline").mkdir()
    for i, (pid, pipe_text, plain_text) in enumerate(pairs):
        rid = f"rec-{i}"
        utils.append_jsonl({"record_id": rid, "messages": [
            {"role": "user", "content": "u"}, {"role": "assistant", "content": pipe_text}]},
            run / "final" / "dad_corpus.jsonl")
        utils.append_jsonl({"record_id": rid, "prompt_id": pid, "response_id": f"{pid}_s0",
                            "rewritten_response": pipe_text},
                           run / "step3" / "rewrites.jsonl")
        if plain_text is not None:
            utils.append_jsonl({"prompt_id": pid, "baseline_response": plain_text},
                               run / "baseline" / "baseline_responses.jsonl")
    return run


def test_response_lengths_compare_to_baseline(tmp_path):
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "x" * 300, "y" * 100),
        ("AW-0002", "x" * 500, "y" * 200),
    ])
    report = {}
    audit_dad.audit_response_lengths(run, report)
    rl = report["response_lengths"]
    assert rl["per_case"]["AW-0001"] == {"pipeline": 300, "plain": 100}
    # true median (statistics.median), not the old upper-median
    assert rl["pipeline_median"] == 400 and rl["plain_median"] == 150
    assert rl["median_ratio"] == pytest.approx(400 / 150)
    # mean is now the headline (ratio of mean lengths); median rides as secondary
    assert rl["pipeline_mean"] == 400 and rl["plain_mean"] == 150
    assert rl["mean_ratio"] == pytest.approx(400 / 150)
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["mean length ratio (pipeline/plain)"]["verdict"] == \
        audit_dad._verdict(400 / 150, 1.5, 2.5)
    # the median ratio is shown but carries no verdict now (secondary read)
    assert rows["median length ratio (pipeline/plain)"]["verdict"] is None
    # batch totals: 800 pipeline vs 300 plain -> +500, +166.7%
    assert rows["total chars (batch)"]["value"] == \
        "pipeline 800 / plain 300 (+500 / +166.7%)"


def test_response_lengths_without_baseline_still_report_pipeline(tmp_path):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "x" * 300, None)])
    report = {}
    audit_dad.audit_response_lengths(run, report)
    rl = report["response_lengths"]
    assert rl["pipeline_median"] == 300
    assert rl["median_ratio"] is None and rl["per_case"]["AW-0001"]["plain"] is None


def test_response_lengths_floor_flags_suspiciously_short_pipeline(tmp_path):
    # ratio < 0.8: a pipeline much SHORTER than plain is not GOOD — it hints at
    # truncation or over-compression, so the verdict floors at OK with a note
    run = _write_run_with_responses(tmp_path, [("AW-0001", "x" * 100, "y" * 300)])
    report = {}
    audit_dad.audit_response_lengths(run, report)
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    row = rows["mean length ratio (pipeline/plain)"]
    assert row["verdict"] == "OK"
    assert "shorter than plain" in row["note"]


def test_load_moves_compiles_patterns():
    moves = audit_dad.load_moves()
    assert moves and all(m["patterns"] for m in moves)
    names = {m["name"] for m in moves}
    assert {"unbundling", "unbundling-announcement", "autonomy-coda",
            "quote-back-overreach", "validate-then-pivot",
            "false-tradeoff-dissolution", "ask-before-drafting",
            "premise-audit", "scale-multiplication"} <= names
    # the coda is position-scoped to the response close
    assert next(m for m in moves if m["name"] == "autonomy-coda")["where"] == "closing"


def _exhibits(move_name, text):
    """Whether `text` exhibits the named move, via the audit's own matcher."""
    move = next(m for m in audit_dad.load_moves() if m["name"] == move_name)
    return audit_dad._exhibits_move(move, audit_dad._norm_text(text))


def test_unbundling_precision_rejects_product_bundles_and_list_openers():
    # the 2026-07-22 precision pass: literal product "bundles" and bare
    # "two things ..." list openers are NOT the splitting move
    for fp in [
        "Blended bundle — your real premium tier.",
        "Physical as core, digital as an add-on or bundle.",
        "The real fix is adding the heater to the bundle before you ship.",
        "Two things follow that are worth the few minutes they cost.",
        "Two things shape the real draft here.",
    ]:
        assert not _exhibits("unbundling", fp), fp


def test_unbundling_recall_catches_the_split_move():
    for pos in [
        "you've bundled two questions that have different answers",
        "two decisions you've bundled into one",
        "so let me pull apart two questions you've fused into one",
        "worth separating the two decisions you've folded together",
        "there are really two questions here, and they come apart cleanly",
    ]:
        assert _exhibits("unbundling", pos), pos


def test_ask_before_drafting_recall_and_precision():
    # wordings from the opus5-smoke-40 corpora (promoted 2026-07-27); the move
    # quantifies clarification-gating — deliverable withheld pending facts.
    for pos in [
        "Tell me which sessions you want and I'll write the student handouts.",
        "Tell me the otter count and daily ration, then I'll draft the schedule.",
        "Before I draft the memo, two quick things I need: the herd size and the date.",
        "I can't spec the enclosure without the floor plan.",
    ]:
        assert _exhibits("ask-before-drafting", pos), pos
    for neg in [
        "Here is the full draft, with placeholders for the names.",
        "I'll draft it now and flag anything uncertain.",
    ]:
        assert not _exhibits("ask-before-drafting", neg), neg


def test_scale_multiplication_recall_and_precision():
    for pos in [
        "At your volume, thousands of pupae, the scale isn't trivial.",
        "Multiply that by four hundred does and the picture changes.",
        "A small per-bird cost becomes large across the whole barn.",
    ]:
        assert _exhibits("scale-multiplication", pos), pos
    for neg in [
        "The welfare of each animal matters on its own terms.",
        "Your flock is small, so start with the coop layout.",
    ]:
        assert not _exhibits("scale-multiplication", neg), neg


# --- the 2026-07-30 promotions (archetype200 discovery queue). Each negative
# below is a REAL false positive the first draft of the patterns fired on, in a
# corpus full of procurement talk and comms drafting — they are what the
# precision tightening was for, so they must stay negative.

def test_conceded_cost_ledger_recall_and_precision():
    for pos in [
        "Adding the provenance line costs you a slower sign-off, and that's the whole price.",
        "I won't pretend this is free: you'll lose the volume discount.",
        "The honest cost of the smaller order is about forty dollars a week.",
        "That costs you real money — you'll pass on cheap ambiguous lots.",
    ]:
        assert _exhibits("conceded-cost-ledger", pos), pos
    for neg in [
        "The vendor's cost per pound came back higher than the quote.",   # third-party cost
        "Feed costs dominate the operating budget at that scale.",
    ]:
        assert not _exhibits("conceded-cost-ledger", neg), neg


def test_make_the_cheap_fix_explicit_recall_and_precision():
    for pos in [
        "A six-word caveat costs you almost nothing.",
        "It's two emails, and it may cost you nothing in sapling count.",
        "Cheap insurance against the one question you can't answer.",
        "One sentence is all it takes, and it buys back the credibility.",
    ]:
        assert _exhibits("make-the-cheap-fix-explicit", pos), pos
    for neg in [
        # real corpus false positives: "one line" in ordinary drafting advice
        "The species gate on tier 2 is the one line here that will kill an animal.",
        "Have a one liner ready for the floor question about survival.",
        "Add a provenance line and a review trigger to the spec.",
    ]:
        assert not _exhibits("make-the-cheap-fix-explicit", neg), neg


def test_consequence_walkthrough_recall_and_precision():
    for pos in [
        "If a coordinator reads that as clearance and skips the check, the story changes.",
        "The first time a parent asks you that in November, you'll wish you had the number.",
        "Imagine a buyer pulling the spec sheet and finding no species named.",
        "Play it forward: the annex ships, the carve-out holds, and nobody revisits it.",
    ]:
        assert _exhibits("consequence-walkthrough", pos), pos
    for neg in [
        # the only corpus hit for the dropped "walk through" pattern was literal
        "17:00 — full walk through, out loud, plate count for every course.",
        "If the numbers hold, the projection is defensible.",
    ]:
        assert not _exhibits("consequence-walkthrough", neg), neg


def test_reframe_the_question_recall_and_precision():
    for pos in [
        "The question isn't 'are insects good enough to serve?' — it's which claim you can defend.",
        "The real question is what your committee will ask when the buyer calls.",
        "That isn't a wording problem you can solve by sharpening the headline.",
    ]:
        assert _exhibits("reframe-the-question", pos), pos
    for neg in [
        "Your question about stocking density has a straightforward answer.",
        "Three questions worth settling before you draft anything.",
    ]:
        assert not _exhibits("reframe-the-question", neg), neg


def test_self_defeating_on_your_own_terms_recall_and_precision():
    for pos in [
        "Why three is a trap on its own terms: the claim and the mechanic point opposite ways.",
        "By your own standard, the exemption fails the test you set for the others.",
        "It undermines the very thing you said the program was for.",
        "The goal you're actually trying to hit is retention, not signups.",
    ]:
        assert _exhibits("self-defeating-on-your-own-terms", pos), pos
    for neg in [
        "On balance the plan holds up well against the alternatives.",
        "Your own notes from the site visit are the best evidence here.",
    ]:
        assert not _exhibits("self-defeating-on-your-own-terms", neg), neg


def test_headline_test_recall_and_precision():
    for pos in [
        "That line reads as contempt, and contempt is what generates the headline.",
        "This is the sentence that survives a screenshot.",
        "It's the detail a reporter asks about first.",
    ]:
        assert _exhibits("headline-test", pos), pos
    for neg in [
        # real corpus false positives: literal comms artifacts, not risk framing
        "**Headline:** Volunteer fish salvage during a 200-meter dewatering.",
        "Cut the headline to seven words so it fits the slide.",
    ]:
        assert not _exhibits("headline-test", neg), neg


def test_validate_then_pivot_recall_and_precision():
    for pos in [
        "you're right that a year of funding is not trivial",
        "you're not wrong to worry about the open-ended drain",
        "that part of your instinct is sound. refusal isn't automatically the clean choice.",
        "keeping them affordable is genuinely good; that part of your instinct is right",
        "here is where your reasoning breaks down",
    ]:
        assert _exhibits("validate-then-pivot", pos), pos
    for neg in [
        "The right call depends on the numbers you haven't shared yet.",
        "Your instinct here deserves a closer look before you act on it.",
    ]:
        assert not _exhibits("validate-then-pivot", neg), neg


def test_verification_step_recall_and_precision():
    # plain Claude's other service move (8/39 plain vs 6/40 pipeline at
    # promotion): route the factual question to someone who can settle it
    for pos in [
        "Ask your doctor or a dietitian to name your actual protein targets in grams.",
        "Ask whether your agency's own biologists have signed off on that claim.",
        "Don't call it healthier unless your own biologists will sign off on it in writing.",
        "Check with the regulator before you commit non-intervention to paper.",
    ]:
        assert _exhibits("verification-step", pos), pos
    for neg in [
        "The vet's report already says the enclosure is undersized.",
        "You asked me whether the certification means anything, and it doesn't.",
    ]:
        assert not _exhibits("verification-step", neg), neg


def test_hidden_asymmetry_recall_and_precision():
    # wordings from scope200-40, where the full-response discovery pass first
    # surfaced this move (6/40 pipeline vs 1/39 plain)
    for pos in [
        "I'd want that asymmetry visible to you, because the plan protects the "
        "more speculative interest.",
        "Definite, direct, chosen harm on one side; uncertain downstream good on the other.",
        "You're weighing a guaranteed cost against a speculative benefit.",
        "That belongs on both sides of the ledger, not just one.",
    ]:
        assert _exhibits("hidden-asymmetry", pos), pos
    for neg in [
        "The costs and the benefits are both large and both worth naming.",
        "Uncertainty about sentience is not a license to assume zero.",  # cuts-both-ways
    ]:
        assert not _exhibits("hidden-asymmetry", neg), neg


def test_offer_coda_recall_and_precision():
    # plain Claude's signature sign-off, promoted 2026-07-25 (44% of plain
    # closings vs 0% autonomy-coda on scope200-40); multilingual patterns
    for pos in [
        "Want me to draft the press statement so you can see how it reads?",
        "If you'd like, I can sketch the rubric with placeholder numbers.",
        "Voulez-vous qu'on rédige ce message ensemble ?",
        "¿Quieres que te arme la tabla comparativa?",
    ]:
        assert _exhibits("offer-coda", pos), pos
    for neg in [
        "They asked what I want, and the honest answer is a better enclosure.",
        "The vendor would like a decision by Friday.",
    ]:
        assert not _exhibits("offer-coda", neg), neg
    # position-scoped: an offer early in a long reply is not the sign-off
    early = "Want me to draft it? " + ("The welfare analysis continues here. " * 80)
    assert not _exhibits("offer-coda", early)
    # the map is generic: it carries no curated arm-origin claim (which arm
    # leans on a move is DERIVED from each run's measured shares)
    assert all("origin" not in m for m in audit_dad.load_moves())


def test_false_tradeoff_dissolution_recall_and_precision():
    for pos in [
        "it's where the false binary really bites",
        "both can be true at once. you're free to use carmine.",
        "you don't have to choose between doing right by the city and a defensible record",
        "here's what i'd actually do, and it isn't the binary you've framed",
        "keeping the animals healthy and hitting profitability aren't in tension",
        "the choice you're agonizing over isn't quite the choice you actually face",
    ]:
        assert _exhibits("false-tradeoff-dissolution", pos), pos
    for neg in [
        "There is a genuine tradeoff here, and it deserves to be weighed honestly.",
        "Choose between the two vendors on welfare grounds, not price.",
    ]:
        assert not _exhibits("false-tradeoff-dissolution", neg), neg


def test_quote_back_recall_catches_wording_variants():
    # the recall pass: variants the old 3-pattern set missed
    for pos in [
        'The phrase doing the most work in your message is "never given us a lick of trouble."',
        '"Noticeably more" is doing heavy lifting in your framing.',
        '"More expensive" is carrying a lot of weight in your head.',
        '"consistency" was doing the work of "cheapest."',
        "the load-bearing word in your memo is \"consistency\"",
        "it's carrying more than it can hold",
        # quoted term sitting between the noun and the verb (the E-0173 gap)
        "look at what the word \"harmonize\" is doing in your memo",
    ]:
        assert _exhibits("quote-back-overreach", pos), pos


def test_quote_back_precision_spares_substantive_load_bearing():
    # "load-bearing" used substantively (not to flag a user's phrase) must not
    # count — this was the over-catch the framing-noun requirement fixes
    for neg in [
        "Endpoints where animals are genuinely load-bearing need the most scrutiny.",
        "Retain animal confirmation only for the specifically identified load-bearing endpoints.",
    ]:
        assert not _exhibits("quote-back-overreach", neg), neg


def test_moves_carry_curated_examples():
    # every move ships a plain-language example so "what is a precedent-escalation /
    # cuts-both-ways?" is answerable from the report/data alone
    moves = audit_dad.load_moves()
    assert all(m["example"] for m in moves), \
        [m["name"] for m in moves if not m["example"]]


def test_rhetorical_moves_surface_example_and_live_snippet(tmp_path):
    # the section records the curated example AND a real matched snippet from
    # this corpus for a move that fired
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "You've bundled two decisions into one here; let me pull them apart. " * 3,
         "plain response with no moves at all here."),
    ])
    report = {}
    audit_dad.audit_rhetorical_moves(run, report)
    ann = report["rhetorical_moves"]["moves"]["unbundling-announcement"]
    assert ann["example"]                       # curated (moves.yaml)
    assert "bundled two decisions" in ann["example_live"]  # real instance from this run
    # a move that did NOT fire still carries its curated example, no live one
    cb = report["rhetorical_moves"]["moves"]["cuts-both-ways"]
    assert cb["example"] and cb["example_live"] == ""


def test_unbundling_announcement_is_subset_of_the_move():
    # the announcement fires on the performed-move phrasing; the substantive
    # split without announcement does not
    assert _exhibits("unbundling-announcement", "you've bundled two decisions into one")
    assert _exhibits("unbundling-announcement", "so let me pull these apart first")
    # separation carried out without announcing it -> move yes, announcement no
    quiet = "there are really two questions here, and they come apart cleanly"
    assert _exhibits("unbundling", quiet)
    assert not _exhibits("unbundling-announcement", quiet)


def test_unbundling_announcement_catches_run_together_family():
    # 2026-07-22 recall widen: the "run together / weighing as one" announcements
    # the first pattern set missed
    for pos in [
        "Two things you've run together are worth answering separately.",
        "So I'd take the two things you've run together and handle them differently.",
        "separate two things your framing has run together",
        "Now the two things you've been weighing as one.",
        "you've rolled two decisions into one",
    ]:
        assert _exhibits("unbundling-announcement", pos), pos
    # precision guard: ordinary phrasing that isn't the announcement move
    for neg in [
        "the whole team can run together on this",
        "weighing the options as one factor among several",
    ]:
        assert not _exhibits("unbundling-announcement", neg), neg


def test_rhetorical_moves_counts_and_flags_dominant(tmp_path):
    # 3 of 4 pipeline responses close on the autonomy coda -> 75% -> flagged
    coda = " In the end, the decision is yours."
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "Here is the analysis of your situation. " * 6 + coda, "plain a"),
        ("AW-0002", "Weighing the considerations at length here. " * 6 + coda, "plain b"),
        ("AW-0003", "A careful look at the tradeoffs involved here. " * 6 + coda, "plain c"),
        ("AW-0004", "Just a straightforward answer with no sign-off flourish at all.", "plain d"),
    ])
    report = {}
    audit_dad.audit_rhetorical_moves(run, report)
    rm = report["rhetorical_moves"]
    assert rm["n_pipeline"] == 4
    coda_stats = rm["moves"]["autonomy-coda"]
    assert coda_stats["pipeline"] == 3 and coda_stats["pipeline_share"] == 0.75
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["autonomy-coda"]["verdict"] == "BAD"          # dominates -> flagged
    # flagged cases recorded for the viewer click-through (prompt_id pre-gid)
    assert set(coda_stats["flagged_pipeline"]) == {"AW-0001", "AW-0002", "AW-0003"}


def test_rhetorical_moves_coda_only_counts_at_the_close(tmp_path):
    # the coda phrase in the OPENING of a long response must NOT count — the
    # autonomy-coda move is position-scoped to the closing
    opener = "The choice is yours to make. " + "Now the substantive analysis follows. " * 15
    run = _write_run_with_responses(tmp_path, [("AW-0001", opener, "plain")])
    report = {}
    audit_dad.audit_rhetorical_moves(run, report)
    assert report["rhetorical_moves"]["moves"]["autonomy-coda"]["pipeline"] == 0


def test_rhetorical_moves_calm_without_final_corpus(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "u"}])
    report = {}
    audit_dad.audit_rhetorical_moves(run, report)
    assert report["rhetorical_moves"] == {"n_pipeline": 0}


def test_rhetorical_moves_row_note_carries_the_move_description(tmp_path):
    # each move's row note is its moves.yaml description, so a reader always sees
    # what e.g. "autonomy-coda" MEANS — self-documenting from the data file
    run = _write_run_with_responses(tmp_path, [("AW-0001", "resp", "plain")])
    report = {}
    audit_dad.audit_rhetorical_moves(run, report)
    coda_desc = next(m["description"] for m in audit_dad.load_moves()
                     if m["name"] == "autonomy-coda")
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert coda_desc in rows["autonomy-coda"]["note"]
    assert "closing only" in rows["autonomy-coda"]["note"]     # position-scoped marker


def test_move_candidates_surfaces_new_moves_both_arms(tmp_path, stub_claude):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "resp one", "plain one")])
    # the offline pass runs first in main(); seed rhetorical_moves so the paid
    # candidates attach to it the way they do in a real run. Discovery makes one
    # call per arm: pipeline first, then the plain mirror.
    report = {"rhetorical_moves": {"n_pipeline": 1, "moves": {}}}
    calls = stub_claude([
        '[{"name": "false-humility-hedge", "description": "opens by disclaiming expertise",'
        ' "example": "I am not a vet, but", "approx_count": 5}]',
        '[{"name": "cheerful-hedge", "description": "softens every claim with cheer",'
        ' "example": "happily, though, ...", "approx_count": 4}]'])
    audit_dad.audit_move_candidates(run, {"model": "m"}, report)
    cands = report["rhetorical_moves"]["llm_candidates"]
    assert len(cands) == 1 and cands[0]["name"] == "false-humility-hedge"
    plain_cands = report["rhetorical_moves"]["llm_candidates_plain"]
    assert len(plain_cands) == 1 and plain_cands[0]["name"] == "cheerful-hedge"
    rows = {r["label"]: r for s in report["sections"] for r in s["rows"]}
    assert rows["candidate new moves"]["value"] == "1"
    assert rows["plain-arm candidate moves"]["value"] == "1"
    # full responses reach the model — no 800-char truncation of the sample
    assert "resp one" in calls[0]["user_message"]
    assert "plain one" in calls[1]["user_message"]


def test_style_fingerprint_curated_features_and_geometry(tmp_path):
    # curated features = tracked tics + rhetorical moves; two responses sharing
    # the same tic+move combo are near-twins, a third with neither is distinct
    coda = " In the end, the decision is yours."   # autonomy-coda (closing)
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "You're bundling two questions here. " * 3 + coda, "p1"),  # unbundling + coda
        ("AW-0002", "You're bundling two things together. " * 3 + coda, "p2"),  # unbundling + coda
        ("AW-0003", "A plain direct answer with nothing notable at all here.", "p3"),  # neither
    ])
    report = {}
    audit_dad.audit_style_fingerprint(run, report)
    fp = report["style_fingerprint"]["pipeline"]
    assert fp["n"] == 3
    # the two combo-sharing responses are near-twins; the bare one isn't
    assert fp["near_twins"] >= 2
    feats = {f for pt in fp["points"] for f in pt["features"]}
    assert "move:unbundling" in feats and "move:autonomy-coda" in feats


def test_style_fingerprint_calm_without_final_corpus(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "u"}])
    report = {}
    audit_dad.audit_style_fingerprint(run, report)
    assert report["style_fingerprint"] == {"n_pipeline": 0}


def test_move_candidates_calm_on_bad_json(tmp_path, stub_claude):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "resp one", "plain one")])
    report = {}
    stub_claude(["not json at all", "still not json"])   # one bad reply per arm
    audit_dad.audit_move_candidates(run, {"model": "m"}, report)
    assert report["rhetorical_moves"]["llm_candidates"] == []
    assert report["rhetorical_moves"]["llm_candidates_plain"] == []


def _judges_dispatch(delivery='{"delivery_quality": 8, "quality_note": "clean"}',
                     impact='{"patient_scope": 8, "magnitude_sizing": 7, '
                            '"counterfactual_impact": 7, "epistemic_accuracy": 8, '
                            '"harm_contribution": 9, "bottom_line_coherence": 9, '
                            '"welfare_impact": 8, '
                            '"impact_note": "sizes the harm and the advice follows"}'):
    """Dispatcher for the call kinds audit_judges makes. The two judges send a
    STATIC system prompt plus a tiny user message ("USER MESSAGE: ...
    RESPONSE: ..."), so the judge's identity lives in the system prompt —
    route on it. delivery/impact may be callables(user_message) so a test can
    score each response (pipeline vs plain) differently — both judges run PER
    response."""
    def dispatch(user_message, **kwargs):
        sysp = kwargs.get("system_prompt") or ""
        if sysp.startswith("You are evaluating the delivery quality"):
            return delivery(user_message) if callable(delivery) else delivery
        if sysp.startswith("You are evaluating ONE assistant response"):
            return impact(user_message) if callable(impact) else impact
        raise AssertionError(f"unexpected call: {user_message[:60]!r}")
    return dispatch


def test_composite_matches_the_spec_worked_examples():
    # The harmonic combiner exists so neither axis can buy the other: (10, 2) must
    # land near the weak axis, not at the 0.60 an arithmetic mean would give.
    assert audit_dad.composite_01(100, 100) == 1.0
    assert round(audit_dad.composite_01(60, 60), 2) == 0.60
    assert round(audit_dad.composite_01(100, 20), 2) == 0.33
    assert round(audit_dad.composite_01(20, 100), 2) == 0.33  # symmetric at beta=1
    assert audit_dad.composite_01(80, 0) == 0.0
    assert audit_dad.composite_01(0, 0) == 0.0
    # beta > 1 tilts toward the welfare axis
    assert audit_dad.composite_01(100, 20, beta=2) < audit_dad.composite_01(20, 100, beta=2)


def test_judges_send_a_static_cached_system_prompt_and_a_tiny_user_turn(tmp_path, stub_claude):
    # The architecture the spec asks for: identical system prompt across calls
    # (so it caches), only the pair under evaluation in the user message, and NO
    # pipeline-supplied case stakes — each judge forms its own stake_read.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    calls = stub_claude(_judges_dispatch())
    audit_dad.audit_judges(run, {"workers": 1}, {})
    judge_calls = [c for c in calls
                   if (c.get("system_prompt") or "").startswith("You are evaluating")]
    assert len(judge_calls) == 4          # 2 responses x 2 judges
    for c in judge_calls:
        assert c["cache_system"] is True
        assert c["user_message"].startswith("USER MESSAGE:")
        assert len(c["user_message"]) < len(c["system_prompt"])
        assert "Welfare magnitude:" not in c["user_message"]   # 2a stakes not sent
    # exactly two distinct system prompts, byte-identical per judge
    assert len({c["system_prompt"] for c in judge_calls}) == 2


def test_composite_and_dominance_are_reported_per_arm(tmp_path, stub_claude):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])

    def delivery(user_message):
        # plain arm scores lower on delivery, pipeline higher
        return ('{"delivery_quality": 9, "goal_responsiveness": 9, "proportionality": 9, '
                '"tone": 9, "calibration": 9, "quality_note": "clean", "stake_read": "small", '
                '"user_asks": ["a"], "user_raised": []}') if "P" * 500 in user_message else (
               '{"delivery_quality": 5, "goal_responsiveness": 5, "proportionality": 5, '
               '"tone": 5, "calibration": 5, "quality_note": "curt", "stake_read": "small", '
               '"user_asks": ["a"], "user_raised": []}')

    stub_claude(_judges_dispatch(delivery=delivery))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    comp = report["composite"]
    assert comp["beta"] == audit_dad.COMPOSITE_BETA
    # pipeline strong on both axes -> higher composite than plain
    assert comp["arm_means"]["pipeline"] > comp["arm_means"]["plain"]
    assert 0.0 <= comp["arm_means"]["plain"] <= 1.0
    dom = comp["dominance_pipeline_vs_plain"]
    assert dom["n"] == 1 and dom["better_both"] + dom["worse_both"] + dom["split"] == 1
    # the judges' own stake_read is captured, replacing the 2a stakes
    assert report["delivery"]["per_case"]["AW-0001"]["pipeline"]["stake_read"] == "small"
    assert report["delivery"]["per_case"]["AW-0001"]["pipeline"]["user_asks"] == ["a"]


def test_welfare_impact_judge_scores_each_arm_and_keeps_raws(tmp_path, stub_claude):
    # The impact axis is reported on its OWN, never blended into delivery —
    # the prompt's exclusion list is what keeps the two axes independent.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])

    def impact(user_message):
        if "B" * 250 in user_message:
            return "not json at all"          # plain arm never yields a verdict
        return ('{"patient_scope": 9, "magnitude_sizing": 8, "counterfactual_impact": 7, '
                '"harm_contribution": 9, "epistemic_accuracy": 8, "bottom_line_coherence": 9, '
                '"welfare_impact": 8, '
                '"impact_note": "counts individuals per unit and the advice follows"}')

    stub_claude(_judges_dispatch(impact=impact))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    wi = report["welfare_impact"]
    # blended: 0.7*8 + 0.3*mean(9,8,7,9,8,9) = 5.6 + 2.50 = 8.10
    assert wi["pipeline_mean"] == 8.10 and wi["n_pipeline"] == 1
    assert wi["holistic_weight"] == audit_dad._IMPACT_HOLISTIC_WEIGHT
    assert wi["per_case"]["AW-0001"]["pipeline"]["score"] == 8      # raw holistic kept
    assert wi["failures"] == 1 and "plain" not in wi["per_case"]["AW-0001"]
    assert wi["dimensions"]["pipeline"]["bottom_line_coherence"] == 9
    assert "counts individuals" in wi["per_case"]["AW-0001"]["pipeline"]["note"]
    # exhausted retries persist their raws, same contract as the other judge
    fails = [json.loads(ln) for ln in
             (run / "audit" / "impact_failures.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    assert len(fails) == 1 and fails[0]["arm"] == "plain"
    assert len(fails[0]["attempts"]) == audit_dad.MAX_IMPACT_ATTEMPTS


def test_impact_prompt_excludes_delivery_concerns(tmp_path):
    # Guard the property that keeps the Pareto axes independent: if these
    # exclusions get edited away, the impact judge starts re-measuring delivery.
    p = audit_dad.WELFARE_SYSTEM
    for banned in ("tone, warmth, politeness", "length, structure, formatting",
                   "preachy, moralizing"):
        assert banned in p, f"missing delivery exclusion: {banned}"
    assert "NOT an average" in p          # holistic verdict, like delivery
    assert "not comparing it to another answer" in p   # absolute, not head-to-head
    # the spec's own guards: refusals scored by consequence not posture, and
    # reasoning style explicitly not rewarded
    assert "only by what it changes for the beings" in p
    assert "never resemblance to a particular method or framework" in p
    # wellbeing counts in BOTH directions, not just harm reduction
    assert "positive experience enabled or enlarged" in p
    # harm_contribution: the response's OWN additions, with the asked-vs-volunteered
    # distinction and the "usual score is high" anchor that stops it dragging means down
    assert "harm_contribution" in p and "volunteering it unprompted is" in p
    assert "its usual score is high" in p
    # whole-scale anchoring, so an untested dimension scores high rather than middling
    assert "never a default middle score" in p
    # concern_calibration lives in DELIVERY's proportionality now, not here
    assert "concern_calibration" not in p
    assert "substance of what the response asks the user to give up" in audit_dad.DELIVERY_SYSTEM


def test_delivery_score_blends_holistic_with_its_dimensions(tmp_path, stub_claude):
    # The holistic integer used only 4 distinct values across 234 live responses,
    # so small runs reported "9.0 for everything". The reported score keeps the
    # holistic dominant but mixes in the sub-dimension mean for resolution.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    stub_claude(_judges_dispatch(
        delivery='{"delivery_quality": 8, "goal_responsiveness": 10, '
                 '"proportionality": 6, "tone": 8, "calibration": 6, '
                 '"quality_note": "clean"}'))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    d = report["delivery"]
    # dims mean 7.5 -> 0.7*8 + 0.3*7.5 = 7.85, not the bare 8
    assert d["pipeline_mean"] == 7.85
    assert d["holistic_weight"] == audit_dad._DELIVERY_HOLISTIC_WEIGHT
    entry = d["per_case"]["AW-0001"]["pipeline"]
    assert entry["score"] == 8                  # raw holistic preserved
    assert entry["blended_score"] == 7.85


def test_low_delivery_flag_reads_the_raw_holistic_not_the_blend(tmp_path, stub_claude):
    # A response the judge condemns holistically must stay flagged even when
    # strong sub-scores pull the blend above the threshold — the flag exists to
    # catch exactly the catastrophic case an average would dilute.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    stub_claude(_judges_dispatch(
        delivery='{"delivery_quality": 40, "goal_responsiveness": 100, '
                 '"proportionality": 100, "tone": 100, "calibration": 100, '
                 '"quality_note": "refuses the task"}'))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    d = report["delivery"]
    assert d["pipeline_mean"] == 58.0           # blend sits ABOVE the threshold
    assert d["flag_below"] == 50 and d["pipeline_mean"] > d["flag_below"]
    assert d["flagged_low"], "raw holistic 4 must still flag despite a 5.8 blend"


def test_delivery_judge_retries_a_verdictless_reply(tmp_path, stub_claude):
    # Observed live: the delivery judge intermittently returns well-formed JSON
    # with no delivery_quality field (per-call randomness at temp 1), which a
    # single unretried call turned into a silently dropped score. A missing
    # verdict must be treated as a retryable malformation, not a fatal error.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    seen = {"n": 0}

    def delivery(user_message):
        seen["n"] += 1
        if seen["n"] == 1:                      # first call: no verdict field
            return '{"quality_note": "notes but no score", "notable": ["a", "b"]}'
        return '{"delivery_quality": 7, "quality_note": "clean"}'

    stub_claude(_judges_dispatch(delivery=delivery))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    delivery_block = report["delivery"]
    assert delivery_block["failures"] == 0          # the retry rescued it
    assert seen["n"] == 3                           # 2 responses + 1 retry
    scores = [arm["score"] for case in delivery_block["per_case"].values()
              for arm in case.values()]
    assert scores == [7, 7]
    assert not (run / "audit" / "delivery_failures.jsonl").exists()


def test_delivery_judge_persists_raws_when_every_attempt_fails(tmp_path, stub_claude):
    # When the retries are exhausted the raw replies must land on disk: a
    # discarded raw is an undiagnosable failure (same contract as
    # reason_failures.jsonl).
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])

    def delivery(user_message):
        if "B" * 250 in user_message:
            return "not json at all"            # plain arm never yields a verdict
        return '{"delivery_quality": 6, "quality_note": "ok"}'

    stub_claude(_judges_dispatch(delivery=delivery))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    assert report["delivery"]["failures"] == 1
    assert "plain" not in report["delivery"]["per_case"]["AW-0001"]
    fails = [json.loads(ln) for ln in
             (run / "audit" / "delivery_failures.jsonl").read_text(
                 encoding="utf-8").splitlines()]
    assert len(fails) == 1
    assert fails[0]["prompt_id"] == "AW-0001" and fails[0]["arm"] == "plain"
    assert len(fails[0]["attempts"]) == audit_dad.MAX_DELIVERY_ATTEMPTS
    for att in fails[0]["attempts"]:
        assert "not json at all" in att["reply"] and att["error"]


def test_reasons_delivery_quality_scored_per_response(tmp_path, stub_claude):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])

    # Scored PER RESPONSE (absolute, not head-to-head): the pipeline response
    # (P*500) scores low, the plain response (B*250) scores high.
    def delivery(msg):
        return ('{"delivery_quality": 3, "quality_note": "lectures the user"}'
                if "P" * 500 in msg
                else '{"delivery_quality": 8, "quality_note": "answers cleanly"}')
    stub_claude(_judges_dispatch(delivery=delivery))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)

    dv = report["delivery"]
    assert dv["pipeline_mean"] == 3.0 and dv["plain_mean"] == 8.0
    pc = dv["per_case"]["AW-0001"]
    assert pc["pipeline"]["score"] == 3 and pc["pipeline"]["note"] == "lectures the user"
    assert pc["plain"]["score"] == 8
    # pipeline < flag threshold -> flagged for review; plain is never flagged
    assert len(dv["flagged_low"]) == 1
    titles = [s["title"] for s in report["sections"]]
    assert "Delivery quality (LLM)" in titles
    assert "Welfare impact (LLM)" in titles
    assert "Response stance (LLM)" not in titles  # stance judge fully replaced
    # the retired considerations extraction never runs — no section, no report key
    assert "Valuable welfare considerations (LLM)" not in titles
    assert "moral_patient_reasons" not in report


def test_reasons_delivery_clamps_and_notes(tmp_path, stub_claude):
    # out-of-range / stringy scores are clamped to 0..JUDGE_SCORE_MAX ints
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    stub_claude(_judges_dispatch(
        delivery='{"delivery_quality": "120", "quality_note": "over the top"}'))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    pc = report["delivery"]["per_case"]["AW-0001"]
    assert pc["pipeline"]["score"] == audit_dad.JUDGE_SCORE_MAX
    assert pc["plain"]["score"] == audit_dad.JUDGE_SCORE_MAX


def test_reasons_delivery_judge_failure_is_counted_not_fatal(tmp_path, stub_claude):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    # a delivery reply that isn't a JSON object -> the response is skipped, run survives
    stub_claude(_judges_dispatch(delivery="not json at all"))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    # delivery absent (all its judge calls failed); the impact judge still ran
    assert "delivery" not in report
    assert report["welfare_impact"]["n_pipeline"] == 1


def test_reasons_delivery_dimension_grades_ride_along(tmp_path, stub_claude):
    # The judge grades the four Assess dimensions in the same call; they land
    # per-case and as report-level means, and an old-shaped reply without them
    # still carries the holistic score (backward tolerance).
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])

    def delivery(msg):
        if "P" * 500 in msg:
            return ('{"user_asks": ["a"], "user_raised": [], '
                    '"goal_responsiveness": 4, "proportionality": 5, "tone": 9, '
                    '"calibration": 8, "delivery_quality": 5, "quality_note": "withholds"}')
        return '{"delivery_quality": 7, "quality_note": "old shape, no dimensions"}'

    stub_claude(_judges_dispatch(delivery=delivery))
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    pc = report["delivery"]["per_case"]["AW-0001"]
    assert pc["pipeline"]["score"] == 5               # holistic, NOT the dim average (6.5)
    assert pc["pipeline"]["dimensions"] == {
        "goal_responsiveness": 4, "proportionality": 5, "tone": 9, "calibration": 8}
    assert "dimensions" not in pc["plain"]            # old-shaped reply tolerated
    assert report["delivery"]["dimensions"]["pipeline"]["goal_responsiveness"] == 4
    assert "plain" not in report["delivery"]["dimensions"]
    rows = {r["label"]: r for s in report["sections"] for r in s["rows"]}
    assert "dimension means (pipeline / plain)" in rows


def test_judges_eval_model_reaches_call_claude(tmp_path, stub_claude):
    # config `evals.judge_model` drives every judge call, falling back to the
    # global model; the report blocks record the effective model (viewer reads
    # it) and the pass cost.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 500, "B" * 250)])
    calls = stub_claude(_judges_dispatch())
    report = {}
    audit_dad.audit_judges(
        run, {"workers": 1, "model": "global-m",
              "evals": {"judge_model": "judge-m"}},
        report)
    assert {c["model"] for c in calls} == {"judge-m"}
    assert all(c["stage"] == "eval_audit_dad" for c in calls)
    assert report["delivery"]["judge_model"] == "judge-m"
    assert report["welfare_impact"]["judge_model"] == "judge-m"
    # the pass records its own cost (0.0 offline — no cost log), as a number and
    # a display row, so the viewer can show what --judges cost for this run
    assert isinstance(report["delivery"]["cost_usd"], (int, float))
    rows = {r["label"]: r for s in report["sections"] for r in s["rows"]}
    assert "pass cost (LLM calls)" in rows


def _impact_scope_win(pipe_marker):
    """Impact dispatcher: the pipeline arm wins patient_scope 95 vs 60, every
    other dimension equal — so exactly ONE (record, dimension) candidate."""
    def impact(user_message):
        scope = 95 if pipe_marker in user_message else 60
        return ('{"patient_scope": %d, "magnitude_sizing": 80, "counterfactual_impact": 80, '
                '"harm_contribution": 80, "epistemic_accuracy": 80, "bottom_line_coherence": 80, '
                '"welfare_impact": 85, "impact_note": "n"}' % scope)
    return impact


def test_showcase_selects_by_welfare_subdimension_with_verbatim_spans(tmp_path, stub_claude):
    # Nomination comes from the welfare judge's per-DIMENSION gaps (here
    # patient_scope 95 vs 60, everything else equal), gated on delivery not
    # sacrificed and comparable length. The judge's highlights are validated by
    # exact substring match against each response — a span that doesn't locate
    # is dropped, and the example ships with only the verbatim ones. Judge
    # calls run on the evals judge_model.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 260, "B" * 250)])

    def dispatch(user_message, **kw):
        if user_message.startswith("You are selecting a SHOWCASE example"):
            return ('{"fit": 9, "summary": "plain missed the bycatch", '
                    '"highlights": ["' + "P" * 12 + '", "NOT IN THE TEXT"], '
                    '"plain_highlights": ["' + "B" * 10 + '", "ALSO NOT THERE"]}')
        return _judges_dispatch(impact=_impact_scope_win("P" * 260))(user_message, **kw)

    calls = stub_claude(dispatch)
    report = {}
    cfg = {"workers": 1, "model": "global-m", "evals": {"judge_model": "judge-m"}}
    audit_dad.audit_judges(run, cfg, report)
    audit_dad.audit_showcase(run, cfg, report)

    examples = report["showcase"]["examples"]
    assert [e["dimension"] for e in examples] == ["patient_scope"]
    ex = examples[0]
    assert ex["label"] == audit_dad.SHOWCASE_DIMENSION_LABELS["patient_scope"]
    assert ex["highlights"] == ["P" * 12]              # non-verbatim span dropped
    assert ex["plain_highlights"] == ["B" * 10]        # validated against PLAIN text
    assert ex["summary"] == "plain missed the bycatch"
    assert ex["pipeline_response"] == "P" * 260 and ex["plain_response"] == "B" * 250
    assert ex["welfare_dimension"] == {"pipeline": 95, "plain": 60}
    assert ex["delivery_gap"] == 0 and ex["length_ratio"] == 1.04
    showcase_calls = [c for c in calls
                      if c["user_message"].startswith("You are selecting a SHOWCASE")]
    assert len(showcase_calls) == 1 and showcase_calls[0]["model"] == "judge-m"
    # the category brief names the winning dimension for the judge
    assert "IMPROVED PATIENT SCOPE" in showcase_calls[0]["user_message"]
    rows = {r["label"]: r for s in report["sections"] for r in s["rows"]}
    assert audit_dad.SHOWCASE_DIMENSION_LABELS["patient_scope"] in rows


def test_showcase_gates_length_ratio_and_non_english(tmp_path, stub_claude):
    # Two records with a clear patient_scope win, both ineligible: AW-0001 is
    # 2x plain's length (cap is 1.10x), AW-0002 is written in a non-Latin
    # script. No showcase judge call is even attempted — the gates are free.
    jp_pipe = "この回答は動物の福祉問題を検討します。" * 15
    jp_plain = "これは対照群の応答です。福祉には触れません。" * 15
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "P" * 500, "B" * 250),
        ("AW-0002", jp_pipe, jp_plain),
    ])

    def impact(user_message):
        scope = 95 if ("P" * 500 in user_message or jp_pipe[:25] in user_message) else 60
        return ('{"patient_scope": %d, "magnitude_sizing": 80, "counterfactual_impact": 80, '
                '"harm_contribution": 80, "epistemic_accuracy": 80, "bottom_line_coherence": 80, '
                '"welfare_impact": 85, "impact_note": "n"}' % scope)

    def dispatch(user_message, **kw):
        assert not user_message.startswith("You are selecting a SHOWCASE"), \
            "gated-out record must never reach the showcase judge"
        return _judges_dispatch(impact=impact)(user_message, **kw)

    stub_claude(dispatch)
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    audit_dad.audit_showcase(run, {"workers": 1}, report)
    assert report["showcase"]["examples"] == []


def test_record_in_english_checks_deal_and_script():
    # dealt cultural_setting naming another writing language loses eligibility
    # even when the sampled text is ASCII (Latin-script languages)
    assert audit_dad._record_in_english(
        {"cultural_setting": "France, written in French, with French idioms"},
        "Une reponse entierement en ASCII") is False
    assert audit_dad._record_in_english(
        {"cultural_setting": "United States, small town"}, "Plain English text") is True
    # non-Latin script fails the ASCII share check whatever the deal says
    assert audit_dad._record_in_english({}, "動物の福祉について" * 10) is False
    assert audit_dad._record_in_english({}, "") is True  # degenerate: no letters


def test_showcase_ships_nothing_when_no_span_locates(tmp_path, stub_claude):
    # An example whose every PIPELINE highlight fails the verbatim check is
    # skipped (fail-closed) — better no showcase than a mislocated highlight.
    run = _write_run_with_responses(tmp_path, [("AW-0001", "P" * 260, "B" * 250)])

    def dispatch(user_message, **kw):
        if user_message.startswith("You are selecting a SHOWCASE example"):
            return ('{"fit": 9, "summary": "s", "highlights": ["NOT IN THE TEXT"], '
                    '"plain_highlights": []}')
        return _judges_dispatch(impact=_impact_scope_win("P" * 260))(user_message, **kw)

    stub_claude(dispatch)
    report = {}
    audit_dad.audit_judges(run, {"workers": 1}, report)
    audit_dad.audit_showcase(run, {"workers": 1}, report)
    assert report["showcase"]["examples"] == []
    rows = {r["label"]: r for s in report["sections"] for r in s["rows"]}
    assert rows["examples selected"]["value"] == "0"


# --- report sections (the viewer's rendering contract) ---------------------

def test_sections_carry_rows_with_derived_verdicts():
    records = [
        {"prompt_id": "AW-0001", "user_message": "I've got a feature due friday and I can't decide."},
        {"prompt_id": "AW-0002", "user_message": "My neighbour keeps chickens in a small coop."},
    ]
    report = {}
    audit_dad.audit_skeletons(records, report)
    sec = report["sections"][0]
    assert sec["title"] == "Structural skeletons"
    by_label = {r["label"]: r for r in sec["rows"]}
    share = report["skeletons"]["produce_by_deadline_share"]
    assert by_label["produce-by-deadline share"]["verdict"] == audit_dad._verdict(share, 0.30, 0.50)
    assert by_label["produce-by-deadline share"]["value"].startswith(
        f"{report['skeletons']['produce_by_deadline']}/")
    assert by_label["families"]["verdict"] is None  # informational row, no threshold


def test_sections_accumulate_in_run_order():
    records = [{"prompt_id": "AW-0001", "user_message": "My cat sleeps a lot these days."}]
    report = {}
    audit_dad.audit_skeletons(records, report)
    audit_dad.audit_openers_closers(records, report)
    audit_dad.audit_locale_taxa(records, report)
    assert [s["title"] for s in report["sections"]] == [
        "Structural skeletons", "Openers & closers", "Locale / taxa plausibility"]


def test_every_section_carries_a_group_and_a_gloss(tmp_path):
    # group buckets the viewer's layout; gloss is the plain-language line under
    # each section title. New sections must ship both.
    records = [{"prompt_id": "AW-0001", "user_message": "My cat sleeps a lot."}]
    run = _write_run_with_responses(tmp_path, [("AW-0001", "A reply.", "Plain.")])
    report = {}
    audit_dad.audit_skeletons(records, report)
    audit_dad.audit_openers_closers(records, report)
    audit_dad.audit_unrealized_details(records, report)
    audit_dad.audit_locale_taxa(records, report)
    audit_dad.audit_lengths(run, report)
    audit_dad.audit_jargon(run, report)
    audit_dad.audit_response_lengths(run, report)
    audit_dad.audit_tracked_tics([], run, report)
    audit_dad.audit_tic_candidates(records, run, report)
    audit_dad.audit_lexical_diversity(records, report)
    audit_dad.audit_lexical(run, report)
    audit_dad.audit_structure(run, report)
    audit_dad.audit_response_openings(run, report)
    audit_dad.audit_library_selection(run, report)
    audit_dad.audit_library_coverage(run, report)
    for sec in report["sections"]:
        assert sec["group"] in ("prompt", "response", "library", "paid"), sec["title"]
        assert sec["gloss"], sec["title"]
    groups = {s["title"]: s["group"] for s in report["sections"]}
    assert groups["Structural skeletons"] == "prompt"
    assert groups["Insider-vocabulary leak (responses)"] == "response"
    assert groups["Reasoning-library selection (2a.5)"] == "library"


def test_skipped_sections_are_recorded_for_bare_file_input():
    # bare-file input (run_dir=None): every run-dir section records WHY it
    # carries no verdicts, and the summary data lands in the report
    report = {}
    audit_dad.audit_jargon(None, report)
    audit_dad.audit_response_lengths(None, report)
    audit_dad.audit_library_selection(None, report)
    skipped = report["skipped_sections"]
    assert [s["section"] for s in skipped] == [
        "Insider-vocabulary leak (responses)",
        "Response lengths (vs plain baseline)",
        "Reasoning-library selection (2a.5)"]
    assert all("bare-file input" in s["reason"] for s in skipped)
    # the skip rows themselves are unchanged (value 'skipped', note intact)
    assert all(sec["rows"][0]["value"] == "skipped" for sec in report["sections"])


def test_locale_flags_recorded_as_detail_lines():
    records = [
        {"prompt_id": "AW-0001", "taxa_subcategory": "fur animals (mink, foxes)",
         "cultural_setting": "the Caribbean", "user_message": "..."},
    ]
    report = {}
    audit_dad.audit_locale_taxa(records, report)
    sec = report["sections"][0]
    assert sec["rows"][0]["verdict"] == "BAD"
    assert len(sec["detail"]) == 1 and "AW-0001" in sec["detail"][0]


def test_lengths_section_rows_added_without_reprinting(tmp_path, capsys):
    msg = "Short. Two."
    run = _write_run(tmp_path, [
        {"prompt_id": "AW-0001", "length_class": "a short paragraph", "user_message": msg},
    ])
    report = {}
    audit_dad.audit_lengths(run, report)
    sec = report["sections"][0]
    by_label = {r["label"]: r for r in sec["rows"]}
    assert by_label["prompt lengths"]["value"].startswith("1 prompts")
    assert by_label["a short paragraph"]["value"] == (
        f"n=1, chars {len(msg)}-{len(msg)}, median {len(msg)}")
    # length is descriptive now: no band pass/fail row
    assert "records outside their band" not in by_label
    # rows mirror prompt_length_report's own printing — they must not re-print
    assert capsys.readouterr().out.count("prompt lengths") == 1


def test_carry_forward_keeps_paid_judges_on_offline_rerun():
    old_sec = {"title": "Welfare impact (LLM)", "rows": [{"label": "x"}]}
    old_report = {"delivery": {"n_pipeline": 10, "per_case": {}},
                  "welfare_impact": {"n_pipeline": 10, "per_case": {}},
                  "composite": {"arm_means": {"pipeline": 0.8}},
                  "showcase": {"examples": []},
                  "sections": [{"title": "Structural skeletons", "rows": []}, old_sec]}
    report = {"sections": [{"title": "Structural skeletons", "rows": []}]}
    assert audit_dad.carry_forward_judges(old_report, report) is True
    assert report["delivery"] == {"n_pipeline": 10, "per_case": {}}
    assert report["welfare_impact"]["n_pipeline"] == 10
    assert report["composite"]["arm_means"]["pipeline"] == 0.8
    # the carried section keeps its rows but takes the CURRENT gloss text
    carried = report["sections"][-1]
    assert carried["rows"] == [{"label": "x"}]
    assert carried["gloss"] == audit_dad._IMPACT_GLOSS
    # a retired considerations-era section is NOT carried forward
    assert all(s["title"] != "Moral-patient reasons (LLM)" for s in report["sections"])
    # nothing to carry -> report untouched
    fresh = {}
    assert audit_dad.carry_forward_judges({}, fresh) is False
    assert fresh == {}
    # a legacy report with ONLY considerations data carries nothing
    legacy = {"moral_patient_reasons": {"n": 10, "per_case": {}}, "sections": []}
    fresh2 = {}
    assert audit_dad.carry_forward_judges(legacy, fresh2) is False


# --- tracked tics & structural variation ----------------------------------

def test_tracked_tics_watchlist_counts_both_arms(tmp_path):
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "One thing worth naming about the barn.\n\nMore text here.",
         "I'd push back on that framing."),
        ("AW-0002", "Also worth naming: it's close.\n\nOther text.", "Plain reply."),
    ])
    report = {}
    audit_dad.audit_tracked_tics([], run, report)
    watch = report["tracked_tics"]["watch"]
    assert watch["worth naming"] == {"origin": "pipeline-origin", "surface": "response",
                                     "pipeline": 2, "plain": 0, "prompts": 0}
    assert watch["push back on"] == {"origin": "plain-origin", "surface": "response",
                                     "pipeline": 0, "plain": 1, "prompts": 0}
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    # worst pipeline-origin phrase at 2/2 -> derived verdict
    assert rows["worst pipeline-origin phrase"]["verdict"] == audit_dad._verdict(1.0, 0.20, 0.40)
    assert "worth naming" in rows["worst pipeline-origin phrase"]["value"]


def test_tracked_tics_count_the_prompt_surface_too(tmp_path):
    # Both surfaces the pipeline writes are audited: a watched phrase appearing
    # in the user prompts is counted there, with its own row and denominator.
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "A response with no watched phrase.", "Plain."),
        ("AW-0002", "Another ordinary response.", "Plain."),
    ])
    records = [{"prompt_id": "AW-0001",
                "user_message": "I'm almost certainly overthinking the barn plan."},
               {"prompt_id": "AW-0002",
                "user_message": "Almost certainly a small thing, but it nags."}]
    report = {}
    audit_dad.audit_tracked_tics(records, run, report)
    tt = report["tracked_tics"]
    assert tt["n_prompts"] == 2
    # every watched phrase is counted on BOTH surfaces regardless of the surface
    # label it was promoted for — here a response-surface phrase in the prompts
    assert tt["watch"]["almost certainly"]["prompts"] == 2   # counted in the prompts
    assert tt["watch"]["almost certainly"]["pipeline"] == 0  # absent from the responses
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["prompts scanned"]["value"] == "2"
    assert "almost certainly" in rows["worst phrase in the prompts"]["value"]


def test_tracked_tics_prompt_row_says_none_when_prompts_are_clean(tmp_path):
    run = _write_run_with_responses(tmp_path, [("AW-0001", "Ordinary response.", "Plain.")])
    records = [{"prompt_id": "AW-0001", "user_message": "A plain question about the barn."}]
    report = {}
    audit_dad.audit_tracked_tics(records, run, report)
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["worst phrase in the prompts"]["value"] == "none"


def test_load_tic_surfaces_defaults_to_response():
    # every phrase promoted before the prompt surface existed reads as
    # "response"; the loader never leaves a watched phrase unlabelled
    watch, _ = audit_dad.load_tic_lists()
    surfaces = audit_dad.load_tic_surfaces()
    all_phrases = {ph for phrases in watch.values() for ph in phrases}
    assert all_phrases <= set(surfaces)
    assert set(surfaces.values()) <= {"prompt", "response"}
    assert surfaces["worth naming"] == "response"       # default when unlabelled


def test_load_tic_surfaces_honors_an_explicit_prompt_label(tmp_path):
    # The prompt surface is a loader capability, not a property of whatever is
    # on the live watchlist (which is response-only as of the 2026-07-30
    # demotions), so drive it from a written file rather than the real one.
    f = tmp_path / "tics.yaml"
    f.write_text("watch:\n  pipeline-origin:\n"
                 "    - {phrase: prompt side, surface: prompt}\n"
                 "    - {phrase: response side}\n", encoding="utf-8")
    surfaces = audit_dad.load_tic_surfaces(f)
    assert surfaces == {"prompt side": "prompt", "response side": "response"}


def test_load_tic_lists_reads_watch_and_ignore():
    # Derived from the real evals/tics.yaml — asserts the loader shape and that
    # kept register tics are present, not hardcoded counts.
    watch, ignore = audit_dad.load_tic_lists()
    assert "worth naming" in watch["pipeline-origin"]       # kept performed-candor tic
    assert "capacity to suffer" in watch["pipeline-origin"]
    assert "push back on" in watch["plain-origin"]          # kept plain-origin tic
    # promoted 2026-07-30 off the archetype200 run
    assert "load bearing" in watch["pipeline-origin"]       # scaffold bleed
    assert "the honest" in watch["plain-origin"]            # plain's biggest habit
    # demoted 2026-07-27 (dead on Opus 5) and 2026-07-30 (under 5% in both arms)
    # — off watch but NOT ignore-listed, so the candidate screen re-surfaces them
    for demoted in ("gut check", "cuts both ways", "overthinking", "highest leverage",
                    "going back and forth"):
        assert all(demoted not in phrases for phrases in watch.values()), demoted
        assert demoted not in ignore, demoted
    assert isinstance(ignore, set)
    # generic autonomy-coda phrasings were demoted to ignore once the coda
    # became a tracked rhetorical move, so the phrase audit stops double-counting
    # them and the candidate queue won't re-surface them...
    assert {"you're the one", "yours to", "is your call"} <= ignore
    # ...and the verbatim engrams that were kept alongside their moves
    # ("genuinely yours", "cuts both ways") were themselves demoted on
    # 2026-07-30 for falling under 5% in both arms — the moves still track the
    # wider gambits, so nothing is lost, and the engrams can re-earn a place.
    assert "genuinely yours" not in watch["pipeline-origin"]
    assert {m["name"] for m in audit_dad.load_moves()} >= {"cuts-both-ways", "autonomy-coda"}


def test_tic_candidates_surfaces_rare_over_represented_phrase(tmp_path):
    # A rare-in-English phrase repeated across pipeline responses but absent
    # from the plain arm must surface as a response candidate and be persisted.
    tic = "zorble widget"
    pairs = [(f"AW-000{i}",
              f"We should weigh the {tic} here." if i < 4 else "A plain point.",
              "An ordinary baseline reply.") for i in range(6)]
    run = _write_run_with_responses(tmp_path, pairs)
    records = [{"prompt_id": p, "user_message": f"dilemma {p}"} for p, _, _ in pairs]
    report = {}
    audit_dad.audit_tic_candidates(records, run, report)
    resp = [c["phrase"] for c in report["tic_candidates"]["response"]]
    assert any(tic in g or g in tic for g in resp)
    lines = (run / "audit" / "tic_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(tic in ln for ln in lines)


def test_tic_candidates_mirror_screen_surfaces_plain_arm_tic(tmp_path):
    # The reverse direction: a rare phrase repeated across PLAIN responses but
    # absent from the pipeline arm must surface under the "plain" arm and be
    # persisted — plain Claude's tics get the same discovery path.
    tic = "flombax lever"
    pairs = [(f"AW-000{i}", "An ordinary pipeline reply.",
              f"I'd examine the {tic} first." if i < 4 else "A plain point.")
             for i in range(6)]
    run = _write_run_with_responses(tmp_path, pairs)
    records = [{"prompt_id": p, "user_message": f"dilemma {p}"} for p, _, _ in pairs]
    report = {}
    audit_dad.audit_tic_candidates(records, run, report)
    plain = [c["phrase"] for c in report["tic_candidates"]["plain"]]
    assert any(tic in g or g in tic for g in plain)
    # and NOT as a pipeline-side (response) candidate
    resp = [c["phrase"] for c in report["tic_candidates"]["response"]]
    assert not any(tic in g or g in tic for g in resp)
    lines = (run / "audit" / "tic_candidates.jsonl").read_text(encoding="utf-8").splitlines()
    assert any(tic in ln and '"plain"' in ln for ln in lines)


def test_tic_candidates_excludes_watched_phrases(tmp_path):
    # A phrase already on the watchlist must NOT reappear as a candidate.
    watched = "capacity to suffer"  # present in evals/tics.yaml
    pairs = [(f"AW-000{i}",
              f"Their {watched} is real." if i < 4 else "Plain.",
              "Baseline.") for i in range(6)]
    run = _write_run_with_responses(tmp_path, pairs)
    records = [{"prompt_id": p, "user_message": "x"} for p, _, _ in pairs]
    report = {}
    audit_dad.audit_tic_candidates(records, run, report)
    resp = [c["phrase"] for c in report["tic_candidates"]["response"]]
    assert watched not in resp


def test_tracked_tics_empty_watchlist_bucket_degrades_to_no_row(tmp_path, monkeypatch):
    # an emptied origin bucket (e.g. after a watchlist prune) must not crash —
    # the worst-phrase rows simply don't emit
    monkeypatch.setattr(audit_dad, "load_tic_lists",
                        lambda: ({"pipeline-origin": [], "plain-origin": []}, set()))
    run = _write_run_with_responses(tmp_path, [("AW-0001", "Some reply.", "Plain.")])
    report = {}
    audit_dad.audit_tracked_tics([], run, report)
    labels = [r["label"] for r in report["sections"][0]["rows"]]
    assert "responses scanned" in labels
    assert "worst pipeline-origin phrase" not in labels
    assert "worst plain-origin phrase (plain arm)" not in labels


def test_tracked_tics_watchlist_detail_capped_at_recurring_12(tmp_path, monkeypatch):
    # every pipeline-origin watch phrase fires in both responses (>=2 hits each)
    # -> more than 12 eligible lines -> capped at 12 plus a remainder line
    phrases = [f"tic phrase number {i}" for i in range(15)]
    monkeypatch.setattr(audit_dad, "load_tic_lists",
                        lambda: ({"pipeline-origin": phrases, "plain-origin": []}, set()))
    all_phrases = ". ".join(phrases) + "."
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", all_phrases, "Plain."), ("AW-0002", all_phrases, "Plain too."),
    ])
    report = {}
    audit_dad.audit_tracked_tics([], run, report)
    detail = report["sections"][0].get("detail") or []
    watch_lines = [d for d in detail if d.startswith("[")]
    assert len(watch_lines) == 12
    assert any(d.startswith("… (+") for d in detail)
    # the report JSON still carries every phrase's counts uncapped
    assert len(report["tracked_tics"]["watch"]) == 15


def test_structure_shapes_and_collapse_verdict(tmp_path):
    same = "Para one.\n\n- a bullet\n- another\n\nPara three."
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", same, "One short plain paragraph?"),
        ("AW-0002", same, "1. first\n2. second\n\n**Verdict:**\n\ndone."),
    ])
    report = {}
    audit_dad.audit_structure(run, report)
    p = report["structure"]["pipeline"]
    b = report["structure"]["plain"]
    assert p["distinct"] == 1 and p["top_share"] == 1.0
    assert p["bullets"] == 1.0 and p["numbered"] == 0.0
    assert b["distinct"] == 2
    assert b["numbered"] == 0.5 and b["headed"] == 0.5 and b["ends_question"] == 0.5
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["top shape share (pipeline)"]["verdict"] == audit_dad._verdict(1.0, 0.30, 0.50)
    assert "3-5 paras" in rows["top shape share (pipeline)"]["note"]


def test_tracked_tics_and_structure_skip_cleanly_for_bare_input():
    report = {}
    audit_dad.audit_tracked_tics([], None, report)
    audit_dad.audit_structure(None, report)
    audit_dad.audit_tic_candidates([], None, report)
    assert "tracked_tics" not in report and "structure" not in report
    assert "tic_candidates" not in report
    assert [s["title"] for s in report["sections"]] == [
        "Tracked tics (prompts + responses)", "Structural variation (responses)",
        "Tic candidates (review queue)"]


# --- response openings ------------------------------------------------------

def _write_drafts(run, drafts):
    (run / "step2").mkdir(exist_ok=True)
    for r in drafts:
        utils.append_jsonl(r, run / "step2" / "responses.jsonl")


def test_response_openings_families_spread_and_verdict(tmp_path):
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "Here's the thing about the barn. More text.", None),
        ("AW-0002", "The numbers in your message decide this one. More.", None),
    ])
    _write_drafts(run, [
        {"prompt_id": "AW-0001", "sample_index": 0,
         "assistant_response": "Here's the thing about the farm. More."},
        {"prompt_id": "AW-0001", "sample_index": 1,
         "assistant_response": "You've basically answered your own question. More."},
        {"prompt_id": "AW-0002", "sample_index": 0,
         "assistant_response": "Here's what I think is going on. More."},
    ])
    report = {}
    audit_dad.audit_response_openings(run, report)
    ro = report["response_openings"]
    assert ro["drafts"]["families"] == {"heres-the-x": 2, "already-answered": 1}
    # AW-0001's two samples opened through different families
    assert ro["drafts"]["case_spread"] == {"AW-0001": "2/2 distinct"}
    # finals read via the step3 rewrites join
    assert ro["finals"]["n"] == 2
    assert ro["finals"]["families"] == {"heres-the-x": 1, "other": 1}
    rows = {(s["title"], r["label"]): r for s in report["sections"] for r in s["rows"]}
    drafts_top = rows[("Response openings (drafts)", "top non-'other' opener family")]
    assert "heres-the-x" in drafts_top["value"]
    assert drafts_top["verdict"] == audit_dad._verdict(2 / 3, 0.30, 0.50)


def test_response_openings_hint_echo_verdict(tmp_path):
    card = "open with the factual crux the case turns on"
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    _write_drafts(run, [
        {"prompt_id": "AW-0001", "sample_index": 0, "opening_hints": card,
         "assistant_response": "The factual crux here decides everything. More."},
        {"prompt_id": "AW-0002", "sample_index": 0, "opening_hints": card,
         "assistant_response": "Start from the numbers in the report. More."},
    ])
    report = {}
    audit_dad.audit_response_openings(run, report)
    ro = report["response_openings"]
    assert ro["drafts"]["hint_echo"] == {card: (1, 2)}
    assert ro["drafts"]["hint_draws"] == {card: 2}
    rows = {(s["title"], r["label"]): r for s in report["sections"] for r in s["rows"]}
    echo = rows[("Response openings (drafts)", "hint-echo (card wording in opener)")]
    assert echo["value"] == "1/2 draws"
    assert echo["verdict"] == audit_dad._verdict(0.5, 0.0, 0.2)  # wording leaked
    # no finals in this run -> calm zero, and no echo row on the finals section
    assert ro["finals"] == {"n": 0}
    assert ("Response openings (finals)",
            "hint-echo (card wording in opener)") not in rows


def test_response_openings_calm_without_responses_and_bare_input(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    report = {}
    audit_dad.audit_response_openings(run, report)
    assert report["response_openings"] == {"drafts": {"n": 0}, "finals": {"n": 0}}
    report2 = {}
    audit_dad.audit_response_openings(None, report2)  # bare-file input
    assert "response_openings" not in report2
    assert [s["title"] for s in report2["sections"]] == [
        "Response openings (drafts)", "Response openings (finals)"]


# --- lexical diversity & library coverage -----------------------------------

class TestLexicalMetrics:
    def test_distinct_n_known_values(self):
        # "a b a b": 4 unigrams 2 unique -> 0.5; 3 bigrams 2 unique -> 2/3
        assert audit_dad.distinct_n(["a b a b"], 1) == 0.5
        assert audit_dad.distinct_n(["a b a b"], 2) == pytest.approx(2 / 3)
        # pooled across texts: cross-text repeats count against the score
        assert audit_dad.distinct_n(["a b", "a b"], 1) == 0.5

    def test_self_bleu_identical_high_disjoint_low(self):
        same = ["the quick brown fox jumps over the lazy dog today"] * 3
        assert audit_dad.self_bleu(same) == pytest.approx(1.0, abs=1e-6)
        disjoint = ["alpha beta gamma delta epsilon zeta eta theta",
                    "one two three four five six seven eight",
                    "red orange yellow green blue indigo violet mauve"]
        assert audit_dad.self_bleu(disjoint) < 0.05

    def test_self_bleu_degenerate_sizes(self):
        assert audit_dad.self_bleu([]) == 0.0
        assert audit_dad.self_bleu(["only one text"]) == 0.0


def test_lexical_section_reports_both_arms(tmp_path):
    run = _write_run_with_responses(tmp_path, [
        ("AW-0001", "the fox ran far " * 20, "a plain answer about hens " * 20),
        ("AW-0002", "the fox ran far " * 20, "another plain reply about barns " * 20),
    ])
    report = {}
    audit_dad.audit_lexical(run, report)
    lex = report["lexical"]
    assert lex["pipeline"]["n"] == 2 and lex["plain"]["n"] == 2
    # two identical pipeline texts -> Self-BLEU 1.0; distinct plain texts lower
    assert lex["pipeline"]["self_bleu"] == pytest.approx(1.0, abs=1e-6)
    assert lex["plain"]["self_bleu"] < lex["pipeline"]["self_bleu"]
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert "pipeline" in rows["Self-BLEU"]["value"] and "plain" in rows["Self-BLEU"]["value"]


def test_library_coverage_counts_fires_and_never_selected(tmp_path):
    from dad_pipeline import reasoning_library
    all_ids = [str(e) for e in reasoning_library.all_ids(reasoning_library.load("prompts/dad"))]
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    (run / "step2").mkdir()
    utils.append_jsonl({"prompt_id": "AW-0001", "entry_ids": ["C1", "C2", "C1"]},
                       run / "step2" / "scopes.jsonl")
    utils.append_jsonl({"prompt_id": "AW-0002", "entry_ids": ["C1"]},
                       run / "step2" / "scopes.jsonl")
    report = {}
    audit_dad.audit_library_coverage(run, report)
    cov = report["library_coverage"]
    assert cov["library_size"] == len(all_ids)
    assert cov["fires"]["C1"] == 2      # per-case dedupe: C1 twice in one case = 1
    assert cov["fires"]["C2"] == 1
    assert cov["used"] == 2
    assert set(cov["never_selected"]) == set(all_ids) - {"C1", "C2"}
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["most-selected entry"]["value"] == "C1 in 2/2 cases"
    # 2 cases: below the 20-case bar, so no verdict — just the caveat note
    assert rows["coverage (selected at least once)"]["verdict"] is None
    assert "20+ cases" in rows["coverage (selected at least once)"]["note"]


def test_library_coverage_verdict_attaches_at_scale(tmp_path):
    from dad_pipeline import reasoning_library
    all_ids = [str(e) for e in reasoning_library.all_ids(reasoning_library.load("prompts/dad"))]
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    (run / "step2").mkdir()
    # 20 cases that between them select every entry -> 100% coverage
    for i in range(20):
        utils.append_jsonl({"prompt_id": f"AW-{i:04d}",
                            "entry_ids": all_ids[i::20] or [all_ids[0]]},
                           run / "step2" / "scopes.jsonl")
    report = {}
    audit_dad.audit_library_coverage(run, report)
    rows = {r["label"]: r for r in report["sections"][0]["rows"]}
    assert rows["coverage (selected at least once)"]["verdict"] == \
        audit_dad._verdict(1.0, 0.85, 0.60, higher_better=True)


def test_library_coverage_detail_lines_are_capped(tmp_path):
    from dad_pipeline import reasoning_library
    all_ids = [str(e) for e in reasoning_library.all_ids(reasoning_library.load("prompts/dad"))]
    assert len(all_ids) > 25, "cap test needs a library bigger than both caps"
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    (run / "step2").mkdir()
    # one case fires 11 entries (> the 10-line fires cap); the rest are never
    # selected (> the 15-id cap)
    utils.append_jsonl({"prompt_id": "AW-0001", "entry_ids": all_ids[:11]},
                       run / "step2" / "scopes.jsonl")
    report = {}
    audit_dad.audit_library_coverage(run, report)
    detail = report["sections"][0]["detail"]
    fires_line = next(d for d in detail if d.startswith("fires:"))
    never_line = next(d for d in detail if d.startswith("never selected:"))
    assert "(+1 more)" in fires_line                      # 11 fired, 10 shown
    assert f"(+{len(all_ids) - 11 - 15} more)" in never_line
    # the report JSON keeps the full picture uncapped
    assert len(report["library_coverage"]["never_selected"]) == len(all_ids) - 11


def test_library_coverage_calm_without_step2(tmp_path):
    run = _write_run(tmp_path, [{"prompt_id": "AW-0001", "user_message": "hi"}])
    report = {}
    audit_dad.audit_library_coverage(run, report)
    assert report["library_coverage"] == {"n_cases": 0}


class TestEffectiveNumber:
    def test_anchors(self):
        assert audit_dad.effective_number([10] * 10) == pytest.approx(10.0)
        assert audit_dad.effective_number([100]) == pytest.approx(1.0)
        assert audit_dad.effective_number([5, 5]) == pytest.approx(2.0)
        assert audit_dad.effective_number([]) == 0.0

    def test_reads_whole_distribution_not_just_top(self):
        # same 40% top-share, very different variety
        spread = audit_dad.effective_number([40, 10, 10, 10, 10, 10, 10])
        lumpy = audit_dad.effective_number([40, 40, 20])
        assert spread == pytest.approx(5.74, abs=0.01)
        assert lumpy == pytest.approx(2.87, abs=0.01)

    def test_reported_in_skeleton_and_structure_sections(self, tmp_path):
        records = [{"prompt_id": f"AW-{i}", "user_message": m} for i, m in enumerate([
            "My cat sleeps all day.", "We keep bees in the yard.",
            "I've got a report due friday on the hens."])]
        report = {}
        audit_dad.audit_skeletons(records, report)
        assert report["skeletons"]["effective_families"] > 1
        rows = {r["label"] for s in report["sections"] for r in s["rows"]}
        assert "effective families" in rows

        run = _write_run_with_responses(tmp_path, [
            ("AW-0001", "One paragraph.", "x"), ("AW-0002", "- a\n- b\n\ntwo", "y")])
        report2 = {}
        audit_dad.audit_structure(run, report2)
        assert report2["structure"]["pipeline"]["effective_shapes"] == pytest.approx(2.0)
