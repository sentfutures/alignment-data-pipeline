# Code Quality Report v2 — alignment-data-pipeline

**Date:** 2026-07-29 · **Scope:** full repo at `main@464c5bd` (all source, tests, CI, docs; committed `outputs/` scanned for hygiene)
**Supersedes:** the 2026-07-10 report (audited `cf8d91e`, reconciled 07-13) — 206 commits and 26 merged PRs ago. This v2 both tracks the fate of every original finding and audits the ~11,000 lines of new or rewritten source the original never saw.
**Purpose:** pre-delivery quality assessment with prioritized recommendations, ahead of handing the pipeline to a frontier lab.

## How this audit was produced

1. **Mechanical baseline re-run** — ruff, `ruff format --check`, mypy, `pytest --cov`, vulture, dependency review at `464c5bd` (report-only; nothing modified). July→now deltas in Appendix A.
2. **Fate ledger** — every July finding re-checked against today's code (file-content probe + lead-reviewer spot-reads of the non-trivial claims).
3. **Six charter reviews** of the post-July code, run as parallel agents: SDF matrix architecture, DAD rebuild, the audit_dad evidence subsystem, the publish_hf delivery path, shared-infrastructure growth, and delivery/process.
4. **Adversarial verification** — every high finding was independently attacked by two verifier agents (refute lens + reachability lens), every medium by one; low/info pass through unverified by design. A usage-limit outage killed 11 verifiers mid-run; the workflow was resumed and **all verifications completed** — final tally: 21 confirmed, 9 confirmed with corrections (corrected wording used here), 20 unverified low/info, **zero refuted**.

**Verification legend:** ✓✓ adversarially confirmed · ✓c confirmed with corrections · ✓m verified by the lead reviewer reading the cited code · △ unverified (low/info — probable, not proven).

---

## Executive summary

**The trajectory is strongly positive, and the new code is better than the old.** Since July the repo has grown from 6.8k to 16.8k source LOC while test count grew faster (295 → 742, all green; non-viewer coverage 76% → **85%**). The July report's two worst findings — layer 4 rewriting fail-open and layer 3 truncation blindness — were **fixed outright** in the SDF matrix rebuild, with exactly the recommended pattern (stop_reason fail-closed, skip-without-checkpoint, poison-call isolation, a systemic-failure abort). The parser fragmentation was consolidated into tested `shared/utils` helpers. The three headline new subsystems arrive well-engineered: the matrix machinery genuinely delivers composition-by-construction (verified against its own contract tests), `publish_hf.py` matches its documented safety contract on every headline claim checked, and `audit_dad` largely designed out the "parse failures counted as clean" failure class the July audit flagged in its SDF sibling.

The residual risk now clusters in four places:

1. **The delivery surfaces themselves.** `publish_hf` can `rmtree` a *different paid run directory* handed to `--staging-dir` (high, confirmed); a re-publish with an existing `--tag` silently leaves the tag pointing at the old — possibly wrong — data while printing success; and three committed audit reports (which publish verbatim to the public HF dataset) embed a contributor's home path.
2. **A systematic truncation/refusal gate gap on the response side.** DAD 2b, step 3, and the baseline arm reject only `stop_reason == "max_tokens"`; a mid-stream refusal cut (documented as occurring on this pipeline's core topics) or any subscription-backend stop value ships a half-finished argument into training records. The SDF layers already accept-only-on-`end_turn`; the DAD response stages never got the same fix.
3. **Evidence integrity in the audit subsystem.** On multi-sample runs the audit silently examines one sample per prompt while claiming full coverage; three judge-parse fallbacks silently change metric semantics instead of counting as failures; and the paid auto-fired `--reasons` pass has no checkpointing.
4. **The unfixed July carry-overs.** Checkpoint/JSONL crash-safety, the zero-tested eval judges, `--run-id` validation, pref prompt freezing, and the entire process/tooling tier (no linter, formatter, type-check, coverage gate, lockfile) are byte-for-byte where July left them — while the codebase they protect has doubled.

Nothing found suggests systemic quality decay — the opposite. But the highest-severity items are now concentrated exactly where a frontier-lab handoff is most sensitive: what gets published, and whether the published evidence describes the data.

---

## Fate of the July 10 findings

| July finding | Status on `main@464c5bd` | Note |
|---|---|---|
| 1.1 Layer-4 rewrite fails open (top finding) | **FIXED** ✓✓ | Matrix rebuild: stop_reason fail-closed, skip-without-checkpoint, failure log, SystemExit backstop — but its failure branches are untested (→ Tier 2.7) |
| 1.2 Layer-5 {5,5,5} sentinel | **PARTIAL** ✓✓ | Sentinel is now deliberate, documented, test-pinned — still threshold-coupled; two sharper confirmed defects remain (→ Tier 1.8) |
| 1.3 Checkpoint/JSONL crash-safety | **STILL LIVE** ✓m | All three sub-items verbatim (Checkpoint now at `utils.py:437-453`) |
| 1.4 Step-1 batch loss + refine fail-open | **PARTIAL / WORSE** ✓c | Refine retry + failure logs + `refine_failed` stamp landed; but the rebuild composed **three** paid fan-outs (1b+1c+1d) into one in-memory window (→ Tier 1.4) |
| 1.5 `strict=False` parsers | **FIXED** | Consolidated into `extract_json_object/array(recover=True)`, well tested |
| 1.6 Layer-3 truncation | **FIXED** ✓✓ | Rebuild; untagged fallback removed |
| 1.7 Checkpoint-only done-detection | **REPLACED, new gap** ✓✓ | All stages now cross-check the output jsonl — but the conjunction silently *drops* a checkpointed id whose record is missing (→ Tier 1.12) |
| 1.8 Eval judges (score_sdf/score_dad) | **STILL LIVE** ✓m | `score_sdf.py` unchanged (zero-sentinel + mark_done, 0% tests); `score_dad.py` parked with the defect intact. The de-facto evidence surface is now audit_dad — which has its own confirmed issues (→ Tier 1.6–1.7) |
| 1.9 `--run-id` without `--resume`; `--layer` choices | **STILL LIVE** ✓m | Unchanged in all three orchestrators |
| 1.10 Pref resume prompt freezing | **STILL LIVE** ✓m | `pref_pipeline/` byte-identical since July |
| 2.1 Parser consolidation | **LARGELY DONE** | Shared salvage/shape-validating parsers, tested; step-1 keeps thin wrappers |
| 2.2 `resumable_stage()` helper | **NOT DONE** ✓m | The idiom is now hand-rolled in more places than July |
| 2.4 Packaging (`[project]`, kill sys.path hack) | **NOT DONE** ✓✓ | pyproject.toml still 7 lines |
| 2.5 step1_dilemmas split | **NOT DONE** | Now 1,015 lines; parsers extracted, everything else grew (→ Tier 2.4) |
| 2.6 Risk-inverted test gaps | **MOSTLY DONE** | 742 tests; new-code coverage 88–100%; remaining gaps are again exactly the failure paths (→ Tier 2.7) |
| 3.1–3.5 Tooling/process tier | **ALL STILL ABSENT** ✓c | No ruff/format/mypy/coverage/lockfile/CI gate/review-prompt policy block (→ Tier 3.1) |
| 3.6 README constitution claim | **FIXED** ✓m | README now names the calls actually made |

---

## Tier 1 — Fix before delivery

### 1.1 `publish_hf --staging-dir` can silently `rmtree` a different paid run directory — ✓✓ HIGH · S
**`evals/publish_hf.py:145`**
`stage_run` wipes any pre-existing staging target unconditionally; the guard above it only refuses the *input* run's own directories. Tab-complete the wrong path — e.g. a sibling `outputs/sdf/runs/<other-run>` — and the tool recursively deletes an uncommitted run worth $50–500 of paid calls before producing any output. **This happens under `--dry-run` too**, which the operator reasonably believes is safe.
**Fix:** drop a marker file (`.publish_hf_staging`) into staging roots the tool creates and only rmtree when the marker is present; add regression tests mirroring the existing guard tests.

### 1.2 DAD response stages accept refusal-truncated output into training records — ✓✓ HIGH · S (+ ✓✓ sibling)
**`dad_pipeline/step2_responses.py:407`, `step3_rewrite.py:91`, `baseline.py:64`**
The gates reject only `not response or stop_reason == "max_tokens"`. Step 1a's own comments document Opus's refusal classifier cutting streams mid-response on insect-welfare topics; a refusal-cut 2b draft (non-empty text, `stop_reason="refusal"`) passes the gate, is checkpointed, and step 3 fluently rewrites the half-finished argument into a plausible training record — silently weaker data on exactly the corpus's core topic. The confirmed sibling finding (`shared/api.py:585`, ✓✓): on the `claude_code`/`auto` backends the CLI applies its own output cap and can return stop values (including `None`) that `== "max_tokens"` never matches, so the same gates fail open for ordinary truncation on subscription-served dev runs.
**Fix:** flip these gates to accept-only-on `stop_reason in ("end_turn", "stop_sequence")` — the polarity the SDF layers already use — or normalize unknown stop values to a rejected sentinel inside `call_claude`. One stub test per stage.

### 1.3 gid registry can silently collide or renumber across runs, merges, and crashes — ✓✓ HIGH · M
**`dad_pipeline/id_registry.py:86`**
A corrupt/conflicted `outputs/dad/id_registry.json` (a git-tracked file shared across runs and contributors) is silently swallowed and every id kind restarts at 1 — re-minting P-/R-/E- gids already stamped on committed runs, so audit and diversity reports join different artifacts under one gid. `save()` is a non-atomic full overwrite, and stages stamp gids onto records *before* the registry is saved. The reset-on-corrupt behavior is currently test-pinned as intended.
**Fix:** fail loudly on an unparseable registry (flip that test deliberately, per CLAUDE.md's spec-first rule); atomic save; save the registry with each stamped batch; document that concurrent DAD runs share one registry.

### 1.4 One step-1 pass holds three paid fan-outs in memory; retry budgets reset on resume — ✓c HIGH · M
**`dad_pipeline/step1_dilemmas.py:817-1002`**
July's finding covered 1b drafts in memory; the rebuild composed 1b drafting + 1c gating + 1d refining into the same window — under shipped defaults (gate and refine enabled, count=40, Opus models) roughly ~120 paid calls per pass held in memory until the serial assembly loop writes the first record. A crash in the 1d fan-out discards and re-bills all of it. Separately, `draft_attempts`/`gate_attempts`/`gate_feedback` are process-local, so budgets reset and gate feedback is lost on `--resume` (extra spend, weaker redrafts).
**Fix:** checkpoint accepted 1b drafts to a `drafts_pending.jsonl` consumed at assembly; rebuild attempt budgets on resume from the failure logs that already exist (`draft_failures.jsonl`, `gate.jsonl`).

### 1.5 Re-publish with an existing `--tag` silently leaves the tag on the old commit — ✓✓ MEDIUM · S
**`evals/publish_hf.py:558`**
`create_tag(..., exist_ok=True)` swallows the 409 and does **not** move the tag (verified against installed huggingface_hub); the script then prints `Tag: <tag>` as if applied. The motivating scenario in the code's own comment — republishing after a typo'd `--input` — is exactly the case where a lab pinning `revision=<tag>` gets the wrong run's data forever, with the tool's output claiming otherwise.
**Fix:** on tag-exists, compare its target with the fresh commit; warn loudly and require an explicit `--retag` (delete+recreate) when they differ. Never print `Tag:` unmoved.

### 1.6 Contributor home paths in committed audit reports, uploaded verbatim to the public HF dataset — ✓✓ MEDIUM · S
**`evals/diversity.py:745` (also `audit_dad.py:2919`, `audit_sdf.py:561`)**
All three report writers record `"input": str(args.input)`; three committed files already carry `/Users/<contributor>/...` publicly on GitHub, and `publish_hf` stages `audit/*.json` verbatim to the Hub.
**Fix:** record run-id-relative paths; add a belt-and-braces sanitizer in `stage_run`; scrub the three committed files; check whether the published dataset already carries a path and re-publish if so.

### 1.7 Audit evidence integrity: multi-sample runs audited 1/per_prompt while claiming full coverage; judge-parse slips silently change metric semantics; paid pass unchecked — ✓✓ ×3 MEDIUM · S–M
**`evals/audit_dad.py:664, 2019, 1936`**
(a) `_final_by_prompt_id` keeps one record per prompt_id, so on a `per_prompt > 1` run every response-side section — including the paid considerations/delivery headline numbers — describes an arbitrary half of the delivered corpus with no flag. (b) The consolidation-failure fallback silently switches "corpus-distinct considerations" from LLM-consolidated to exact-string dedupe (inflating the pipeline-vs-plain lift ~3× in the failure case), and survival/typing parse slips are similarly uncounted. (c) The auto-fired `--reasons` pass (~370 LLM calls at default scale) accumulates everything in memory; an interruption re-bills all of it.
**Fix:** key the join by record_id (or surface an "N of M audited" warning row); count every fallback as a visible event with a report field; persist per-item judge results append-only and skip already-judged items.

### 1.8 Layer 5 remains the odd stage out: no stop_reason check, unvalidated judge scores — ✓✓ ×2 MEDIUM · S
**`sdf_pipeline/layer5_score.py:56, 80`**
Truncated/refused judge replies fall into the checkpointed 5/5/5 sentinel (a permanent silent drop of a paid draft+rewrite under the default threshold — the sentinel's "retry adds nothing" rationale doesn't hold for this class). And scores are used raw: a string `"8"` wedges the gate with a TypeError *after* checkpointing — the run crashes identically on every resume until someone hand-edits `scores.jsonl`; missing keys silently zero-score.
**Fix:** `return_stop_reason=True` + skip-without-checkpoint on non-`end_turn` (the pattern its three sibling stages already use); `int(...)`-coerce and range-check the three scores at parse time.

### 1.9 Partial `prompts.jsonl` reused on resume, silently shrinking the run — ✓✓ MEDIUM · S
**`sdf_pipeline/layer12_plan.py:35`**
File existence is treated as completeness; a crash mid-write (disk-full is realistic) leaves a well-formed short file that every resume silently accepts — the corpus under-delivers *and* the exact-marginals composition guarantee quietly no longer holds.
**Fix:** verify `len(prompts) == sdf["n_prompts"]` on reload (composition is seeded — recomposing is safe; say so in the error), or write via temp file + `os.replace`.

### 1.10 Gemini embeddings never validated against the batch — ✓✓ MEDIUM · S
**`shared/embeddings.py:201`**
The Gemini leg zips returned vectors against requested texts with no count/order check (the OpenAI leg is defensive and test-pinned). A short or reordered response misattributes embeddings to neighboring documents and **poisons the persistent embedding cache**, so the delivered diversity evidence names the wrong near-duplicates even on re-runs.
**Fix:** raise when `len(vectors) != len(batch)` or an entry lacks `values`; mirror the OpenAI-leg test.

### 1.11 Carried from July, still live — ✓m
Checkpoint/JSONL crash-safety (atomic write + tolerant load; the fix is unchanged from the July report); `score_sdf.py` zero-sentinel checkpointing at 0% coverage; `--run-id` without `--resume` minting fresh runs; pref prompt-set freezing. All four remain exactly as written up in July — the fixes there still apply verbatim.

### 1.12 Checkpointed id with missing jsonl record is silently dropped in all four SDF stages — ✓✓ MEDIUM · S
**`sdf_pipeline/layer4_rewrite.py:38-41` (+ layer12/3/5)**
The new done-detection conjunction (`not in existing and not checkpoint.is_done(...)`) can only ever convert "retry" into "silently skip": since mark_done always follows append, a checkpointed id without a record is an inconsistent state (power loss, manual line deletion during recovery) that lands in neither results nor pending — the document vanishes and nothing reconciles corpus size against `n_prompts`.
**Fix:** drop the `is_done` conjunct (record presence is the true done signal) or warn-and-requeue orphans; add a per-layer reconciliation print in `run.py`.

---

## Tier 2 — Structural and robustness

2.1 **Wrap all seven fan-out workers against raised API errors** — ✓✓ M·M — step 2's fused worker (scope/select/respond calls at `step2_responses.py:299,319,395`), step-1 `_plan`/`_draft`, step-3, baseline: one item's terminal API exception aborts the run and discards completed sibling work queued behind it in `pool.map` order (July's 1c/1d finding generalized). One uniform try/except → failure-sentinel + `*_failures.jsonl` + skip-unchecked pattern.
2.2 **backfill_gids: atomic rewrites, no null stamps** — ✓c M·S — in-place `"w"` rewrites of run artifacts (interruption truncates paid output that git can't restore for uncommitted runs) and `response_gid: null` insertions that defeat the skip guards.
2.3 **Stamp gate-unusable records** — ✓✓ M·S — a gate that never produced a verdict ships the record indistinguishable from gate-passed (`gate_failures`/`refine_failed` siblings both stamp; this path only prints). Flip the pinning test deliberately.
2.4 **Split the two god-files** — ✓c M/L — `evals/audit_dad.py` (2,997 lines, visible seam decay: duplicate section banners, 389 lines of section code after the `# ----- main` banner, a 314-line function at complexity 50) → mechanical package split with a re-export shim; `step1_dilemmas.run()` (534 lines, five closures over six loop-carried dicts) → checklist module + `plan_scenarios()` + a per-pass function taking an explicit state object (which also makes the 1.4 resume fix natural).
2.5 **Move viewer audit headline math into tested code** — ✓✓ M·M — the 0%-covered, 1,337-line audit page computes its two boldface "+X% considerations" claims inline with two different definitions that can diverge on legacy reports; move to `viewer/rendering.py` as one pure function (the pattern the rest of the page already follows).
2.6 **Auto-evals: don't re-bill on resume** — △ — write a done-marker keyed on corpus state; `--force-evals` escape hatch (pairs with 1.7c).
2.7 **Close the failure-path test gaps** — ✓✓/✓c M·M — layer 4's exception/truncation/SystemExit branches (the alignment-critical stage) are entirely untested — port layer12's three tests; audit_dad's `main()` and the stakes join are untested (a scope key rename would silently strip stakes from all delivery grading); publish_hf's three untested Hub-wrapper bodies carry safety-relevant kwargs (`repo_type="dataset"`).
2.8 **Token-truncation observability in the diversity eval** — ✓c M·S — `embed_texts` silently token-truncates (8192 OpenAI / 2048 Gemini) while the report counts truncation only at the character level; CJK-script documents (which the matrix deliberately produces) can be embedded as prefixes with `n_truncated: 0` reported.
2.9 **publish_hf hardening bundle** — △ — pre-publish interlocks (refuse empty corpus; `--allow-dev` gate for dev/dirty runs), dry-run printing the destructive half (delete_patterns, tag), huggingface_hub version pin + post-publish staged-files verification, per-stage model names on the card (knob→model pairs, not a nameless set), DAD language metadata from the manifest instead of hardcoded `["en"]`.

---

## Tier 3 — Process, docs, delivery hygiene

3.1 **Toolchain gates, still absent, now at 16.8k LOC** — ✓c M·L — pyproject.toml is still 7 lines; CI is still compileall+pytest. Today's numbers: 64/71 files unformatted; ~140 substantive ruff findings (complexity now dominates: `step1_dilemmas.run()` C901=73, `viewer/rendering.render_prompt` 61); mypy 157–174 (env-dependent). The two new ruff B023 hits in step-1 closures were **checked by the lead reviewer: benign as written** (consumed within their defining pass) — same latent-trap class as July's, worth the one-line default-bind fix but not bugs today. The July adoption plan (mechanical format commit + blame-ignore, curated lint select, per-module mypy, coverage floor — a 74% floor over non-viewer modules would pass today at ~85% with room to ratchet) still applies; every month of delay grows the one-time diff.
3.2 **Dependencies** — ✓✓ M·S — numpy imported in four modules but undeclared (the Python-floor comment even cites it); tqdm/jsonlines dead; seven deps unpinned, no lockfile — the dependency set behind the delivered corpora and their audit numbers is unreproducible. Fix set: add numpy, drop the dead two, commit a compiled lock, install CI from it.
3.3 **CLAUDE.md documents a `dad_segmented/` pipeline that has never existed in git history** — ✓✓ M·S — prominent call-out plus directory-listing entry for a module never pushed on any ref. Push it or delete both references; add a pre-delivery check that every path named in CLAUDE.md/README exists in `git ls-files`.
3.4 **Stale org URLs** — ✓c M·S — README clone URL and git-workflow-guide still say Mycelium-tools. Verifier correction: GitHub transfers leave a permanent redirect and the old org remains team-controlled, so the links *work* today and the name is not currently claimable — this is doc hygiene, not an active supply-chain hole. Update both anyway.
3.5 **README drift** — ✓✓ M·S — quick-test cost still "$0.05–0.15" (~30× stale, open issue #54); DAD step-1 described as 1a/1b/1c-optional against the shipped 1a–1d gate+refine.
3.6 **Triage the open-issue list before delivery** — ✓✓ M·S — issue #53 says "SDF is currently broken out-of-the-box on main" against code the matrix rebuild deleted wholesale; a lab skimming open issues reads an un-rebutted claim that the flagship pipeline crashes. Close/retitle #53, re-verify #24/#30, fix #54 with the README line.
3.7 **Smaller items** — △ — vendored 1.68MB tiktoken blob needs a provenance/license/regeneration README; the committed global `outputs/cost_log.jsonl` is a guaranteed cross-contributor merge conflict (gitignore it; per-run logs are the record); `claude-sonnet-5` priced at post-intro rates in `_PRICING` (cost logs overstate ~1.5× through 2026-08-31 — the window in which cost evidence is being quoted); review-prompt policy block and `call_claude` overloads carried from July.

---

## What is already good (and better than July)

- **The rebuilds fixed the worst July findings with the recommended patterns** — layers 12/3/4 are now models of fail-closed money-path engineering (per-item poison isolation, skip-without-checkpoint, systemic-failure aborts), and layer12's failure-path tests are exemplary.
- **The matrix machinery's composition guarantee is real**: exact largest-remainder marginals, all-or-nothing weight validation, and a contract-test suite that pins live templates to the live variables file.
- **`publish_hf` matched every documented safety claim checked**: delete_patterns correctly pipeline-scoped, the staging≠run-dir guard exists with four regression tests, sibling metadata fetched outside the staging tree, dry-run provably makes zero Hub calls, and the card builds from measured fields.
- **`audit_dad` largely designed out the July "parse failures counted as clean" class**: raw judge failures logged, extraction failures rendered as gaps ("a missing bar is a gap, not a zero"), showcase quotes verbatim-validated fail-closed.
- **Tests tripled** (295 → 742) and tracked the new code: compose_scenarios 88%, step1 94%, step2 100%, matrix 99%, publish_hf 97%, audit_dad 88%. Non-viewer coverage 85%.
- **Backend growth is disciplined** — bedrock reuses the api path (verified retry classification, wire-only model translation); the `auto` backend's baseline-arm exactness rule and loud sticky demotion are correctly implemented; pricing tables and cache multipliers verified against published rates.
- **Hygiene scan clean**: 998 committed output files — no API keys, HF tokens, or emails (the single leak is the home-path finding, 1.6).
- **CLAUDE.md is unusually well-maintained** against the rebuilt architecture (auto-evals wiring, publish contract, matrix keys all verified accurate) — the `dad_segmented` ghost (3.3) is the notable exception.

---

## Appendix A — Mechanical baseline, July → now

| Measure | 2026-07-10 (`cf8d91e`) | 2026-07-29 (`464c5bd`) |
|---|---|---|
| Source LOC (non-test) | ~6,841 (33 files) | 16,838 (48 files) |
| Test LOC / tests | 3,319 / 295 | 10,149 / 742 (all passing) |
| Coverage: total / non-viewer | 58% / ~76% | 68% / **84.8%** |
| Zero-coverage modules | score_sdf, score_dad, rate.py, viewer UI | score_sdf, score_dad_parked, rate.py, viewer UI (incl. new 77KB audit page) |
| ruff (broad select) | 541 (430 E501; 13 C901) | 1,215 (997 E501; 24 C901, 21 B905, 2 B023-benign) |
| `ruff format --check` | 45/52 files | 64/71 files |
| mypy (lenient, this env) | 69 | 174 (157 in the charter's env — env-dependent) |
| vulture | 1 unused var | 1 unused var (same one) |
| Dependencies | floors-only, no lock; numpy missing | same, + tiktoken/wordfreq/huggingface_hub; numpy still missing |
| Largest file | step1_dilemmas.py 819 | audit_dad.py 2,997 (step1_dilemmas 1,015) |

Tool versions differ between runs (ruff 0.16.0 here vs. the July uvx build); counts are indicative, not strictly comparable. Commands to reproduce are in the PR body.

## Appendix B — Findings ledger

50 findings from 6 charters; verdicts: 21 confirmed, 9 confirmed-with-corrections, 20 unverified (all low/info, unverified by design — the △ items above). Zero findings were refuted. Low/info items not detailed above include: a zero-value downstream axis silently emptying the deck sample; `is_incoherent` fail-open on tag-less plans; un-intersected arm populations in the jargon comparison; YAML/regex errors crashing the whole audit via the tics curation workflow; carried-forward paid audit data lacking a staleness marker; legacy scenarios missing `taxa_category` crashing record assembly. Full machine-readable findings (evidence, failure scenarios, recommendations, verifier notes) are preserved in the audit workflow's journal at `~/.claude/projects/-Users-declan-Projects-alignment-data-pipeline/00008bca-87eb-47f0-935e-004ea49fee99/subagents/workflows/wf_53baf5b3-219/journal.jsonl` (one JSON line per agent result); the July ledger remains at `.../wf_3e2c7be7-0ed/findings.json`.
