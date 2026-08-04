# animal-welfare-data-pipeline

Synthetic training data pipeline for animal/sentient-being welfare alignment, modeled on Anthropic's "Teaching Claude Why" midtraining technique.

## Overview

Produces two complementary datasets:
- **SDF corpus** (`outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl`): pretraining-style documents depicting a world where AI already reasons carefully about sentient being welfare
- **DAD corpus** (`outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`): chat-format SFT data where a user brings an ethical dilemma and the assistant reasons through it with care. The whole pipeline is documented end-to-end in `prompts/dad/README.md`; the user side is governed by its Parts 1-6, the response side by the animal-ethics reasoning library (step 2) and the constitution (step 3 rewrite).

## Setup

See README "Setup" (venv + `pip install -r requirements.txt`, then `cp .env.example .env`). Auth depends on the `backend` key in `config.yaml`: `api` reads `ANTHROPIC_API_KEY`; `claude_code` bills the contributor's Claude subscription via the Claude Code CLI (logged-in session or `CLAUDE_CODE_OAUTH_TOKEN`); `auto` prefers the subscription and falls back to the api key — per-call: empty-system calls (the DAD baseline arm) always take the api leg so the plain-model condition stays exact, and an exhausted usage window demotes the rest of the run to the api, loudly, with each cost-log record naming the backend that served it. `api` stays the committed default: it is the faithful mode (subscription-served calls don't enforce `max_tokens` and carry CLI scaffolding in context), so runs meant to represent pipeline behavior stay on `api`; `auto`/`claude_code` are dev-iteration modes. Full setup and caveats for the dev backends are in "Dev backends" below. `GEMINI_API_KEY`/`OPENAI_API_KEY` are optional and read only by `evals/diversity.py` (embedding-based diversity audit; either provider works, and the runs to date used Gemini). `evals/publish_hf.py` (Hugging Face dataset publishing) reads `HF_TOKEN` from `.env` if set; otherwise falls back to a one-time `huggingface-cli login`, whose cached token `huggingface_hub` picks up on every later call. Either way the token needs write access to the target repo/org.

### Dev backends (internal — not documented in the public README)

`backend: claude_code` routes calls through the Claude Code CLI, billed to the
contributor's own Claude Max/Pro subscription instead of the API key. Two ways to
give it credentials: an existing interactive Claude Code login is picked up
automatically, or `claude setup-token` prints a token for `CLAUDE_CODE_OAUTH_TOKEN`
in `.env` (a subscription-tied OAuth token valid ~1 year, not a Console API key;
use this path for CI or any non-interactive machine).

Caveats, all reasons to keep this to dev-scale runs:

- **Usage limits.** A 5-hour rolling window plus a weekly cap, shared with your
  interactive Claude Code use. A full-scale run will exhaust the window; the run
  stops with a clear message and progress is checkpointed, so `--resume` continues.
- **Per-call overhead.** ~3K input tokens of CLI scaffolding per call and a CLI
  process per request, so calls are slower. `max_tokens` from `config.yaml` is not
  enforced (Claude Code applies its own output cap), and `cost_usd` in the cost log
  is notional — what the run would have cost at API prices.
- **Empty system prompts get a neutral stand-in.** Claude Code substitutes its own
  agentic CLI prompt when the system prompt is empty, so the one empty-system call
  in the pipelines (the DAD baseline) gets a one-line neutral prompt instead (see
  `_NEUTRAL_SYSTEM` in `shared/api.py`) and is **not reproduced exactly**. On
  `auto` the baseline always takes the API leg for this reason. A one-time warning
  prints when the substitution happens.
- **Policy.** Anthropic's docs steer programmatic workloads toward API keys.
  Running this pipeline on a personal subscription is the same posture as using
  Claude Code itself, but it is a gray area: dev-scale only, and never for runs
  whose outputs represent the pipeline.

`shared/__init__.py` enforces a Python floor (`MIN_PYTHON = (3, 12)`, matching numpy) at import — bump it there if the deps' floor rises. `.venv/` is gitignored.

## Running

```bash
# Full SDF pipeline (layers 1-5); --label defaults to dev
python sdf_pipeline/run.py --config config.yaml --label full-scale

# Full DAD pipeline (steps 1-3)
python dad_pipeline/run.py --config config.yaml --label full-scale

# Resume interrupted run from a specific stage (latest run, or target one with --run-id)
python sdf_pipeline/run.py --config config.yaml --resume --layer 3
python dad_pipeline/run.py --config config.yaml --resume --step 3 --run-id 2026-07-01_14-30_dev

# Evaluate outputs (latest symlink points at the most recent run).
# DAD runs the standard evals AUTOMATICALLY at the end of every full run
# (audit_dad --judges + diversity.py; dad.evals.auto: false to skip) —
# the commands below are for re-runs, partial runs, and older run dirs.
# DAD: corpus-level audit — response lengths, tracked tics/moves, and the
# tic-candidates review queue, offline/free; --judges adds the paid LLM judge
# pass (welfare impact + delivery quality + showcase).
python evals/audit_dad.py --input outputs/dad/latest
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl

# Preference pairs: two responses per prompt (arms a/b), then blind human A/B rating
python pref_pipeline/run.py --config config.yaml --prompts <prompts.jsonl> --label spec-v1-vs-plain
streamlit run pref_pipeline/rate.py

# Corpus-LEVEL audit of an SDF run: composition/register spread, near-dup rate,
# name/phrase collapse, opening shapes, truncation artifacts (offline, free);
# --patterns adds the LLM templating scan (scan -> consolidate -> prevalence)
python evals/audit_sdf.py --input outputs/sdf/latest
python evals/audit_sdf.py --input outputs/sdf/latest --patterns

# Semantic diversity audit (SDF or DAD run): embedding-space near-dup rate,
# most-similar pairs, Vendi effective-document count, per-type spread.
# Embeds via GEMINI_API_KEY or OPENAI_API_KEY (cents per run, cached);
# --compare a previous diversity_report.json for run-over-run deltas
python evals/diversity.py --input outputs/sdf/latest
python evals/diversity.py --input outputs/dad/latest
```

> **`evals/publish_hf.py` publishes a run's final corpus + audit reports to the public Hugging Face dataset repo `sentientfutures/animal-welfare-training-claude` — this is a deliberate, human-initiated action, not a routine post-run step.** Most runs are dev/exploratory and were never meant to become, or overwrite, the canonical published snapshot. **Only run this when a human developer explicitly asks for a specific run to be published** — never on your own initiative as part of a normal run, resume, or eval pass, and never for a run whose provenance (backend, label) you haven't confirmed with them first.

That one repo holds **both** corpora as separate HF *configs* (each gets its own selector in the dataset viewer), staged under per-pipeline subdirectories — `sdf/` and `dad/`, each with its corpus jsonl, `run_manifest.json`, and `audit/`. Consequences worth knowing before running it:

- Publishing one pipeline **regenerates the whole card**, so the script fetches the sibling's metadata off the Hub (`fetch_sibling`) to rebuild its section. Its corpus and HTML are never downloaded or re-uploaded.
- `delete_patterns` is scoped to `<pipeline>/audit/*`. A bare `audit/*` would delete the *sibling* dataset's audit files on every publish.
- **Tags are repo-wide**, so prefix them per dataset (`sdf-v1-…`, `dad-v1-…`). The pre-multi-config `v1-fullscale-500-opus5` tag predates this convention.
- `--dry-run` makes zero network calls, so it cannot see a sibling already on the Hub and says so — the preview shows only the pipeline being published. (It therefore also skips the `git fetch` the merge check would otherwise do, and says `origin/main` may be stale.)
- **An unmerged publish warns and asks, and is recorded on the card.** Before staging, the script checks whether the current `HEAD` and the run's own `git_commit` are reachable from `origin/main` (`utils.merge_state`). If either isn't — or can't be verified — it prints what's unmerged and requires a typed `yes`; with no TTY it exits telling you to pass `--allow-unmerged`. Proceeding stamps "published from an unmerged branch" into the dataset card's provenance block and into the Hub commit message. The stamp persists in `<pipeline>/card_meta.json`, so it survives the sibling's next publish regenerating the card, and clears itself once that pipeline publishes something merged. A **dirty tree at run time is context, never a trigger** — every run so far has been dirty, and a warning that fires on all of them is one people learn to ignore. This is a guardrail against accidents, not an access control: the write token is on contributors' laptops, so anyone can bypass the script entirely — which is exactly why the card, not the terminal, carries the record.

```bash
# Stages final/{sdf,dad}_corpus.jsonl + run_manifest.json + audit/*.{json,jsonl,html}
# into <pipeline>/ (report_content.json excluded — editorial, already baked into
# corpus_report.html) and writes a dataset card built entirely from the audit
# files' own measured fields. Provenance lists the per-stage `*_model` overrides,
# not just the manifest's top-level `model` (which reads claude-sonnet-5 even on
# runs whose real generation stages were all Opus). Requires a Hub token with
# write access to the target repo/org, one time (`huggingface-cli login`, or
# HF_TOKEN in .env); --dry-run stages + prints the card with no network calls.
# An unmerged run prompts for confirmation first (--allow-unmerged skips the
# prompt; the card records it either way).
REPO=sentientfutures/animal-welfare-training-claude
python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run
python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id $REPO \
    --pretty-name "Animal-welfare training dataset" --tag sdf-v1-<run-label>
python evals/publish_hf.py --input outputs/dad/runs/<run_id> --repo-id $REPO \
    --pretty-name "Animal-welfare training dataset" --tag dad-v1-<run-label>
```

## Run Organization

Each pipeline invocation creates a fresh run directory `outputs/{sdf,dad}/runs/<YYYY-MM-DD_HH-MM>_<label>/` containing the per-stage dirs (`layer12`, `layer3`–`layer5` / `step1`–`step3`; steps 2–3 keep explicit checkpoints, step 1 resumes from its own append-only jsonl files; DAD runs also hold `baseline/` — a plain-model response per dilemma serving as the viewer’s control arm and as the advisory "first take" in the 2b prompt, never trained on; toggled by `dad.baseline.enabled`, see `dad_pipeline/baseline.py`), `final/`, `run_manifest.json` (label, git commit + branch + dirty state, model, full config snapshot; `manifest_version` 3 added `git_branch`, so every earlier run has a commit but no branch), and a per-run `cost_log.jsonl`. This keeps outputs from separate runs isolated — checkpoints live inside the run dir, so `--resume` (latest run by default, or `--run-id`) continues exactly one run. The label is purely descriptive (`dev` by default; scale knobs stay in `config.yaml`). An `outputs/<pipeline>/latest` symlink always points at the most recent run (gitignored, as are `local_*` run dirs, for every pipeline including pref). Run-scoping helpers (`create_run_dir`, `resolve_run_dir`) live in `shared/utils.py`.

## Scale / Cost

All knobs are in `config.yaml`. For development, reduce `sdf.n_prompts` (SDF — documents per run, deck-sampled from the variables matrix) and `dilemmas.count` (DAD) to keep test runs cheap. `sdf.seed` pins the deck sample; same seed + same variables file = the same composed prompts.

SDF supports per-stage model overrides (`sdf.plan_model` / `sdf.draft_model` / `sdf.rewrite_model` / `sdf.score_model`, each falling back to the global `model`): plans and drafts tolerate a cheap model, but the layer-4 rewrite and layer-5 scoring are the quality-critical calls — spend there first.

DAD likewise: `dad.scenario_model` (1a scenario plan) / `dad.prompt_draft_model` (1b) / `dad.prompt_gate_model` (1c gate) / `dad.prompt_refine_model` (1d refine — a separate knob; the gate never falls back to it) / `dad.response_scope_model` (2a) / `dad.response_select_model` (2a.5 library-entry selection; falls back to `response_scope_model` before the global) / `dad.response_draft_model` (2b) / `dad.constitution_rewrite_model` (step 3), each falling back to the global `model` — step 3 is the alignment-critical rewrite, spend there first. The global `temperature` (1.0) is wired into every call; generation wants 1.0 (diversity is the product — 1b register variety, 2b independent samples), and `call_claude` accepts a per-call override for eval/debug use.

`workers` sets how many API calls run concurrently within each SDF layer and each fan-out DAD stage — 1a scenario plans, 1b drafts (one call per scenario), 1c gate judgments, 1d refine rewrites, step 2 (one worker per dilemma: scope + its responses), step 3 rewrites (all via `utils.parallel_map`; set to 1 for serial debugging). Workers only call the API and parse — all file writes and checkpoint marks stay on the main thread, in input order.

Rough cost anchor (Sonnet 5, July 2026): a DAD example costs ~$0.20–0.25 end-to-end, so the default 40-example run is ~$9–10; smoke runs of 3–5 examples are under $1.

Running cost is tracked per run in `outputs/{sdf,dad}/runs/<run_id>/cost_log.jsonl` (evals log to the global `outputs/cost_log.jsonl`) — check it any time. Each record carries a `stage` tag (`prompt_draft`, `layer4`, `constitution_rewrite`, …) matching the model-knob names; the viewer's run list renders the per-stage cost breakdown (pre-tag records show as "(untagged)"). Records also log `duration_s` and `attempts` (API-retry count), and DAD calls tag an `item_id` naming the record served (scenario_id for 1a/1b/1c/1d — pre-rework runs comma-joined a 1b batch's ids — prompt_gid for 2a/2a.5, `{prompt_gid}_s{n}` for 2b, response_id for step 3; older runs' cost logs key the 2a/2b slots by their retired per-run prompt ids); the viewer's lineage page reads these via `loader.call_stats` to show model · cost · time · retries in each step expander (runs logged before these fields fall back to a model-only note).

## Preference Pipeline

`pref_pipeline/run.py` generates one pair per input prompt: a response from each of two arms defined in `config.yaml` under `pref.arms` (`name` + inline `system_prompt` or `system_prompt_file` relative to the repo root, optional per-arm `model`/`max_tokens`). Use it to A/B test candidate response specs against each other or against the bare model. Prompts come from any JSONL with a `user_message`, `refined`, or `prompt` field (handwritten sets, DAD step-1 `dilemmas.jsonl`). Runs live in `outputs/pref/runs/<run_id>/` with the same manifest/checkpoint/resume/cost-log conventions as SDF/DAD; resolved arms are frozen into `inputs/arm_prompts.yaml` at run creation so `--resume` replays them. Checkpointing is per **arm** (`pairs/arm_responses.jsonl`), so one failed arm never discards or re-bills its sibling's paid response.

`streamlit run pref_pipeline/rate.py` is the blind rating UI: arm identities are hidden, side order is fixed per pair (md5 of `pair_id` → `left_arm`, so it carries no signal but survives reloads), choices are Response 1 / Response 2 / Tie / Both bad plus an optional note, keyed by rater name. Ratings append to `ratings/ratings.jsonl` (both the blinded side and the deblinded arm); after every rating `final/preferences.jsonl` is rebuilt with one `{user_message, chosen, rejected, chosen_arm_name, rater}` record per decisive rating (ties/both-bad excluded). Data logic lives in `pref_pipeline/prefdata.py` (no Streamlit imports).
## Testing

- Run `pytest` from the repo root (deps are in `requirements.txt`). The suite is fully offline and finishes in seconds; it runs inside the required `smoke` check on every PR (`.github/workflows/ci.yml`, a job with no API secret exposed), so a failing test blocks merge.
- Tests NEVER call the Anthropic API. Four layers enforce this: pytest-socket (`--disable-socket` in `pyproject.toml`) blocks all network at the socket level; an autouse fixture sets a fake `ANTHROPIC_API_KEY` and resets `shared.api` globals per test; and both backend seams — `shared.api._call_with_retry` and `shared.api._call_claude_code_with_retry` (which would otherwise spawn the Claude Code CLI) — are replaced with functions that raise. The embeddings seam (`shared/embeddings.py`, both providers) gets the identical layered treatment (fake `OPENAI_API_KEY`, globals reset, `_embed_with_retry` blocked).
- To exercise pipeline stages, use the `stub_claude` fixture in `tests/conftest.py` (queue of canned response strings, or a callable dispatcher) — it patches `shared.api.call_claude`, the single chokepoint every module uses. Never let real `anthropic` error types reach the real `_call_with_retry`; tenacity would sleep minutes. For the diversity eval, `stub_embeddings` patches `shared.embeddings.embed_texts` the same way (deterministic per-text vectors, or pass exact geometry).
- All test outputs go to pytest `tmp_path`; the `PIPELINE_OUTPUT_ROOT` env var redirects the `run.py` orchestrators away from the real `outputs/` tree.
- Determinism: an autouse fixture seeds `random`; `sample_language` accepts an injectable `rng`; uuid/timestamp values are asserted by shape, never by value.
- Tests encode CURRENT behavior, including known quirks. Don't change pipeline behavior just to make a test expectation nicer — decide the spec first, then flip the test deliberately.

### PR expectations (required for contributions)

- **Run `pytest` after every functional change** — after editing any code under `shared/`, `sdf_pipeline/`, `dad_pipeline/`, or `evals/`, and again before each commit or push. The suite is offline and takes ~2 seconds; don't wait for CI to find out.
- **Every PR description must include a "How to test" section** with the manual steps a reviewer can run to verify the change and the expected results (see `.github/pull_request_template.md`). Note that `gh pr create --body` bypasses the template — when opening a PR from a Claude session, write the section into the body explicitly. These instructions serve reviewers before merge and become the historical record when a feature later needs to be understood or reverted.
- **Review responses are posted by a human, never by an agent.** Replies to review threads, review comments, approvals, and thread resolutions are the PR author's to post — an agent commenting under a contributor's account makes a PR look like a human weighed the review when none did, and keeping the poster human keeps the record truthful about who actually decided. An agent addressing review feedback should apply the comments it agrees with, report the rest to the author with a recommendation, and draft reply text for them to post rather than posting it. The `pr-review-watch` skill carries the full workflow (verify every claim against the code first; escalate disagreements and design trade-offs instead of guessing).

### Writing tests for new code (required for contributions)

Every PR that adds or changes pipeline behavior must add or update tests in the same style — CI runs the suite on every PR, and a stage without tests is a stage that silently breaks at $50 a run. Follow these rules:

- **FIRST**: fast (the whole suite runs in ~1s — keep it that way), independent (no test depends on another's state; `shared.api` globals are reset per test by the autouse fixture), repeatable (seed or inject randomness; assert uuid/timestamps by shape), self-validating (plain asserts, no eyeballing output), timely (written with the change, not after).
- **Test behavior, not implementation**: drive each stage through its public `run()` and assert on returned records, files written, and what reached `call_claude` (the `calls` list from `stub_claude`). Don't reach into private helpers or assert on internal call order unless that IS the contract.
- **Mock only the external boundary**: `stub_claude` replaces `shared.api.call_claude` — the only external dependency. Real prompt templates, real constitution files, and real (tmp) filesystems stay in play; that's what makes the tests catch template/pipeline drift.
- **Never touch the network or the repo's outputs/**: the API guard and pytest-socket enforce the first; `tmp_path` + `PIPELINE_OUTPUT_ROOT` enforce the second. If a new stage grows a second external dependency, stub it in `tests/conftest.py` the same layered way.
- **Cover the money paths**: every new stage needs at least a parse-happy-path test, a malformed-response fallback test, and a checkpoint/resume test asserting zero API calls for completed work — resume correctness is what protects paid work when a run dies.
- **Derive, don't hardcode, constitution-shaped expectations**: counts and principle ids come from `load_segments()`/`META_PRINCIPLE_IDS`/`_PRINCIPLE_KEYWORDS` (the section count is pinned once, in `test_constitution_loader.py`) — the reading is actively edited and hardcoded ids renumber. FIFO queue stubs are for serial stages only; stages that fan out via `parallel_map` need a callable dispatcher (the stub fails loudly if violated).
- If you change a prompt template's placeholders or add a template, update `tests/test_prompts_render.py` (and the e2e dispatcher markers in `tests/test_e2e_smoke.py` if the opening prose changed).

## When running in CI

Rules for Claude sessions launched by the repo's automation (`.github/workflows/claude-fix.yml`, `claude-review-fix.yml` — the pipeline that turns `claude-fix`-labeled issues into PRs on `claude/issue-<n>` branches; `scripts/kickoff.sh` files the issues):

- **Test gate** (dependencies are already installed by the workflow): `python -m compileall -q shared sdf_pipeline dad_pipeline pref_pipeline evals viewer && pytest` from the repo root — the exact required `smoke` check. Run it before every push; a PR only opens when it is green.
- **Definition of done**: the gate is green, the issue's acceptance criteria are met, and there are **no unrelated changes** — touch only files the plan names; never touch `outputs/`, `.github/workflows/`, or `code_quality/`.
- **Incremental commits**: commit AND push after every coherent step, so a run killed by a usage limit leaves resumable state on the branch. Small commits with imperative subjects. Never force-push; never commit scratch files (plan.md, pr_body.md, review-body.md).
- **PR body format** (the workflow opens the PR from files you write; `gh pr create --body` bypasses the PR template, so every section is written explicitly): `Closes #<n>`, `## Plan` (the plan as approved by the risk gate), `## Risk class`, `## How to test` (concrete reviewer steps + expected results — required, see PR expectations above), and a final Claude-generated callout line.
- **Boundaries**: never merge, approve, or close PRs; never remove the `needs-human` label; when giving up, always post a comment explaining the state you left behind before ending the turn.

## Constitution

Three source files, loaded by `shared/constitution_loader.py` (the two markdown files are joined in memory, never combined on disk):

- `constitution/constitution_claude.md` — the original Claude constitution, verbatim.
- `constitution/constitution_sentient_beings.md` — the animal-welfare reading, parsed by `## ` headers into 16 sections by `load_segments()`, each with a `principle_id` (0–15; ids 0, 14, and 15 are the `META_PRINCIPLE_IDS` meta sections — scope note, violation-typology appendix, closing humility note). Not sent by any generation call — it was the source context for distilling the principles CSV; the viewer still renders the legacy runs that used its sections as per-example anchors.
- `constitution/constitution_principles.csv` — the distilled welfare-relevant principles (`number`, `principle`, `welfare_application`, `constitution_excerpts`). `load_principles()`/`format_principles()` render each principle with its welfare application and verbatim constitution excerpts as the `CONSTITUTION PRINCIPLES` block in the DAD step-3 rewrite prompt and as the principles half of the SDF prompts.

SDF layers 3-5 embed the constitution (and, for layers 3-4, the formatted principles CSV) in each template's labeled SYSTEM section via `{constitution_claude}` / `{constitution_principles}` (`load_constitution_claude()` + `format_principles()`); the pipeline splits the rendered file on the `=== SYSTEM PROMPT ===` / `=== USER PROMPT ===` markers and sends the sections as system prompt and user message. `load_constitution_with_principles()` remains for the viewer and legacy runs. `load_full_constitution()` (constitution + sentient-beings reading) is not sent by any pipeline; it remains for the viewer and legacy runs. The DAD pipeline never sends the full constitution — sending it per rewrite call was the dominant token cost of the step.

## Key Design Decisions

- **Extended thinking OFF** everywhere — training data should show user-facing reasoning, not internal scratchpads
- **SDF documents depict a world; they never argue an implanted claim.** The corpus shows careful welfare reasoning as normal and constitution-grounded. A reviewed sister pipeline instilled beliefs by having every document assert paraphrases of fixed claims ("capable AI naturally extends moral consideration...") — that belief-implantation route was deliberately rejected (conflicts with the constitution's honesty/calibration commitments and with TCW's teach-why finding); its *scaffolding* (latent slice, register balance, entity pools, corpus audits, per-stage models) was adopted instead.
- **Composition by construction (the matrix)**: SDF layers 1-2 are not LLM calls — a weighted variables matrix (`prompts/sdf/variables.txt`: document type, culture/language, tone, resolution, centrality, AI-literacy, kinds of minds, framing, domain, decision scale, AI role) is deck-sampled so per-variable shares match the weights exactly (largest-remainder quotas). One plan call per document turns each combination into a self-contained DOCUMENT DESCRIPTION spec; only the spec travels downstream (extracted fail-closed; INCOHERENT combinations are checkpointed as deliberate rejections). Every downstream stage is anchored to the spec so the engineered composition survives drafting, rewriting, and gating — layer 5's `spec_conformance` dimension (which replaced the per-doc `diversity` score a single-document judge cannot honestly produce) measures exactly that, advisorily. The centrality axis reserves a weighted slice for documents where welfare is "a minor detail mentioned only in passing" — background world-knowledge, the matrix analog of the old latent slice. Three further deliberate slices guard against corpus-level failure modes: a no-welfare-stake resolution arc (~10%, the AI correctly raises nothing — breaks the "aligned AI always brings up welfare" pattern), identity document types (~5%, Claude in its own voice, targeting TCW's persona-attachment gap), and the framing axis's web-of-correlations value (welfare reasoning tied to the rest of the aligned character). `evals/audit_sdf.py --principles` judges which distilled constitution principles each sampled doc exercises and flags starved principles (fix at the arc/weight level, not per-doc assignment).
- **Skeptic preservation is enforced at three stages**: the plan assigns tone, layer 4 must not resolve a skeptical stance into agreement (a conversion failure observed in an early validation run), and layer 5's alignment rubric explicitly allows a skeptical document to score 10. Verified composition-neutral at n=100 (20/20 skeptical docs passed the gate).
- **Fictional entities by construction**: the composer injects locale-matched people/org names into each plan prompt from per-culture seeded Faker pools (`shared/entity_pools.py`, native script where the locale uses one; instruction-only fallback for uncovered locales) — prevents invented-name collapse ("Elara", "Meridian Institute") and keeps fabrications from ever attaching to real organisations. The spec carries the chosen names downstream; layers 4-5 treat spec-provided entities as fictional-by-construction, never fabrications to strip.
- **Corpus-level audit after every run** (`evals/audit_sdf.py`): per-document judges cannot see corpus properties (register collapse, name reuse, templated openings — the failure mode that same early run exposed), so composition, redundancy, and templating are measured over the corpus as a set; `--patterns` runs the LLM scan wired to `prompts/tools/pattern_scan.txt`. Near-duplicate culling also runs inside the pipeline (layer 2 subtypes via `sdf.subtype_dedup_threshold`, final corpus via `sdf.near_dup_threshold`).
- **DAD pipeline construction** (design settled as of 2026-07-30; the step templates remain normative for prompt wording — `prompts/dad/step1*.txt` + `variables.txt`, `step2_*.txt`, `step3_rewrite.txt`). Step 1 deals a stratified variable combination per example from the weighted matrix in `prompts/dad/variables.txt` (same architecture as SDF layers 1-2: `dad_pipeline/compose_scenarios.py` is the composer, structural rules and taxa/length tables live at the top of it), then runs four paid sub-stages per example: **1a** one plan call per deal writes a scenario description (`prompts/dad/step1a_scenario.txt`; INCOHERENT combinations are checkpointed as deliberate rejections); **1b** drafts the user prompt; **1c** a pass/fail quality gate (`prompts/dad/step1c_gate.txt`) — a reject routes the scenario back for redraft with the gate's reasons injected, capped at 3; **1d** review-and-rewrites gate-passed drafts against their dealt cards (`prompts/dad/step1d_refine.txt`; an `<unfixable>` verdict rejects the scenario like 1a INCOHERENT). Step 2 runs three sub-stages per prompt: **2a** scopes the case (`prompts/dad/step2_scope.txt`); **2a.5** a dedicated retrieval call selects the reasoning-library entries that fit it (`prompts/dad/step2_select.txt`; fail-open — an unusable selection sends 2b the whole library rather than retrying); **2b** generates the response per `prompts/dad/step2_respond.txt`, which splits into a system half (the standing generation guidance) and a user half carrying the scope, the selected library rows, and the plain-model baseline as an advisory "first take" (degradable: with the baseline disabled or missing the slot renders empty). The library (`prompts/dad/reasoning_library.csv`) is sampling scaffolding, never named in responses. Step 3 rewrites each response against the distilled constitution principles and is the **alignment-critical pass — do not skip or abbreviate it**. The dealt cards never enter any response-side prompt — the response side reads only the shipped user message. Every generation call rejects truncated output (`stop_reason` checked; failed work is not checkpointed, so `--resume` retries it). The `.md` docs in `prompts/dad/` (`README.md` — the end-to-end pipeline spec, written for outside readers — and `reasoning_library_ABOUT.md`) describe the settled design; the templates and CSV stay authoritative for exact wording and entries.
- **Committed run outputs are deliberate.** Smoke/validation runs under `outputs/*/runs/` are kept in git as reviewable examples of pipeline behavior at each design stage; `local_*`-labeled runs and `latest` pointers stay untracked (gitignore covers all pipelines incl. pref). Prune only with team agreement. When a PR both changes pipeline code and commits a fresh run demonstrating it, prefer landing them as separate PRs (code first, then the run) — bundling them produces diffs dominated by generated data (PR #73: two committed runs made up 71% of the diff's additions), which is hard for both human and automated review to work through. `.gitattributes` marks these paths `linguist-generated` so GitHub's UI collapses them either way.
- **Final DAD records contain only user + assistant messages** — system prompts, reasoning library and scope scaffolding, and the constitution are stripped before training records are written

## Directory Structure

```
constitution/       constitution source documents (Claude constitution + sentient-beings reading)
context_docs/       background reading: tcw.md ("Teaching Claude Why" post this repo implements) + constitution PDF
shared/             API wrapper, utils, constitution loader
sdf_pipeline/       matrix document pipeline: compose+plan (layers 1-2), draft, rewrite, score
dad_pipeline/       3-step chat transcript pipeline
pref_pipeline/      response-pair generation + blind human A/B rating app
prompts/sdf/        prompt templates for SDF layers
prompts/dad/        dilemma prompt spec + reasoning library + DAD step templates
outputs/sdf/        intermediate + final SDF outputs
outputs/dad/        intermediate + final DAD outputs
evals/              scoring scripts and rubric
```
