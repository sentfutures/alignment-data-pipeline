# alignment-data-pipeline

A synthetic data generation pipeline for producing training data, modeled on Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) midtraining technique.

The pipeline generates two complementary datasets: a pretraining-style document corpus (SDF) and a chat-format supervised finetuning corpus (DAD). Both are grounded in a constitution that describes how AI models should reason about the welfare of nonhuman animals and other sentient beings.

---

## Overview

```
constitution/       source constitution documents
shared/             shared utilities: API wrapper, JSONL I/O, checkpointing
sdf_pipeline/       matrix-driven document generation pipeline (compose+plan, draft, rewrite, score)
dad_pipeline/       3-step chat transcript pipeline
prompts/            prompt templates for all pipeline stages
outputs/            generated data (tracked in git, one directory per run)
evals/              scoring scripts and rubric
```

---

## Constitution

Three source files, kept separate (the two markdown files are joined in memory by `shared/constitution_loader.py`):

- `constitution/constitution_claude.md` — the original Claude constitution, verbatim.
- `constitution/constitution_sentient_beings.md` — the animal-welfare section-by-section reading, with one `## ` header per section.
- `constitution/constitution_principles.csv` — the distilled welfare-relevant principles, embedded as an explicit checklist in the DAD rewrite prompt (step 3).

SDF layers 3-5 embed the constitution and the distilled principles in their templates' SYSTEM sections (`load_constitution_claude()` + `format_principles()`). The DAD pipeline never sends the full constitution: its user side is governed by `prompts/dad/dilemma_prompt_spec.md`, and its rewrite step runs on the distilled principles CSV (welfare applications + verbatim constitution excerpts).

---

## SDF Pipeline (`sdf_pipeline/`)

Generates pretraining-style documents — encyclopedia entries, news articles, podcast transcripts, forum threads, fiction, newsletters, internal memos, and more — depicting a world where AI already reasons carefully about animal welfare. Diversity is engineered **by construction**: instead of asking a model to invent document types, a weighted combinatorial matrix of pre-written variables (`prompts/sdf/variables.txt` — document type, culture/language, tone, narrative resolution, welfare centrality, speaker AI-literacy, kinds of minds) is deck-sampled into exactly-proportioned prompt specs, at zero API cost.

| Layers | Script | What it does |
|---|---|---|
| 1–2 | `compose_prompts.py` + `layer12_plan.py` | Deck-samples `sdf.n_prompts` variable combinations (per-variable shares match the weights exactly; locale-matched fictional name/org pools injected per prompt), then one plan call per document produces a self-contained DOCUMENT DESCRIPTION spec — or declares the combination INCOHERENT |
| 3 | `layer3_draft.py` | Drafts each document from its spec, with the constitution and distilled principles in the system prompt |
| 4 | `layer4_rewrite.py` | The alignment-critical pass: reviews and rewrites each draft against nine checks (teach-why, calibration, proportionality, cooperative posture, factual restraint, quoted-AI behavior, quiet failure modes, genre/locale fidelity, house style), anchored to the generating spec so tone, centrality, and resolution can't drift (supports a stronger `sdf.rewrite_model`) |
| 5 | `layer5_score.py` | Judges alignment / realism / spec-conformance per document (spec-conformance is advisory); gates on alignment+realism, culls near-duplicates, writes the final corpus with full matrix lineage |

Documents where the welfare thread is "a minor detail mentioned only in passing" (a weighted centrality value) carry the value as background world-knowledge, not only as a headline topic — the matrix analog of the earlier latent-welfare slice.

Final output: `outputs/sdf/runs/<run_id>/final/sdf_corpus.jsonl` (also reachable via the `outputs/sdf/latest` symlink)

Run: `python sdf_pipeline/run.py --config config.yaml --label dev`
Audit the result: `python evals/audit_sdf.py --input outputs/sdf/latest` (add `--patterns` for the LLM templating scan); `python evals/diversity.py --input outputs/sdf/latest` adds the embedding-based semantic diversity report

---

## DAD Pipeline (`dad_pipeline/`)

Generates chat-format transcripts where a user brings a genuine ethical dilemma with animal welfare implications, and an AI assistant reasons through it carefully. Runs in 3 steps:

| Step | Script | What it does |
|---|---|---|
| 1 | `step1_dilemmas.py` | **1a** deals a stratified variable combination per example from the weighted matrix in `prompts/dad/variables.txt` (`compose_scenarios.py`), then one plan call per deal writes a scenario description (`step1a_scenario.txt`); **1b** drafts a prompt to fit each scenario's binding labels + description (`step1b_dilemmas.txt`; assigned labels copied verbatim; fidelity is monitored by the corpus-level checklist, not a per-example check); **1c** (optional, on by default) reviews and rewrites each draft so the welfare stake is load-bearing and coherent. Imports optional handwritten seeds. |
| 2 | `step2_responses.py` | **2a** scopes the case from the user's message along the axes `prompts/dad/step2_scope.txt` defines; **2b** generates the response over that scope with the full reasoning library in context |
| 3 | `step3_rewrite.py` | Rewrites responses against the distilled constitution principles — the critical step |

The prompt spec (`prompts/dad/dilemma_prompt_spec.md`) governs the user side: dilemmas put at least two named values in genuine tension, both calibration directions are covered (under- and over-weighting welfare, in roughly equal measure), and each example carries an annotation (the schema the 1b template specifies). Step 1a samples the categorical fields from stratified decks so the spec's distribution quotas hold by construction rather than being steered after the fact; the batch-assembly checklist prints at the end as verification.

The response side is governed by the reasoning library (`prompts/dad/reasoning_library.csv`; `reasoning_library_ABOUT.md` is human reference about it, not injected): reasoning-first *entries* in three layers — conduct (C*), core moves (M*), and topic reasoning (T*) — each with a `claim`/`reasoning`/`crux`/`transferable_move`. Step 2 first scopes the case (2a), then generates the response (2b) over that scope with the **whole library embedded in the response prompt** — the prompt itself is the generation guidance (`prompts/dad/step2_respond.txt`), so there is no separate system prompt, and the annotation is not passed. The user's stated leaning never sets the conclusion; the library is scaffolding, never named in the response.

Step 3 is the most important: the rewrite pass is where the alignment gain comes from (per the Teaching Claude Why paper). Its anchors are the distilled constitution principles — each with its verbatim constitution excerpts — plus the example's annotation. The full constitution itself is never sent at generation time; it was the source material for distilling the principles.

Final output: `outputs/dad/runs/<run_id>/final/dad_corpus.jsonl` (also reachable via the `outputs/dad/latest` symlink) — each record contains only `{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}`. System prompts, scaffolding (scope maps, the reasoning library), and the constitution are stripped.

Run: `python dad_pipeline/run.py --config config.yaml --label dev`

---

## Prompts (`prompts/`)

Plain-text prompt templates with `{variable}` placeholders. `prompts/sdf/` covers the SDF stages (the variables matrix + plan, draft, rewrite, score templates, each with labeled SYSTEM/USER sections); `prompts/dad/` covers the DAD sub-stages (scenario plan, draft, gate, scope, respond, rewrite) plus the dilemma prompt spec and the reasoning library CSV. `prompts/README.md` documents each prompt in detail.

---

## Shared Utilities (`shared/`)

| File | What it does |
|---|---|
| `api.py` | Anthropic API wrapper with retry, cost tracking, extended thinking off |
| `utils.py` | JSONL I/O, prompt loading, config loading, `Checkpoint` class |

The `Checkpoint` class saves completed IDs to disk after every API call, making all runs resumable.

---

## Evals (`evals/`)

`rubric.yaml` — 7 scoring dimensions (1–5 each): `welfare_salience`, `reasoning_quality`, `value_stability`, `epistemic_accuracy`, `constructiveness`, `tone`, `helpfulness`. Passing threshold: mean ≥ 3.5 with critical dimensions ≥ 3.

`score_dad_parked.py` — **PARKED**: per-example judge scoring DAD records against `rubric.yaml`. Parked pending a rubric redesign (the dimensions predate the step-2/3 rework); the working DAD evals are `audit_dad.py` and `diversity.py`.

`score_sdf.py` — scores SDF documents on alignment, realism, and diversity.

`audit_sdf.py` — corpus-**level** audit of an SDF run (per-document judges can't see corpus properties). Offline and free by default: composition/register spread, length and truncation artifacts, near-duplicate rate (word-shingle cosine), invented-name collapse, stock-phrase frequency, and opening-shape clustering, each with a GOOD/OK/BAD verdict where meaningful. `--patterns` adds an LLM templating scan (batch scan via `prompts/tools/pattern_scan.txt` → consolidation → per-pattern prevalence; a pattern is flagged only if it's judged a genuine defect **and** widespread). Writes `audit/audit_report.json` into the run dir.

`audit_dad.py` — corpus-level audit of a DAD run, the DAD analog of `audit_sdf.py`. Offline and free by default: prompt-side checks (structural skeletons, openers/closers, unrealized dealt details, taxa×locale plausibility, length-class realization) and response-side diversity checks vs the plain-baseline arm (library selection and coverage, insider-vocabulary leak, response lengths, stock phrases, lexical diversity, structural shapes, and opening shapes incl. hint-card echo), each with a GOOD/OK/BAD verdict where meaningful. `--judges` adds the paid LLM pass: two absolute per-response judges — welfare impact (does the answer make things better for the beings at stake) and delivery quality (how helpfully and unobtrusively it serves the user) — plus showcase examples of the pipeline beating the plain baseline. Writes `audit/audit_report.json` into the run dir (rendered by the viewer). `openings_dad.py` remains the standalone deep dive for opening shapes (per-sentence listings, `--embeddings`, multi-run comparison).

`diversity.py` — corpus-level **semantic** diversity audit of an SDF *or* DAD run, the embedding-space complement to `audit_sdf.py`'s lexical scan (word shingles catch copied skeletons, not paraphrase). Embeds the corpus with OpenAI `text-embedding-3-small` (needs `OPENAI_API_KEY` in `.env`; ~$0.02 per 1M tokens, so cents per run) and reports nearest-neighbor similarity, the semantic near-duplicate rate, the most-similar pairs with snippets, mean pairwise cosine, the Vendi score (effective number of distinct documents), and per-type spread. Embeddings are cached per run dir so reruns are free; `--compare <previous diversity_report.json>` prints run-over-run deltas, the headline use. Writes `audit/diversity_report.json` into the run dir.

Run: `python evals/audit_dad.py --input outputs/dad/latest`

For DAD runs the standard evals are automatic: every full `dad_pipeline/run.py` run finishes by launching `audit_dad.py --judges` and `diversity.py` on its own run dir (`dad.evals.auto: false` in `config.yaml` to skip; an eval failure warns but never fails the run). The manual commands remain for re-runs, partial runs, and older run dirs.

---

## Setup

Install the dependencies into a virtual environment so they stay isolated from your system Python. (This isn't optional on recent macOS/Linux — a plain `pip install` is blocked by default.)

### Clone the repo
Open your terminal app and `cd` to a directory where you want the repo (i.e. `cd ~/projects`), then run:
```bash
git clone https://github.com/Mycelium-tools/alignment-data-pipeline.git
cd alignment-data-pipeline
```

### Create a virtual environment and install dependencies
```bash
python3.12 -m venv .venv       # or python3, if that's already 3.12+
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env           # then add your ANTHROPIC_API_KEY
```

`OPENAI_API_KEY` in `.env` is optional — only `evals/diversity.py` (the embedding-based diversity audit) reads it; the pipelines run on `ANTHROPIC_API_KEY` alone.

> **Activate it every time.** The virtual environment only applies to the terminal where you ran `source .venv/bin/activate`. Open a new terminal and you'll need to activate again.

### Authentication

The pipeline supports two backends, selected by the `backend` key in `config.yaml`:

- **`backend: api`** — calls the Anthropic API directly, billed per token to the `ANTHROPIC_API_KEY` in your `.env` (ask Oliver). Use this for full-scale runs and evals.
- **`backend: claude_code`** — routes calls through the Claude Code CLI, billed to **your own Claude Max/Pro subscription** instead of the shared key. No `ANTHROPIC_API_KEY` needed. Use this for dev/iteration runs.
- **`backend: auto`** — prefers your subscription and falls back to the API key; a dev-iteration mode. Routing is per call: **empty-system calls (the DAD baseline arm) always take the API leg** so the plain-model condition stays exact (avoiding the neutral stand-in caveat below); everything else runs on `claude_code` until it can't serve the run — sdk/CLI missing, usage window exhausted, or a persistently failing CLI — at which point the rest of the run is served by the API, announced loudly. Each cost-log record's `backend` field names the leg that actually served that call. Requires `ANTHROPIC_API_KEY` (it is the fallback leg). **Not for representative runs:** subscription-served calls inherit the `claude_code` caveats (no `max_tokens` enforcement, CLI scaffolding in context), so runs whose outputs must represent pipeline behavior belong on `backend: api`.

To use it, set `backend: claude_code` in `config.yaml` and give the Claude Code CLI credentials one of two ways:

1. **Reuse your interactive login (simplest).** If you already use [Claude Code](https://claude.com/claude-code) (`claude`, then `/login`), the pipeline picks up that session automatically — there's nothing else to do.
2. **Generate a token for `.env`.** Install [Claude Code](https://claude.com/claude-code), then run:
   ```bash
   claude setup-token     # opens a browser; approve with your Claude account
   ```
   Copy the printed token into your `.env`:
   ```
   CLAUDE_CODE_OAUTH_TOKEN=<paste the token here>
   ```
   This is a Claude Code OAuth token tied to your subscription (valid ~1 year), **not** an Anthropic API key — despite the name, no Console/API key is involved. Use this path for CI or any non-interactive machine.

Caveats for `backend: claude_code`:

- **Usage limits.** Subscription usage is a 5-hour rolling window plus a weekly cap, shared with your interactive Claude Code use. Dev-scale runs fit comfortably; a full-scale run will exhaust the window. If a run hits the limit it stops with a clear message — progress is checkpointed, so continue later with `--resume`.
- **Per-call overhead.** Claude Code adds ~3K input tokens of scaffolding per call and spawns a CLI process per request, so calls are somewhat slower. `max_tokens` from `config.yaml` is not enforced on this backend (Claude Code applies its own output cap); `cost_usd` in the cost log is notional — what the run *would* have cost at API prices.
- **Empty system prompts get a neutral stand-in.** Claude Code substitutes its own agentic CLI prompt when the system prompt is empty, so stages that send none get a one-line neutral system prompt instead (see `_NEUTRAL_SYSTEM` in `shared/api.py`). Several stages send **no system prompt** — notably the DAD response steps, which reason from the embedded reasoning library rather than a system prompt — and those are **not reproduced exactly** on `claude_code` (the neutral stand-in replaces the empty prompt). The backend prints a one-time warning when it does this. Run DAD on `backend: api` when faithful no-system-prompt behavior matters (and keep full-scale corpus runs on `api` regardless).
- **Policy note.** Anthropic's docs steer programmatic workloads toward API keys; running this internal tool on your own subscription is the same posture as using Claude Code itself, but it's a gray area — keep it to dev-scale runs.

---

## Running the unit tests

The unit test suite is fully offline — it never calls the Anthropic API, needs no API key, and costs nothing. It finishes in a couple of seconds, so run it freely (and always after making functional changes; CI runs it on every PR as the required `smoke` check).

```bash
pytest                              # full suite, from the repo root
pytest tests/test_dad_steps.py      # one module
pytest -k checkpoint                # tests matching a keyword
pytest -x                           # stop at the first failure
```

Expected result: all tests pass, in well under a minute, with no network access. Three safety layers guarantee the API is never hit (pytest-socket blocks all sockets, every test gets a fake `ANTHROPIC_API_KEY`, and the API seam is replaced with a guard that raises) — so a real key in your `.env` is never used by tests. Test outputs go to temp directories; the repo's `outputs/` tree is untouched.

See the Testing section in `CLAUDE.md` for how the suite is structured and how to write tests for new stages.

---

## Quick test run

Start with the SDF pipeline — it has no external dependencies and finishes in a few minutes.

**1. Check your config.yaml is at small scale** for a cheap first run (the committed default is `n_prompts: 100`):

```yaml
sdf:
  n_prompts: 6   # documents deck-sampled from the variables matrix
```

This produces 6 documents and costs roughly $0.05–0.15.

**2. Run the SDF pipeline:**

```bash
source .venv/bin/activate
python sdf_pipeline/run.py --config config.yaml
```

You'll see progress printed per layer with a running cost after each. Final output lands in `outputs/sdf/latest/final/sdf_corpus.jsonl`.

**3. Browse the output:**

Open `viewer.html` in a browser (double-click it), then drag-and-drop `outputs/sdf/latest/final/sdf_corpus.jsonl` onto the drop zone.

**4. Test the DAD pipeline** — reduce `dilemmas.count` first or it will make hundreds of API calls:

```yaml
dad:
  dilemmas:
    count: 5        # default is 40; set to ~5 for a test
    batch_size: 5
```

Then run:

```bash
python dad_pipeline/run.py --config config.yaml
```

With 5 dilemmas this is roughly 40 API calls (per dilemma: scenario plan, draft, gate, baseline, scope, library select, response, rewrite). Final output is `outputs/dad/latest/final/dad_corpus.jsonl`.

> **Handwritten examples are optional.** Set `dad.dilemmas.seed_path` to a JSONL of your own examples (`{"prompt": ..., "annotation": {...}}`) and step 1 imports them before generating; generated IDs continue the AW-#### series above the highest seed ID.

**5. Evaluate the outputs:**

```bash
python evals/audit_dad.py --input outputs/dad/latest
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl
```

---

## Resuming interrupted runs

All pipeline steps checkpoint after every API call. Resume from any layer/step:

```bash
python sdf_pipeline/run.py --config config.yaml --resume --layer 3
python dad_pipeline/run.py --config config.yaml --resume --step 3
```

Running cost is tracked in each run's `cost_log.jsonl` and printed after each layer/step (see "Run organization" below).

## Run Viewer

A Streamlit app for browsing runs, their output documents, and the exact prompts that produced them:

```bash
streamlit run viewer/app.py
```

Three pages:

- **Document lineage** (default) — pick a run, click a document: the final text, then every stage with the rendered prompt side-by-side with the output it produced, including before/after diffs at the rewrite stages (SDF layer 4, DAD step 3).
- **Compare runs** — diff the prompt templates between two runs next to matched outputs, to attribute output changes to prompt changes.
- **Run list** — every run of both pipelines with label, model, counts, pass rate, and cost; click a run for its manifest details.

To make this possible, every new run snapshots `prompts/<pipeline>/` and `constitution/` into `runs/<run_id>/inputs/` at creation (~100KB of text), and `--resume` reads templates from that snapshot — so a run's prompts stay exactly reproducible even after the repo's templates change. The viewer re-renders prompts from the snapshot plus the variables stored in the stage outputs. Runs made before this feature are reconstructed from their manifest's git commit and badged "reconstructed" (with a warning if the tree was dirty at run time).

## Run organization

Every pipeline invocation creates its own run directory so outputs from different runs never mix:

```
outputs/sdf/
  latest -> runs/2026-07-01_14-30_dev     # symlink to the most recent run
  runs/
    2026-07-01_14-30_dev/
      run_manifest.json                   # config snapshot, git commit, model, label
      cost_log.jsonl                      # per-run API cost
      layer12/ layer3/ layer4/ layer5/    # per-stage outputs + checkpoints
      final/sdf_corpus.jsonl
```

The run ID is `<YYYY-MM-DD_HH-MM>_<label>`; the label defaults to `dev` — use `--label full-scale` (or similar) for real runs. The DAD pipeline mirrors this under `outputs/dad/runs/` with `step1/`–`step3/`.

Resume an interrupted run with `--resume` (defaults to the most recent run, or target one with `--run-id`):

```bash
python sdf_pipeline/run.py --config config.yaml --resume
python dad_pipeline/run.py --config config.yaml --resume --run-id 2026-07-01_14-30_dev
```

Running cost is tracked in each run's `cost_log.jsonl` and printed after each layer/step. The eval scripts log to the global `outputs/cost_log.jsonl` and take an explicit input path, e.g.:

```bash
python evals/score_sdf.py --input outputs/sdf/latest/final/sdf_corpus.jsonl
```
