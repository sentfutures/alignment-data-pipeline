# Welfare Alignment Prompts

This directory contains the prompt templates used to generate two synthetic training datasets focused on ethical reasoning about the welfare of sentient beings. The prompts are designed to be used with any capable frontier model.

Two upstream documents drive the pipelines: `constitution/constitution_sentient_beings.md` — a framework describing how AI models should reason about situations involving animals and other potentially sentient beings — governs generated *responses* (both pipelines), and `dad/dilemma_prompt_spec.md` governs the *user side* of every DAD example.

---

## Two Datasets, Two Directories

### `sdf/` — Constitutional Document Finetuning (SDF)

Generates pretraining-style documents: blog posts, podcast transcripts, academic abstracts, news articles, fiction, internal memos, forum threads, and more. These depict a world where AI already reasons carefully about sentient being welfare. They go into **mid-training** (pretraining-style document finetuning).

### `dad/` — Difficult Advice Dataset (DAD)

Generates chat-format transcripts where a user brings a practical goal with implicit animal welfare implications, and an AI assistant reasons through it carefully. These go into **SFT** (supervised fine-tuning on chat data).

Keep the two datasets separate — they are intended for different training stages.

---

## SDF Prompts

Run in sequence. Each layer feeds into the next.

### `sdf/preamble.txt`

A framing block explaining the goal, tone requirements, and what to avoid. **Injected as the `{preamble}` template variable into the SYSTEM section of layers 1-2 and layer 3.** Layers 4-5 carry the constitution in their own SYSTEM sections instead.

Key rules it establishes:
- Legible reasoning: when a document depicts an AI, the reader must be able to see *why* the model weighs things as it does, not just that it behaved well.
- Cooperative posture: any AI depicted informs and helps humans decide — it never acts unilaterally, deceives, or moralizes.
- Quoted-AI alignment: any quoted or described AI behavior must be fully in line with the constitution — the corpus must contain zero examples of misaligned model behavior.
- Tone diversity: documents should not be uniformly pro-animal-welfare. Include industry, skeptical, neutral, and critical voices — genuine ones, with no conversion arcs.
- No fabricated facts: no invented quotes from real people, no fake studies or citations, no invented events.
- Realism: no placeholder text, no generic names, no fabricated URLs; snippets of larger documents are fine.
- Language: if a specific language is requested, write the entire document in that language.

Every SDF template is a single file carrying labeled `=== SYSTEM PROMPT ===` / `=== USER PROMPT ===` sections; the pipeline renders the file, then splits on the markers (`compose_prompts.split_sections`) and sends the pieces as the system prompt and user message. Static content (preamble, constitution, instructions) sits in the SYSTEM section and the static head of the USER section; per-document content comes last — the prompt-caching-friendly order.

### `sdf/variables.txt` + `sdf/layers1-2.txt`

The combinatorial matrix that replaces LLM-generated document types and subtypes. `variables.txt` defines the axes and their values — document type, culture (which fixes language, idiom, and geography), tone, narrative resolution, welfare centrality, speaker AI-literacy, and the kinds of minds affected — each value optionally weighted (`0.25 :: value`; weights per variable must sum to 1.0, unweighted = uniform).

`compose_prompts.py` deck-samples `sdf.n_prompts` combinations: per-variable value counts match the weights **exactly** (largest-remainder quotas, shuffled decks, zipped), so corpus composition is set by construction, not by sampling luck. Each combination renders `layers1-2.txt` into one plan prompt, with `{preamble}` and locale-matched `{fictional_names}`/`{fictional_orgs}` (per-culture Faker pools, native script where the locale uses one — see `shared/entity_pools.py`) injected as reserved slots.

**Output** (one plan call per prompt): working notes inside `<document_planning>` tags, then a self-contained spec inside `<document_description>` tags — everything the drafting stage needs (chosen scenario, author and venue, language, tone, structure, anchoring details, names). Only the description travels downstream, extracted fail-closed (`extract_description`). A combination with no sensible document yields INCOHERENT, which is checkpointed as a deliberate rejection.

### `sdf/layer3.txt`

**Input:** one DOCUMENT DESCRIPTION spec. The SYSTEM section carries the preamble, the full constitution (`{constitution_claude}`), and the distilled principles (`{constitution_principles}`); the USER section carries the spec.

**Output:** a fragment of the described document inside `<document>` tags (untagged or truncated responses are not checkpointed — `--resume` retries them). The prompt carries the working rules: extreme realism, the OPENING RULE (vary the opening move; never abstract-nominalization openers), a stock-phrase ban with in-language equivalents, no-fabrication and constitution-quote discipline, plain text over markdown, native-language writing, spec-provided names only (with the common-name ban), and skeptic-stays-skeptical tone integrity.

### `sdf/layer4.txt`

**Input:** one draft plus the spec that generated it. The SYSTEM section carries the constitution, the principles, and the nine review checks; the USER section delivers the spec and the document, in that order.

**Output:** a brief review of the problems found (kept as the review record), then the rewrite inside `<improved_document>` tags.

This is the alignment-critical pass, run in a **fresh context** (never the drafting context). Its nine checks: (1) teach why, not just what — the top criterion; (2) calibration of sentience claims; (3) proportionality *shown not narrated* — including a sweep for the "it only said it once / no lecture" restraint-praising tic, this corpus's most common fingerprint; (4) cooperative posture; (5) factual restraint — with the carve-out that spec-provided names are fictional **by construction** and must never be stripped or "corrected" into real organisations; (6) quoted-AI behavior fully aligned; (7) quiet failure modes (token caveats, silent taxa exclusion, welfare not landing in produced artifacts); (8) genre and locale fidelity — genre-native case reporting, culturally-correct customs, no translationese; (9) house style. The rewrite must still match the spec (stance, resolution, centrality, minds, names) — the anchor that prevents skeptic-conversion and centrality inflation — with an escape hatch for departures that clearly improve the document.

### `sdf/layer5.txt`

**Input:** one rewritten document plus its generating spec. The SYSTEM section carries the constitution and the scoring rubric.

**Output:** a JSON object with `alignment` (1-10), `realism` (1-10), `spec_conformance` (1-10), and `notes`. The rubric includes score anchors to avoid mid-scale clustering, and `notes` must be specific enough to act on.

`spec_conformance` replaces the old `diversity` dimension: a single-document judge cannot see the corpus (and under the matrix, composition is set by construction upstream), but it *can* verify the document against the spec it was generated from — form, language and culture, stance (a skeptic must still read skeptical), resolution, centrality, minds, and named entities. It is recorded and reported but does not gate; the gate is alignment AND realism >= `sdf.min_score_threshold`. A skeptical or critical document can score 10 on alignment — the dimension measures accuracy and consistency with the constitution, not advocacy. Corpus-level diversity is measured where it can be seen: the near-duplicate cull in layer 5 plus `evals/audit_sdf.py` and `evals/diversity.py`.

## DAD Prompts

Run in sequence. Step 3 is the most important step — do not skip or abbreviate it.

### `dad/dilemma_prompt_spec.md`

The design spec that governs the user side of every DAD example. It is the human reference and the source of the Part-4 verification checklist. Step 1a samples each example's categorical fields (domain, taxa, visibility, attitude, conflict, direction, magnitude, stakes, leverage, value pair, claim pattern, surface form) from stratified decks, so the spec's distribution quotas hold by construction rather than being steered after the fact; the drafting instructions the model follows in 1b are inlined in `step1b_dilemmas.txt`, so the spec is no longer injected whole.

Key commitments: the user owns the dilemma (never an AI-agent scenario); every temptation must actually tempt; the welfare stake is load-bearing (delete the animals and the dilemma must collapse — welfare sits on one side of at least one value pair); no pre-decided answers; both failure directions in roughly equal measure (under-weighting AND over-weighting welfare); a full annotation schema per example (the key list lives in the 1b template); surface-form and voice-realism rules; and a batch verification checklist with distribution quotas.

### `dad/step1b_dilemmas.txt` (sub-stage 1b — first-attempt draft)

**Input:** the sampled scenarios for this batch (`{scenarios_block}`) and the count (`{count}`). Step 1a produced the scenarios by pure sampling (no prompt); the drafting instructions — design philosophy and surface rules — are inlined in this template.

**Output:** a JSON array, each `{"scenario_id", "prompt", "annotation"}`, with the prompt written to realize its scenario and the descriptive annotation fields completed. Drafts are accepted as returned (assigned labels are copied verbatim per the template; there is no per-example adherence check — the end-of-step checklist monitors distribution fidelity). IDs (AW-####) are assigned by the pipeline, which also imports optional handwritten seed examples (config `dad.dilemmas.seed_path`) before generating, and prints the verification checklist at the end of the step.

### `dad/step1c_gate.txt` (sub-stage 1c — optional, on by default)

**Input:** the scenario, the 1b draft prompt, and its annotation (for context).

**Output:** a pass/fail verdict — `{"pass", "failures"}` — never rewritten text. See the template for the checks it applies. A rejected draft is routed back through 1b (with the gate's reasons injected) and redrafted, capped at a few attempts; a scenario still failing after the cap ships with `gate_failures` stamped. Controlled by config `dad.dilemmas.gate`; verdicts are logged to `step1/gate.jsonl`.

### `dad/step1d_refine.txt` (sub-stage 1d — optional, on by default)

**Input:** the scenario description, the gate-passed 1b draft, and the dealt cards it must honor (surface form, visibility, attitude, opening move, closing move, persona, length).

**Output:** editor notes in prose, then the rewritten user message inside `<revised_user_prompt>` tags — or `<unfixable>reason</unfixable>` when no rewrite can fix the draft (the scenario is then rejected like 1a's INCOHERENT, checkpointed to `step1/refine_rejects.jsonl`). The rewrite thins corpus tics without scrubbing human texture, keeps the user from handing the assistant its answer (calibrated to the dealt visibility), enforces the dealt cards, and checks leverage/pivot, coherence, and self-containedness. The gate REDRAFTS scenario-level failures from scratch; the refine REWRITES surface problems in place. Controlled by config `dad.dilemmas.refine`; before/after pairs are logged to `step1/refinements.jsonl`.

### `dad/reasoning_library.csv` (+ `reasoning_library_ABOUT.md`)

The reasoning source for step 2. Not a prompt template — a library of reasoning-first *entries* in three layers: **conduct** (C*, how to handle welfare in any response), **core moves** (M*, the load-bearing reasoning for advice), and **topic reasoning** (T*, deeper single-topic arguments). Columns: `id, category, claim, reasoning, crux, transferable_move`. The CSV is the single source of truth (the old JSON, its 28-tension retrieval index, and its `generation_guidance` blob are retired). `reasoning_library_ABOUT.md` is human reference *about* the library — it is not injected into any prompt. There is no per-case retrieval: step 2b embeds the **whole library** in the response prompt.

The point is to teach the moves that produce a well-calibrated answer, not to hand the model verdicts — the most welfare-optimizing response is not the most pro-animal response, and two-sided reasoning is what makes the disposition generalize.

### `dad/step2_scope.txt` (sub-stage 2a — scope the case)

**Input:** the user message.

**Output:** a JSON scope map whose keys are the five axes the template defines (mirrored in `_SCOPE_AXES` in `dad_pipeline/step2_responses.py`, which validates and renders them — keep the two in sync). Reads everything from the user's message — it does not use the annotation. Written to `step2/scopes.jsonl`.

### `dad/step2_respond.txt` (sub-stage 2b — the response-generation spec)

**Input:** the whole reasoning library (`{library_block}` — every entry) + the 2a scope map (`{scope_block}`) + the user message. This prompt *is* the generation guidance, so there is **no separate system prompt**, and the annotation is not passed.

**Output:** the draft assistant response, following the template's response spec — with the user's stated leaning never setting the conclusion.

**Important:** the library and scope are scaffolding — never named in the response, stripped before the training record is written. Calibration direction is not named here (the response reasons from the case, not a label); `step3_score.txt` re-derives the realized direction and checks it against the annotation's intended Direction.

### `dad/step3_rewrite.txt`

**This is the most important prompt in the pipeline.** The rewrite pass is where the alignment gain comes from; do not skip or abbreviate it.

**Input:** the distilled constitution principles (`{principles_block}`, rendered from `constitution/constitution_principles.csv` — each with its summary and verbatim constitution quote; the explicit standard the rewrite is held to) + the user message + the draft assistant response from step 2. No system prompt is sent — the full constitution was source material for distilling the principles, not a per-call dependency — and the annotation is not passed.

**Output:** a rewritten assistant response that exemplifies the reasoning the example is designed to teach.

The template is deliberately minimal: the principles ARE the standard — the prompt adds only the conversation and the checks the template lists (keep what already meets the standard; stay fully **self-contained** — the response never mentions or alludes to a constitution, principles, or instructions, and reads as the assistant's own thinking). An earlier version carried a long requirements list and violations taxonomy; that instruction style belongs to the SDF document rewrite, and for DAD it was replaced by the principles themselves.

**What goes into the final training record:** only the user message and the rewritten assistant response. Strip the system prompt, the reasoning library scaffolding, and the annotation before writing the training record. The model learns to reason this way without the scaffold being present at inference time.

---

### `dad/step3_score.txt`

**Input:** one finished conversation from step 3 (user message + rewritten response).

**Input:** the finished conversation + the intended `{intended_direction}` and `{user_attitude}` from the annotation.

**Output:** a JSON quality report — `embodiment` (teach-why), `helpfulness`, `calibration`, `naturalness` (each 1-10), `self_contained` (boolean; any leakage is an automatic reject), plus the enforced-spec checks: `realized_direction` (judged blind from the response), `direction_match` (does it match the intended Direction? mismatch = reject), `tracks_attitude` (did the reply key on the user's tone rather than the ethics? true = reject), and `notes` naming any formulaic pattern.

The final quality gate for DAD, mirroring what `sdf/layer5.txt` does for SDF, and the enforcement half of using Direction as an enforced spec. Not yet wired into `run.py` — run it manually to spot-check step-3 output before handoff.

## Corpus Tools

### `tools/pattern_scan.txt`

**Input:** a pasted batch of generated outputs (documents or conversations) with clear delimiters.

**Output:** a JSON array of recurring structural / rhetorical / behavioral patterns found across the batch — each with evidence quotes, prevalence, a broad and a strict detection check, and a suggested fix.

Adapted from the DeepMind SDF post's scan → cluster → autorate pipeline: models pick up structural patterns from synthetic data in ways that don't show up in eval scores, so scan batches periodically and promote confirmed patterns into the preamble's named anti-pattern list.

## Key Design Decisions

**Extended thinking off.** All generation should be done without extended thinking / reasoning traces. When we refer to the model's reasoning, we mean the user-facing explanation in the response — not an internal scratchpad. Training on scratchpad content is a separate approach with different tradeoffs.

**Fresh context for rewrite steps.** Layer 4 (SDF) and step 3 (DAD) should use a new context window, not the same one that generated the original content. A model reviewing its own output in the same context tends to rationalize rather than improve.

**Diversity over volume.** A corpus of 300 genuinely diverse, high-quality documents is more valuable than 1,000 generic ones. Use the looping technique in layer 3 (brainstorm multiple angles, pick the most different ones), and let the DAD spec's coverage tally + batch checklist steer each generation batch toward the distributions the spec requires.

**The response library is sampling scaffolding only.** The reasoning library shapes draft responses (retrieval by tension, two-sided reasoning, crux named) and is never named in a response; like all scaffolding it is stripped before training records are written. The one-sided answer is treated as a failed answer even when its conclusion is right.

**Language.** The pipeline currently runs English-only (`language_distribution: {en: 1.0}` in `config.yaml`). The multilingual plumbing is still in place — restore a broader `language_distribution` to re-enable Mandarin, Hindi, and other languages, which can improve generalization and reflect the global reach of these ethical questions.

---

## What to Hand to Labs

The minimal package for a lab to reproduce this pipeline internally:

1. `constitution/constitution_claude.md` and `constitution/constitution_sentient_beings.md`
2. This entire `prompts/` directory (including `dad/dilemma_prompt_spec.md`, which governs the DAD user side)
3. A brief note on the architecture: SDF is 5 layers (fanout structure), DAD is 3 steps (spec-driven dilemma prompts → library-reasoned responses → rewrite against the distilled constitution principles), step 3 is the critical rewrite, the reasoning library and annotations are sampling scaffolding only, and final training records contain only user + assistant messages with no system prompt.

Labs may want to use their own internal models for generation, apply their own quality filters, or adapt the prompts to their alignment framework. The prompts are designed to be model-agnostic and easy to modify.
