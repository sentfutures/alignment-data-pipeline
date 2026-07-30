# Code Quality Report v2 — alignment-data-pipeline

**Date:** 2026-07-29, audited at commit `464c5bd` (the tip of `main`).
**Supersedes:** the 2026-07-10 report, which audited commit `cf8d91e` — 206 commits and 26 merged pull requests ago. This version tracks what happened to every finding from the July report, and separately audits the roughly 11,000 lines of new or rewritten code that the July report never saw.
**Purpose:** a pre-delivery quality assessment with prioritized recommendations, ahead of handing this pipeline to a frontier lab.

---

## Background: what this repo does, in one minute

The repo generates synthetic training data about animal/sentient-being welfare, using paid calls to LLM APIs. Two pipelines produce the two datasets:

- **SDF** (`sdf_pipeline/`) writes pretraining-style *documents*. Since mid-July it works as a "matrix" pipeline: a deterministic composition step deals out combinations of axes (topic, region, register, species, and so on) to guarantee corpus diversity by construction, then paid model calls draft each document (layer 3), rewrite it against the project's constitution (layer 4), and score it with an LLM judge (layer 5). Documents scoring below a threshold are dropped.
- **DAD** (`dad_pipeline/`) writes *chat transcripts*: step 1 generates user dilemma prompts in sub-stages (1a scenario planning, 1b drafting, 1c a pass/fail quality gate, 1d a refinement rewrite), step 2 generates assistant responses, step 3 rewrites the responses against the constitution.

Terms that come up repeatedly in the findings:

- **Run directory** — each pipeline invocation creates a folder under `outputs/<pipeline>/runs/<timestamp>_<label>/` holding every intermediate and final file for that run. Because a full run costs real API money (tens to hundreds of dollars), a run directory is the physical artifact of that spend — "losing" one means paying to regenerate it.
- **Checkpoint / `--resume`** — every stage records each completed item's ID in a small `_checkpoint.json` file as it goes. If a run dies partway, rerunning with `--resume` is supposed to skip everything already paid for and retry only what failed. Most money-path bugs in this report are about breaking that promise in some direction: either paid work gets re-billed, or failed work is wrongly recorded as done and never retried.
- **`stop_reason`** — the API's explanation of why the model stopped generating. `"end_turn"` means it finished normally; `"max_tokens"` means the output hit the length cap and was cut off mid-sentence; `"refusal"` means a safety classifier cut the stream. A stage that ignores `stop_reason` can mistake a truncated half-response for a finished one.
- **Fail open / fail closed** — when something goes wrong (unparseable model output, truncation), does the stage let the bad item *through* (fail open — bad data enters the dataset silently) or hold it *back* (fail closed — the item is dropped or retried)? For training-data generation, fail closed is almost always correct.
- **Backends** — `shared/api.py` can route model calls three ways: `api` (the Anthropic API, paid per token), `claude_code` (bills a contributor's Claude subscription via the Claude Code CLI, used for dev runs), and `auto` (prefers the subscription, falls back to the API). Some guarantees — like the output-length cap — only hold on the `api` backend.
- **The evidence surface** — the eval scripts under `evals/` produce the quality reports (audit reports, diversity metrics) that a receiving lab would read to judge the datasets. A bug here doesn't corrupt training data; it misstates the evidence *about* the data, which for a handoff can be worse.
- **Publishing** — `evals/publish_hf.py` uploads a finished run's dataset, manifest, and audit reports to a public Hugging Face dataset repository. It is the literal delivery mechanism, so mistakes here are public and destructive.

## How this audit was produced, and how to read the labels

1. **Mechanical baseline** — standard tools run over the whole tree: `ruff` (a Python linter), `ruff format` (code formatter, check-only), `mypy` (static type checker), `pytest` with coverage measurement, and `vulture` (dead-code finder). Nothing in the repo was modified. Numbers and their July deltas are in Appendix A.
2. **Fate ledger** — every finding from the July report was re-checked against today's code.
3. **Six specialist reviews** of the post-July code, run as parallel review agents, one per area: the SDF matrix machinery, the DAD rebuild, the `audit_dad` evidence subsystem, the `publish_hf` delivery path, shared infrastructure growth, and delivery/process readiness.
4. **Adversarial verification** — every *high*-severity finding was then independently attacked by two more agents (one told to refute the claim by reading the code, one to check the failure is actually reachable under real configurations); every *medium* finding by one. Findings that failed verification were dropped. Final tally: 21 confirmed, 9 confirmed with corrections (the corrected wording is what appears below), zero refuted. Low/informational findings were not adversarially verified and are labeled as such.

Each finding below carries a status line using these words:

- **Severity** — *high*: can destroy or re-bill paid work, put silently wrong data into a training corpus, or publish wrong/private data; *medium*: a real defect or a significant maintenance burden; *low/info*: minor.
- **Verification** — *confirmed*: survived independent adversarial verification; *confirmed with corrections*: survived, with details amended; *verified manually*: I (the lead reviewer) read the cited code myself; *unverified*: low-priority finding reported as observed, treat as probable rather than proven.
- **Fix effort** — *small*: under an hour; *medium*: about a day; *large*: multi-day.

---

## Executive summary

**The trajectory is strongly positive, and the new code is better than the old.** Since July the source has grown from 6.8k to 16.8k lines, but tests grew faster (295 → 742, all passing; coverage over the non-UI code rose from 76% to 85%). The July report's two worst findings — layer 4 silently shipping un-rewritten drafts, and layer 3 not noticing truncated output — were **fixed outright** in the SDF rebuild, using exactly the patterns that report recommended. The scattered JSON-parsing implementations were consolidated into shared, tested helpers. And the three big new subsystems all arrived with real engineering discipline (details in "What is already good").

The remaining risk concentrates in four places:

1. **The delivery path itself** (`publish_hf.py`, the script that uploads datasets to Hugging Face). It has a confirmed way to delete a *different run's* paid output from disk if the operator passes the wrong folder path; a re-publish can silently leave a version tag pointing at old, wrong data while claiming success; and audit reports already uploaded to the public dataset contain a contributor's local home-directory path.
2. **A systematic gap in output-truncation checks on the DAD response side.** The stages that write the actual training transcripts only detect one specific kind of cut-off output. A response cut mid-argument by a safety classifier — documented as actually happening on this pipeline's core topics — passes the check and becomes a training record.
3. **Evidence integrity in the audit subsystem.** On runs configured to generate multiple responses per prompt, the audit silently examines only one response per prompt while reporting as if it covered everything; several judge-output parsing failures silently change what a metric means instead of being counted as failures.
4. **The July leftovers.** The checkpoint files are still not crash-safe, the standalone scoring scripts still have zero tests and un-retryable failure behavior, and the repo still has no linter, formatter, type checker, coverage gate, or dependency lockfile — while the codebase those would protect has doubled in size.

Nothing found suggests quality decay — the opposite. But the highest-severity items now sit exactly where a handoff to an external lab is most sensitive: what gets published, and whether the published evidence describes the data honestly.

---

## Part 1 — What happened to the July findings

Numbers (1.1, 2.4, …) refer to the July report's numbering, kept for cross-reference.

| July finding | Status today | Notes |
|---|---|---|
| 1.1 Layer 4 silently shipped un-rewritten drafts on truncation (July's top finding) | **Fixed** (confirmed) | The rebuild checks `stop_reason`, never checkpoints failed work, isolates per-item failures, and aborts if everything fails. One gap: these new failure branches have no tests (→ finding 2.7) |
| 1.2 Layer 5 records fabricated 5/5/5 scores on parse failure | **Partly fixed** | The 5/5/5 fallback is now deliberate, documented, and pinned by a test — but it is still only safe because the pass threshold happens to be 7, and two sharper problems remain in the same function (→ finding 1.8) |
| 1.3 Checkpoint/JSONL files are not crash-safe | **Still open** (verified manually) | Byte-for-byte the same code, now at `shared/utils.py:437-453` |
| 1.4 DAD step 1 loses a whole batch of paid work on one failure | **Partly fixed / partly worse** | The refinement stage got retries and failure logs — but the rebuild now holds *three* paid stages in memory per pass instead of one (→ finding 1.4 below) |
| 1.5 Step-1 parsers rejected the most common malformed-JSON case | **Fixed** | Consolidated into shared, tested parsing helpers |
| 1.6 Layer 3 ignored truncated output | **Fixed** (confirmed) | |
| 1.7 Crash between file-write and checkpoint-write duplicates records | **Replaced by a new, opposite gap** (confirmed) | Stages now cross-check the output file — but the new logic *silently drops* an item in the reverse inconsistent state (→ finding 1.12) |
| 1.8 Standalone eval judges untested, record un-retryable zero scores | **Still open** (verified manually) | `score_sdf.py` unchanged; `score_dad.py` renamed to `score_dad_parked.py` with the defect intact. The *real* evidence surface is now `audit_dad.py`, which has its own confirmed issues (→ finding 1.7) |
| 1.9 `--run-id` without `--resume` silently starts a fresh (re-billed) run | **Still open** (verified manually) | Unchanged in all three pipelines |
| 1.10 Pref pipeline re-reads a live prompts file on resume | **Still open** (verified manually) | `pref_pipeline/` untouched since July |
| 2.1 Consolidate the seven JSON parsers | **Largely done** | |
| 2.2 Shared resumable-stage helper | **Not done** | The copy-pasted idiom now exists in more places than July |
| 2.4 Make the repo pip-installable | **Not done** (confirmed) | `pyproject.toml` is still 7 lines |
| 2.5 Split the `step1_dilemmas.py` god-file | **Not done** | Now 1,015 lines (was 819); the parsers moved out, everything else grew |
| 2.6 Close the risk-inverted test gaps | **Mostly done** | 742 tests; new code is 88–100% covered; the remaining gaps are, again, exactly the failure paths (→ finding 2.7) |
| 3.1–3.5 Tooling and process (linter, formatter, mypy, coverage gate, lockfile, review-bot policy) | **All still absent** (confirmed) | → finding 3.1 |
| 3.6 README misdescribed the constitution wiring | **Fixed** (verified manually) | |

---

## Part 2 — New findings: fix before delivery

### 2.1 The publish script can delete a different run's paid output from disk

**Where:** `evals/publish_hf.py:145` · **Severity: high** · **Verification: confirmed** · **Fix effort: small**

**Context.** `publish_hf.py` uploads a run's dataset to Hugging Face. Before uploading, it copies the files it intends to publish into a local "staging" folder — a scratch area where the upload is assembled. The `--staging-dir` command-line option lets the operator choose where that scratch folder lives.

**The problem.** If the chosen staging folder already exists, the script deletes it first with `shutil.rmtree` — Python's recursive directory delete, the equivalent of `rm -rf`. There is a safety guard, but it only refuses paths that belong to the run *being published*. Any other existing directory is deleted without warning — including a different run directory full of paid API output.

**How it goes wrong.** An operator publishing run A tab-completes the wrong path and passes a sibling run's folder — say `outputs/sdf/runs/2026-07-20_full-scale` — as `--staging-dir`. The guard passes (that folder doesn't belong to run A), and the script recursively deletes a $50–500 run before producing any output. This also happens with `--dry-run`, the "preview only, make no changes" flag, which an operator would reasonably assume is safe to experiment with.

**Fix.** When the script creates a staging folder, drop a marker file (for example `.publish_hf_staging`) inside it. Only auto-delete a pre-existing folder if the marker is present; otherwise refuse with an error. Add regression tests alongside the existing guard tests.

### 2.2 Half-finished model responses can become training records (DAD response stages)

**Where:** `dad_pipeline/step2_responses.py:407`, `step3_rewrite.py:91`, `baseline.py:64`, plus `shared/api.py:585` · **Severity: high** · **Verification: confirmed, two independent findings** · **Fix effort: small**

**Context.** When a model call returns, the code checks whether the output is usable before saving it. These three stages produce the actual assistant responses that become training data. Their check is: reject if the text is empty, or if `stop_reason == "max_tokens"` (output hit the length cap).

**The problem.** That check only catches *one* way output gets cut off. Two other real cases sail through:

- *Refusal cuts.* A safety classifier can stop the model mid-response, leaving several paragraphs of text with `stop_reason = "refusal"`. The pipeline's own code comments document this happening on insect-welfare topics — the corpus's core subject matter. Such a response is non-empty and its stop reason isn't `"max_tokens"`, so it passes the check.
- *Subscription-backend truncation.* On the `claude_code` and `auto` backends (the documented default for dev runs), the length cap is enforced by the Claude Code CLI rather than the API, and the stop reason it reports can be a different value or missing entirely (`None`). Output truncated this way never equals `"max_tokens"`, so the gate never fires.

**How it goes wrong.** A response is cut mid-argument by the refusal classifier. It passes the check, gets checkpointed as complete, and step 3 fluently rewrites the half-finished argument into a plausible-looking training record. The result is silently weaker training data on exactly the topics the corpus exists to teach, and nothing downstream can detect it.

**Fix.** Invert the check's polarity: instead of rejecting one known-bad value, accept only known-good ones — `stop_reason in ("end_turn", "stop_sequence")`. This is precisely how the rebuilt SDF layers already do it; the DAD response stages never received the same fix. Alternatively, normalize unknown stop values inside `shared/api.py` so every caller rejects them. One stub-model test per stage.

### 2.3 The stable-ID registry can silently renumber everything

**Where:** `dad_pipeline/id_registry.py:86` · **Severity: high** · **Verification: confirmed** · **Fix effort: medium**

**Context.** The DAD pipeline stamps every scenario, prompt, and response with a stable ID (a "gid", like `P-0042`) so that audit reports, diversity metrics, and the run viewer can refer to the same artifact across runs. The mapping lives in a single JSON file, `outputs/dad/id_registry.json`, which is checked into git and shared by every contributor and every run.

**The problem.** Three weaknesses compound:

1. If the registry file is corrupt or unparseable — for example, after a git merge leaves conflict markers in it — the code silently ignores it and starts numbering again from 1. This is currently pinned as *intended* behavior by a test.
2. Saving the registry rewrites the whole file non-atomically (no write-to-temp-then-rename), so a crash mid-save corrupts it.
3. Stages stamp gids onto their output records *before* the registry is saved, so a crash in between leaves records carrying IDs the registry never recorded.

**How it goes wrong.** Two contributors run the pipeline on separate branches; the merge of `id_registry.json` conflicts; the next run silently resets and re-issues `P-0001`, `R-0001`, … for new artifacts — numbers already stamped on committed runs for *different* artifacts. From then on, audit and diversity reports join different responses under the same ID, misattributing quality evidence across runs.

**Fix.** Fail loudly (with a "restore this file from git" message) on an unparseable registry instead of resetting — deliberately flipping the test that pins the current behavior. Make saves atomic. Save the registry together with each batch of stamped records. Document that concurrent DAD runs share one registry and are not safe to run in parallel.

### 2.4 One DAD step-1 pass holds ~120 paid calls in memory; retry budgets reset on resume

**Where:** `dad_pipeline/step1_dilemmas.py:817-1002` · **Severity: high** · **Verification: confirmed with corrections** · **Fix effort: medium**

**Context.** Step 1 generates user prompts in passes. Within one pass, three paid stages run back-to-back as parallel fan-outs: 1b drafts each prompt, 1c has a judge model gate each draft pass/fail, 1d refines the survivors. The repo's own convention (from CLAUDE.md) is that paid work should be written to disk as soon as it completes, so a crash never discards it.

**The problem.** Nothing from a pass is written to disk until a final assembly loop at the end. All 1b drafts, 1c verdicts, and 1d refinements — with default settings, roughly 120 paid calls per pass — live only in Python dictionaries until then. The July report flagged this for 1b alone; the rebuild composed two more paid stages into the same unpersisted window. Separately, the per-scenario retry budgets ("give each draft at most 3 gate attempts") and the gate's feedback text are held only in process memory, so a `--resume` after a crash starts the budgets over and loses the feedback that was supposed to steer the redraft.

**How it goes wrong.** A full-scale run crashes during the 1d fan-out (network outage, Ctrl-C). Every paid call from that pass is discarded and re-billed on resume. Independently, a scenario that had already burned two of its three gate attempts resumes with a fresh budget and can spend up to three more.

**Fix.** Append accepted 1b drafts to an intermediate file (for example `drafts_pending.jsonl`) as they complete, and have the assembly loop consume it. Rebuild attempt budgets on resume from the failure logs that already exist (`draft_failures.jsonl`, `gate.jsonl`).

### 2.5 Re-publishing with the same version tag silently leaves the tag on the old data

**Where:** `evals/publish_hf.py:558` · **Severity: medium** · **Verification: confirmed** · **Fix effort: small**

**Context.** When publishing to Hugging Face, the script can apply a git-style tag (for example `sdf-v1-fullscale`) to the uploaded snapshot, so a consumer can pin exactly that version with `revision="sdf-v1-fullscale"`.

**The problem.** Tag creation passes `exist_ok=True`, which (verified against the installed library version) means: if the tag already exists, do nothing — the tag stays on the *old* commit. The script then prints `Tag: sdf-v1-fullscale` as if it had been applied to the new upload. The code comment justifying `exist_ok=True` describes retrying a publish after a typo'd `--input` — which is exactly the scenario where the tag is already sitting on the wrong data and silently stays there.

**How it goes wrong.** Publish run A with a tag; realize the input path was wrong; re-publish the corrected data with the same tag. The corrected data uploads, the tag stays on the bad upload, the script reports success. A lab pinning the delivered snapshot by tag gets the wrong run's data forever, and the tool's own output said otherwise.

**Fix.** When the tag already exists, compare the commit it points to with the fresh upload's commit. If they differ, print a loud warning and require an explicit `--retag` flag that deletes and recreates the tag. Never print `Tag: X` when the tag was not actually moved.

### 2.6 Contributors' home-directory paths are committed and uploaded to the public dataset

**Where:** `evals/diversity.py:745`, `evals/audit_dad.py:2919`, `evals/audit_sdf.py:561` · **Severity: medium** · **Verification: confirmed** · **Fix effort: small**

**Context.** Every eval script records which run it analyzed by writing an `"input"` field into its report JSON. The publish script uploads those report files verbatim to the public Hugging Face dataset, and the repo's convention is to commit them to git as well.

**The problem.** The `"input"` field stores the full absolute path — including the contributor's macOS username and home-directory layout. Three committed report files already carry `/Users/<contributor's username>/Documents/...` publicly on GitHub.

**Fix.** Record a run-relative path (just the run-directory name) instead. Add a belt-and-braces sanitizer to the publish script's staging step that rewrites any `/Users/...` or `/home/...` prefix. Scrub the three committed files, and check whether the already-published dataset carries a path — if so, re-publish after the scrub.

### 2.7 The audit subsystem can misstate the quality evidence (three confirmed defects)

**Where:** `evals/audit_dad.py:664, 2019, 1936` · **Severity: medium ×3** · **Verification: confirmed** · **Fix effort: small–medium**

**Context.** `audit_dad.py` (about 3,000 lines) produces the corpus-quality report for DAD runs — the primary evidence a receiving lab would read. It has both free offline checks and a paid pass where judge models extract and grade the welfare considerations in each response. It runs automatically at the end of every full pipeline run.

**Three problems.**

- *Multi-sample runs are silently under-audited.* The pipeline can be configured to generate several independent responses per prompt (`per_prompt` in the config). The audit's internal lookup keys responses by prompt ID only, keeping whichever record it saw last — so on a two-samples-per-prompt run, every response-side metric describes an arbitrary half of the delivered corpus while the report claims full coverage. A verbal tic present only in the dropped half reads as 0%.
- *Judge-parse failures silently change metric semantics.* The "corpus-level distinct considerations" number is produced by an LLM consolidation call; if that call's output fails to parse, the code silently falls back to exact-string de-duplication — which counts every paraphrase as distinct and can inflate the headline pipeline-vs-baseline comparison roughly threefold — with no failure counter or note anywhere in the report. Two similar parse-fallbacks (survival verdicts, reason typing) drop or skip items uncounted.
- *The paid pass has no checkpointing.* The judge pass makes roughly 370 model calls on a default-size run, accumulating everything in memory and writing the report only at the end. Any interruption discards and re-bills all of it — and this pass auto-fires after every full pipeline run.

**Fix.** Key the response lookup by record ID (or at minimum surface a prominent "N of M final records audited" warning). Count every parse-fallback as a visible event with a field in the report. Persist per-item judge results append-only as they complete, and skip already-judged items on re-run.

### 2.8 SDF layer 5 (the LLM judge) still trusts its own inputs too much

**Where:** `sdf_pipeline/layer5_score.py:56, 80` · **Severity: medium ×2** · **Verification: confirmed** · **Fix effort: small**

**Context.** Layer 5 sends each finished document to a judge model, parses three numeric scores (alignment, realism, spec-conformance) from the reply, and drops documents scoring below the configured threshold (currently 7). When the reply doesn't parse, it deliberately records placeholder scores of 5/5/5 — safe only because 5 is below today's threshold.

**Two problems.**

- *It never checks `stop_reason`* — the only one of the four SDF stages that doesn't. A judge reply cut off by the length cap, or emptied by a refusal, lands in the 5/5/5 placeholder and is checkpointed. Under the default threshold that permanently discards a fully-paid document (draft + constitution rewrite) that a simple retry would very likely have scored fine.
- *It uses the judge's scores without validating them.* If the judge returns a score as a string (`"8"` instead of `8`), the threshold comparison crashes with a TypeError — *after* the record was checkpointed, so every subsequent `--resume` reloads the bad record and crashes at the same place. The run is wedged until someone hand-edits a JSON file. If the judge omits a key, the document is silently scored 0 and dropped, indistinguishable from a genuine quality failure.

**Fix.** Request and check `stop_reason` like the sibling stages (skip without checkpointing on anything but a normal finish). Coerce the three scores with `int(...)`, range-check 1–10, and route failures into the retry path. Add tests for a string score and a missing key.

### 2.9 A partially-written plan file silently shrinks a resumed SDF run

**Where:** `sdf_pipeline/layer12_plan.py:35` · **Severity: medium** · **Verification: confirmed** · **Fix effort: small**

**Context.** The matrix composition step writes the full list of planned documents to `prompts.jsonl`, one line per document. On `--resume`, the stage reuses that file if it exists.

**The problem.** "Exists" is treated as "complete". The file is written line-by-line, so a crash mid-write (disk-full is the realistic trigger) leaves a well-formed but short file. Every later resume silently accepts it as the whole plan.

**How it goes wrong.** A 5,000-document run dies partway through writing the plan; the operator fixes the disk and resumes; the run completes "successfully" with 3,200 documents. Beyond being short, the matrix's exact-proportions guarantee no longer holds, because a truncated prefix of the deal is not a balanced sample.

**Fix.** On reload, verify the line count equals the configured `n_prompts` and abort with a clear message otherwise (composition is seeded, so deleting the file and recomposing is safe — say so in the error). Or write the file to a temp name and rename it into place only when complete.

### 2.10 The Gemini embeddings path can silently misattribute vectors — and poison the cache

**Where:** `shared/embeddings.py:201` · **Severity: medium** · **Verification: confirmed** · **Fix effort: small**

**Context.** The diversity eval embeds every document and looks for near-duplicates. Embeddings can come from OpenAI or Gemini. Batches of texts go out; batches of vectors come back; results are cached on disk keyed by text hash so re-runs are free.

**The problem.** The OpenAI path defensively re-orders responses by index and is test-pinned. The Gemini path takes the response list as-is, with no check that it has the same length or order as the request batch. If a response is short or reordered, Python's `zip` silently pairs vector k with text k anyway — every document after the gap gets its neighbor's embedding — and the wrong vectors are then written into the persistent cache, so even re-runs stay wrong.

**Fix.** Raise an error when the response count doesn't match the batch (triggering the existing retry), mirroring the OpenAI path's test.

### 2.11 Carried over from July, still open — the fixes there still apply verbatim

**Verification: verified manually** (all four re-read in today's code)

- **Checkpoint and JSONL writes are not crash-safe** (`shared/utils.py:437-453`): the checkpoint file is rewritten in place (a kill mid-write corrupts it and bricks `--resume`), loading it has no error handling, and a partially-written last line in any output file crashes the loader. Fix: atomic write via temp-file-and-rename; tolerate one malformed final line with a warning.
- **The standalone scoring script** (`evals/score_sdf.py`) still records zero scores on parse failure, checkpoints them so re-runs never repair them, and has zero tests.
- **`--run-id` without `--resume`** is silently ignored: intending to resume a specific run but forgetting the `--resume` flag mints a brand-new run directory and re-bills the whole pipeline.
- **The preference pipeline re-reads its prompts file live on resume**, so an edited file mispairs cached responses with different prompts.

### 2.12 A checkpointed item whose output record is missing silently vanishes

**Where:** `sdf_pipeline/layer4_rewrite.py:38-41`, same idiom in layers 12/3/5 · **Severity: medium** · **Verification: confirmed** · **Fix effort: small**

**Context.** Since the rebuild, each SDF stage decides what work remains by requiring an item to be absent from *both* the output file *and* the checkpoint before it counts as pending.

**The problem.** Because the code always writes the output record *before* marking the checkpoint, an ID that is in the checkpoint but missing from the output file is an inconsistent state (a power loss at the wrong moment, or an operator deleting a corrupt line during recovery). The both-must-agree logic puts such an item in neither the "done" set nor the "pending" set — it simply vanishes from the run, with no warning, and nothing anywhere reconciles the final corpus size against the planned document count.

**Fix.** Treat presence in the output file as the sole "done" signal (the checkpoint check adds nothing given the write order), or detect checkpoint-only orphans and loudly re-queue them. Add a per-layer reconciliation line to the run summary (planned N → drafted K → rewritten J → scored I → final H) so silent shrinkage becomes visible.

---

## Part 3 — Structural improvements (worth doing soon after)

**3.1 Wrap all seven parallel workers against raised API errors** — *medium severity, confirmed, ~a day.* When stages fan work out across threads, an API call that exhausts its retries raises an exception that aborts the whole stage — discarding completed work from sibling threads that hadn't been written to disk yet. July flagged two sites; the rebuilds added five more with the same shape (step 2's scope/select/respond calls, step 1's plan and draft workers, step 3, baseline). One uniform pattern fixes all seven: catch the terminal error in the worker, return a failure marker, log it, skip checkpointing so `--resume` retries that one item.

**3.2 Make `backfill_gids` crash-safe** — *medium, confirmed with corrections, small.* This utility retrofits stable IDs onto older runs by rewriting their files **in place** with a plain truncate-and-write. An interruption mid-rewrite destroys a file that, for uncommitted runs, git cannot restore. It can also stamp literal `null` IDs, which later runs then re-process forever. Fix: write to temp file and rename; skip inserting null values.

**3.3 Stamp records whose quality gate never ran** — *medium, confirmed, small.* When the 1c gate produces unusable output for a draft after all retries, the draft ships anyway (a reasonable fail-open choice) — but unlike its sibling paths, nothing on the record says so. A reviewer filtering the corpus by quality flags concludes these records passed the gate. Fix: stamp `gate_unusable: true`, and deliberately flip the test that pins today's unstamped behavior.

**3.4 Split the two god-files** — *medium, confirmed with corrections, multi-day.* `evals/audit_dad.py` is now 2,997 lines with visible seam decay (duplicated section banners, 389 lines of section code *after* the "main" banner, one 314-line function). `step1_dilemmas.run()` is a single 534-line function with five nested closures sharing six mutable dictionaries. Both have concrete, migration-safe split plans in the findings ledger (package split with a re-export shim; pass-state extraction that also enables the resume fix in 2.4).

**3.5 Move the viewer's headline math into tested code** — *medium, confirmed, ~a day.* The audit results page (1,337 lines, zero test coverage) computes its two boldface "+X% considerations" claims inline, in two places, with two subtly different definitions that can diverge on older reports — the exact numbers a lab reads first. Move both into one pure, tested function in `viewer/rendering.py`, which is the pattern the rest of the page already follows.

**3.6 Don't re-bill the automatic evals on resume** — *unverified (info), small.* Resuming an already-complete run makes zero pipeline calls but re-fires the paid audit and embedding evals from scratch. A done-marker keyed on the corpus state, plus a `--force-evals` flag, fixes it.

**3.7 Close the failure-path test gaps** — *medium, confirmed, ~a day.* A familiar July pattern recurs: the untested lines are exactly the failure handling. Layer 4's exception/truncation/abort branches (protecting the most expensive stage) have zero tests — the equivalent tests already exist for the plan stage and just need porting. `audit_dad`'s `main()` and its "stakes" data join are untested (a key rename in step 2 would silently strip case stakes from every delivery grade). Three of `publish_hf`'s Hub-API wrappers carry safety-relevant hard-coded arguments (`repo_type="dataset"`) that no test asserts.

**3.8 Count token-level truncation in the diversity eval** — *medium, confirmed with corrections, small.* Embedding inputs are silently cut to the model's token window (8,192 tokens for OpenAI, 2,048 for Gemini), but the report only counts *character*-level truncation. Documents in dense scripts (the matrix deliberately produces Japanese/Chinese documents) can be embedded as prefixes while the report says nothing was truncated.

**3.9 `publish_hf` hardening bundle** — *unverified (low/info), small each.* Pre-publish sanity interlocks (refuse an empty corpus; require an explicit flag to publish a run labeled `dev` or built from a dirty working tree); make `--dry-run` also print the destructive half of what a real publish would do (the delete patterns and the tag); pin the `huggingface_hub` dependency and verify staged files actually landed after upload; name which pipeline stage each model ran on the dataset card instead of a de-duplicated model list; derive the card's language metadata from the run manifest instead of hardcoding English.

---

## Part 4 — Process, documentation, delivery hygiene

**4.1 Still no toolchain, now at 16.8k lines** — *medium, confirmed with corrections, multi-day overall.* There is still no linter, formatter, type checker, or coverage gate anywhere: CI runs a syntax check and the tests, nothing else. Today's numbers: 64 of 71 files differ from standard formatting; ~140 substantive lint findings (now dominated by complexity warnings — one function scores 73 on a metric where 10 is the conventional ceiling); 157–174 type errors depending on environment. One note of caution from my own re-check: the linter flags two loop-variable captures in step 1 as potential bugs — **I verified both are benign as written** (each closure is fully consumed before the variable changes), the same "latent trap, not live bug" class as July. The July adoption plan still applies: one mechanical format commit (with a `.git-blame-ignore-revs` entry so blame survives), a curated lint rule set, per-module gradual typing, and a coverage floor — which would pass today at ~85% over the non-UI code and can only get cheaper to adopt than it will be later.

**4.2 Dependencies are not reproducible** — *medium, confirmed, small.* `numpy` is imported by four modules but never declared (it arrives only as a transitive dependency of pandas); two declared packages are never imported; seven have no version constraint; there is no lockfile. Consequence for delivery: the exact dependency set that produced the shipped corpora and their audit numbers cannot be reconstructed. Fix: declare numpy, drop the dead two, commit a compiled lockfile, and have CI install from it.

**4.3 CLAUDE.md documents a pipeline that doesn't exist** — *medium, confirmed, small.* The contributor guide prominently warns about an "experimental, parked" `dad_segmented/` directory and lists it in the directory map — but no such directory exists on `main`, and git history shows it never has on any pushed branch. Anyone (human or Claude session) reading the guide first goes looking for it. Either push the experiment or delete both references; add a pre-delivery check that every path named in the docs exists in `git ls-files`.

**4.4 Stale organization URLs** — *medium, confirmed with corrections, small.* The README's clone command and the git workflow guide still point at the pre-rename GitHub organization. The verifier's correction is worth passing on: GitHub transfers leave a permanent redirect and the old org name is still under the team's control, so the links *work* and the name cannot currently be squatted — this is documentation hygiene rather than an active supply-chain hole. Update both anyway.

**4.5 README drift** — *medium, confirmed, small.* The quick-test section still quotes a cost of "$0.05–0.15" for what now costs about $4 (roughly 30× stale — open issue #54 already reports this). The DAD step-1 description still describes the old sub-stage structure rather than the shipped draft/gate/refine design.

**4.6 Triage the issue tracker before delivery** — *medium, confirmed, small.* Open issue #53 states "SDF is currently broken out-of-the-box on main" — describing a crash in code the matrix rebuild deleted entirely. A lab doing due diligence reads open issues; an un-rebutted "the flagship pipeline crashes" claim materially misstates current quality. Close or retitle #53, re-verify #24 and #30, and fix #54 together with the README cost line.

**4.7 Smaller items** — *unverified (low/info).* The vendored 1.68 MB tokenizer data file has no README explaining what it is, its license, or how to regenerate it. The committed shared cost-log file guarantees merge conflicts between contributors (gitignore it; per-run logs are the real record). The cost table prices the default model at its post-introductory rate, overstating logged spend ~1.5× through August — worth a comment wherever cost evidence is quoted. Two July items also still apply: the automated PR reviewer's prompt still doesn't check the repo's own PR policies, and `call_claude`'s return type still defeats type-checking at every call site.

---

## What is already good (and better than July)

- **The rebuilds fixed July's worst findings with the recommended patterns.** The SDF plan/draft/rewrite stages are now models of fail-closed engineering: per-item failure isolation, skip-without-checkpoint retries, and an abort if everything fails. The plan stage's failure-path tests are exemplary.
- **The matrix machinery's diversity guarantee is real.** Exact proportional dealing, all-or-nothing validation of the axis weights, and a contract-test suite that pins the live prompt templates to the live axis definitions.
- **`publish_hf` matched every documented safety claim we checked** — the delete patterns are correctly scoped to the pipeline being published, the staging guard exists with four regression tests, `--dry-run` provably makes zero network calls, and the dataset card builds from measured audit fields. (The findings above are gaps *beyond* its documented contract, not contradictions of it.)
- **`audit_dad` designed out most of July's "parse failures counted as clean" class**: raw judge failures are logged, extraction failures render as visible gaps ("a missing bar is a gap, not a zero"), and showcase quotes are verbatim-validated fail-closed.
- **Tests tripled and tracked the new code**: 295 → 742, with the new modules at 88–100% coverage and 85% across all non-UI code.
- **Backend growth is disciplined**: the Bedrock backend reuses the existing retry/pricing machinery (verified); the `auto` backend's demotion behavior is loud, sticky, and correctly keeps the experiment's control arm on the API; the pricing tables and cache multipliers check out against published rates.
- **Hygiene scan clean**: 998 committed output files, no API keys, tokens, or emails (the one leak is the home-path finding, 2.6).
- **CLAUDE.md is unusually well-maintained** — the publishing contract, auto-eval wiring, and matrix config documentation all verified accurate against the code. The `dad_segmented` ghost (4.3) is the notable exception.

---

## Appendix A — Mechanical baseline, July → now

Tool notes: *ruff* is a Python linter (E501 is its line-length rule — cosmetic); *mypy* is a static type checker; *vulture* finds dead code; coverage is the fraction of code lines executed by the test suite.

| Measure | 2026-07-10 (`cf8d91e`) | 2026-07-29 (`464c5bd`) |
|---|---|---|
| Source lines (excluding tests) | ~6,841 in 33 files | 16,838 in 48 files |
| Test lines / test count | 3,319 / 295 | 10,149 / 742 (all passing) |
| Coverage: total / excluding UI code | 58% / ~76% | 68% / **84.8%** |
| Zero-coverage modules | score_sdf, score_dad, rate.py, viewer UI | score_sdf, score_dad_parked, rate.py, viewer UI (incl. the new audit page) |
| ruff findings | 541 (430 line-length; 13 complexity) | 1,215 (997 line-length; 24 complexity, 21 unchecked zips, 2 benign loop-captures) |
| Files differing from standard format | 45 of 52 | 64 of 71 |
| mypy errors (environment-dependent) | 69 | 157–174 |
| Dead code | 1 unused variable | 1 unused variable (the same one) |
| Dependencies | version floors only, no lockfile; numpy missing | same, plus three new deps; numpy still missing |
| Largest source file | step1_dilemmas.py, 819 lines | audit_dad.py, 2,997 lines (step1_dilemmas now 1,015) |

Tool versions differ between the two runs, so counts are indicative rather than strictly comparable. Reproduction commands are in the PR description.

## Appendix B — Findings ledger

50 findings across the six review areas; verification tally: 21 confirmed, 9 confirmed with corrections, 20 unverified (all low/info — unverified by design), zero refuted. Low/info items not detailed above include: an axis defined with zero values silently emptying the whole matrix deal; the plan-coherence check misreading its own instructions in a malformed reply; the jargon comparison mixing unequal population sizes; a malformed curated-YAML file crashing the entire audit; carried-forward paid audit data lacking a staleness marker; and older runs missing a field that crashes record assembly on resume. The full machine-readable ledger (evidence, failure scenarios, recommendations, and verifier notes per finding) is preserved in the audit workflow's journal at `~/.claude/projects/-Users-declan-Projects-alignment-data-pipeline/00008bca-87eb-47f0-935e-004ea49fee99/subagents/workflows/wf_53baf5b3-219/journal.jsonl`; the July ledger remains at `.../wf_3e2c7be7-0ed/findings.json`.
