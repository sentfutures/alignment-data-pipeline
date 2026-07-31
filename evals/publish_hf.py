#!/usr/bin/env python3
"""Publish a run's final corpus + audit reports as a Hugging Face dataset.

One repo holds BOTH pipelines' corpora as separate HF "configs" (each gets its
own selector in the dataset viewer), so a run is staged under its own
per-pipeline directory rather than at the repo root:

    README.md            <- one card declaring every config present
    sdf/  sdf_corpus.jsonl, run_manifest.json, audit/*
    dad/  dad_corpus.jsonl, run_manifest.json, audit/*

Each dataset dir holds the final corpus jsonl, run_manifest.json for
provenance, and (if present) every audit/*.{json,jsonl,html} file — globbed
rather than named so a future run's eval additions/omissions are picked up or
skipped automatically. Republishing a run clears only ITS OWN
`<pipeline>/audit/*` on the Hub (delete_patterns), so a file only the previous
run of that pipeline produced can't linger — while the sibling pipeline's data
is never touched.

Publishing one pipeline regenerates the whole card, so the sibling's section
would be lost if we only looked locally. fetch_sibling() downloads the
sibling's small metadata (run_manifest.json + audit/*.json) — never its
multi-MB corpus, never its HTML — and the card is rebuilt from both. Since
upload_folder only adds/overwrites paths present in the staged folder, the
sibling's corpus is never re-uploaded either.

The card is built entirely from measured fields already sitting in the audit
JSONs — no interpretive prose is authored here. Audit files with a real,
committed generator (audit_report.json, compliance_report.json,
diversity_report.json — see build_metrics_rows) each contribute one
metrics-table row, cited by source filename; a file that's absent, or present
but missing an expected field, just omits its row. Every OTHER file under
audit/ is still staged/uploaded (glob-based, not a fixed list) and still
listed in the card, just without a bespoke row — see build_metrics_rows'
docstring for why card_fidelity_report.json/realism_ablation.json/
vendi_curve.json (one-off artifacts of a specific historical run, no
generator in this repo) are handled that way rather than hardcoded.
report_content.json (the one editorial input among the audit files — curated
excerpts/translations and report-section prose, read by evals/report_sdf.py)
is excluded from the upload entirely: it's already fully baked into
corpus_report.html, so nothing in it would be invisible to a Hub visitor. Its
title/subtitle strings, already curated by the pipeline for this run, are
reused verbatim as the card header when present.

Tags are repo-wide, so with more than one dataset in the repo they should be
prefixed per pipeline (`sdf-v1-…`, `dad-v1-…`) to stay unambiguous.

Usage:
  REPO=sentientfutures/animal-welfare-training-dataset
  python evals/publish_hf.py --input outputs/sdf/latest --repo-id $REPO --dry-run
  python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id $REPO \
      --tag sdf-v1-fullscale-500-opus5
  python evals/publish_hf.py --input outputs/dad/runs/<run_id> --repo-id $REPO \
      --tag dad-v1-archetypes-40

Requires a Hugging Face token with write access to the target repo/org:
either ``HF_TOKEN`` in .env (checked first) or a one-time ``huggingface-cli
login`` (its cached token is the fallback) — --dry-run needs neither and
makes no network calls.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils

load_dotenv()

CORPUS_FILENAMES = ("sdf_corpus.jsonl", "dad_corpus.jsonl")

# Sidecar holding just the title/subtitle this dataset's card section uses.
# report_content.json itself is never uploaded (it's a large editorial input,
# already baked into corpus_report.html), but without SOME persisted copy of
# those two strings a sibling's curated heading is unrecoverable on the next
# publish, so its section would silently downgrade to the generic name every
# time the other pipeline is published. Only the two strings already rendered
# publicly in the card go in here — nothing otherwise invisible.
CARD_META_FILENAME = "card_meta.json"

# ISO 639-1 codes for every language name sdf_pipeline/compose_prompts.py's
# derive_language() can produce (the `culture` axis in prompts/sdf/
# variables.txt) — no existing name->code mapping exists anywhere in the
# repo to reuse; this set is closed and small, so a literal dict here is
# the right amount of code, not a premature abstraction.
LANGUAGE_CODES = {
    "English": "en", "Portuguese": "pt", "Spanish": "es", "German": "de",
    "French": "fr", "Polish": "pl", "Norwegian": "no", "Hindi": "hi",
    "Urdu": "ur", "Bengali": "bn", "Mandarin Chinese": "zh", "Japanese": "ja",
    "Korean": "ko", "Indonesian": "id", "Vietnamese": "vi", "Arabic": "ar",
}


def resolve_corpus_file(input_arg: str) -> tuple[Path, str]:
    """Return (run_dir, corpus_filename) for an SDF or DAD run directory."""
    run_dir = Path(input_arg)
    if not run_dir.is_dir():
        raise SystemExit(f"Not a run directory: {run_dir}")
    for name in CORPUS_FILENAMES:
        if (run_dir / "final" / name).exists():
            return run_dir, name
    raise SystemExit(
        f"No final/sdf_corpus.jsonl or final/dad_corpus.jsonl under {run_dir}"
    )


def flatten_dad_corpus(src: Path, dst: Path, append: bool = False) -> int:
    """Write the published form of a DAD corpus: one flat record per example
    (example_gid, user_prompt, assistant_response) instead of the training
    format's messages array, so the Hub viewer shows one readable column per
    field with no role/content nesting.

    Deliberately carries NO per-row run column, even when several runs are
    concatenated into one published corpus (append=True for every run after
    the first). Row-to-run attribution comes from the repo instead:
    example_gid is globally unique and content-keyed via the git-tracked
    dad/id_registry.json, so `git grep <gid> -- outputs/dad/runs` resolves any
    published row to exactly one committed run dir. The card's per-run table
    records which runs went into a combined corpus. A repeated run_id string
    on every row bought nothing that trace doesn't already give, and it
    dominated the viewer's first screen.

    The run's own final/dad_corpus.jsonl keeps the SFT chat shape — only the
    staged copy is flattened. A record without a user+assistant string pair
    aborts the publish rather than uploading a mangled row. Returns the number
    of records written.
    """
    n = 0
    with open(src, encoding="utf-8") as fin, \
         open(dst, "a" if append else "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            by_role: dict = {}
            for msg in record.get("messages") or []:
                if isinstance(msg, dict):
                    by_role.setdefault(msg.get("role"), msg.get("content"))
            if not (isinstance(by_role.get("user"), str)
                    and isinstance(by_role.get("assistant"), str)):
                rid = record.get("example_gid") or record.get("record_id") \
                    or f"line {line_no}"
                raise SystemExit(
                    f"{src}: record {rid} has no user+assistant message pair "
                    f"— refusing to publish a mangled row"
                )
            fout.write(json.dumps({
                "example_gid": record.get("example_gid"),
                "user_prompt": by_role["user"],
                "assistant_response": by_role["assistant"],
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def stage_run(run_dirs: list[Path], corpus_name: str, staging_dir: Path,
              pipeline_tag: str) -> dict:
    """Copy the publishable subset of run dir(s) into staging_dir/<pipeline_tag>/.

    The per-pipeline subdirectory is what lets one repo hold both corpora as
    separate HF configs. Returns a manifest dict of what was staged, used both
    for the dataset card and for logging what --dry-run would have uploaded.

    With ONE run dir the layout is the original single-run shape
    (run_manifest.json + audit/*). With several (DAD only — enforced in
    main()), the flattened corpora are concatenated into one jsonl and the
    per-run files move under run-scoped paths so they can't collide:
    manifests/<run_id>.json and audit/<run_id>/*. Those per-run manifests,
    plus the card table built from them, are the combined corpus's provenance
    record — the rows themselves carry no run column (see flatten_dad_corpus).
    """
    # Refuse a --staging-dir that equals or contains any run dir, OR either of
    # the two specific subtrees this function reads from (final/, audit/):
    # rmtree below would delete data we're about to read before the copy even
    # runs. Checking only run_dir itself isn't enough — a --staging-dir
    # pointing at run_dir/final or run_dir/audit directly (an easy typo, since
    # those are real, well-known subdirectory names on every run) would slip
    # past a run_dir-only check while still destroying the corpus or audit
    # reports.
    staging_real = staging_dir.resolve()
    for run_dir in run_dirs:
        for guarded, label in (
            (run_dir.resolve(), "the run directory"),
            ((run_dir / "final").resolve(), "the run's final/ directory"),
            ((run_dir / "audit").resolve(), "the run's audit/ directory"),
        ):
            if guarded.is_relative_to(staging_real):
                raise SystemExit(
                    f"--staging-dir {staging_dir} equals or contains {label} ({guarded}) "
                    f"— refusing to delete it. Pick a --staging-dir outside the run."
                )

    # Wipe first: a reused --staging-dir (e.g. re-running after fixing a typo'd
    # --input) must reflect only THIS run — otherwise leftover files from an
    # earlier invocation ride along into upload_folder silently mixed with
    # this run's data.
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    utils.ensure_dir(staging_dir)
    # Everything for this run lives under <staging>/<pipeline_tag>/ so the
    # sibling pipeline can occupy its own sibling directory in the same repo.
    dataset_dir = utils.ensure_dir(staging_dir / pipeline_tag)
    multi = len(run_dirs) > 1
    staged: dict = {"pipeline": pipeline_tag, "corpus_file": None,
                    "manifest_file": None, "audit_files": [], "n_docs": 0,
                    "runs": []}

    corpus_dst = dataset_dir / corpus_name
    for i, run_dir in enumerate(run_dirs):
        manifest = _load_json(run_dir / "run_manifest.json") or {}
        run_id = manifest.get("run_id") or run_dir.name

        corpus_src = run_dir / "final" / corpus_name
        if corpus_name == "dad_corpus.jsonl":
            n = flatten_dad_corpus(corpus_src, corpus_dst, append=(i > 0))
        else:
            shutil.copy2(corpus_src, corpus_dst)
            with open(corpus_dst, encoding="utf-8") as f:
                n = sum(1 for _ in f)
        staged["n_docs"] += n
        staged["runs"].append({"run_id": run_id, "n_docs": n})

        manifest_src = run_dir / "run_manifest.json"
        if manifest_src.exists():
            if multi:
                dst = utils.ensure_dir(dataset_dir / "manifests") / f"{run_id}.json"
            else:
                dst = dataset_dir / "run_manifest.json"
                staged["manifest_file"] = "run_manifest.json"
            shutil.copy2(manifest_src, dst)

        audit_src = run_dir / "audit"
        if audit_src.is_dir():
            audit_dst = utils.ensure_dir(
                dataset_dir / "audit" / run_id if multi else dataset_dir / "audit")
            # *.jsonl too: evals/audit_dad.py writes audit/tic_candidates.jsonl
            # and audit/reason_failures.jsonl for DAD runs — a fixed
            # *.json/*.html pattern silently dropped both.
            for pattern in ("*.json", "*.jsonl", "*.html"):
                for f in sorted(audit_src.glob(pattern)):
                    if f.name == "report_content.json":
                        continue  # editorial input, already baked into corpus_report.html
                    shutil.copy2(f, audit_dst / f.name)
                    staged["audit_files"].append(
                        f"{run_id}/{f.name}" if multi else f.name)

    staged["corpus_file"] = corpus_name
    return staged


def _get(d: dict, *path, default=None):
    """Nested dict lookup; returns default if any key is missing."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_metrics_rows(dataset_dir: Path,
                       run_id: str | None = None) -> list[tuple[str, str, str]]:
    """(label, value, source_filename) rows, one per known audit file that's
    present in this ONE dataset's dir AND has the fields this function expects.
    run_id scopes the lookup to a combined publish's per-run audit subdir
    (audit/<run_id>/ — see stage_run) instead of the single-run audit/.

    Every value is a measured field lifted verbatim from the file's own
    summary — no thresholds, verdicts, or causal claims added here.

    Deliberately limited to files with an actual committed generator —
    evals/audit_sdf.py, evals/diversity.py (both on main), and
    evals/compliance_sdf.py (landing via PR #103) — rather than every audit/
    filename PR #103's own run happened to carry. card_fidelity_report.json
    and realism_ablation.json have no generator anywhere in this repo (a
    one-off local analysis for that run), and vendi_curve.json is itself an
    editorial input report_sdf.py reads rather than measures. Special-casing
    their exact schemas here would be dead code for every other run. They're
    still staged/uploaded (stage_run globs audit/ unconditionally) and still
    surface in the card's "additional files" line — just without a row a
    future run has no way to reproduce.
    """
    audit_dir = dataset_dir / "audit" / run_id if run_id else dataset_dir / "audit"
    rows: list[tuple[str, str, str]] = []

    d = _load_json(audit_dir / "audit_report.json")
    if d is not None:
        # The two pipelines' audit generators report their corpus size under
        # different keys and neither writes the other's: evals/audit_sdf.py
        # writes n_docs (documents), evals/audit_dad.py writes n_prompts
        # (chat examples). Without the fallback a DAD card renders with only
        # the diversity row.
        n, label = _get(d, "n_docs"), "Documents (offline audit)"
        if n is None:
            n, label = _get(d, "n_prompts"), "Examples (offline audit)"
        if n is not None:
            rows.append((label, str(n), "audit_report.json"))

    d = _load_json(audit_dir / "compliance_report.json")
    if d is not None:
        judged, clean, frac = _get(d, "judged"), _get(d, "clean_documents"), _get(d, "clean_frac")
        if judged is not None and clean is not None:
            rows.append((
                "Constitutional compliance",
                f"{clean} of {judged} judged clean ({frac:.1%})" if frac is not None
                else f"{clean} of {judged} judged clean",
                "compliance_report.json",
            ))

    d = _load_json(audit_dir / "diversity_report.json")
    if d is not None:
        n = _get(d, "n_records")
        vendi_score, vendi_ratio = _get(d, "vendi", "score"), _get(d, "vendi", "ratio")
        mpc = _get(d, "mean_pairwise_cosine")
        if None not in (n, vendi_score, vendi_ratio):
            # "records", not "docs": this row is shared by both pipelines and
            # DAD's records are chat examples, not documents.
            detail = f"Vendi {vendi_score:.1f} effective records of {n} (ratio {vendi_ratio:.3f})"
            if mpc is not None:
                detail += f", mean pairwise cosine {mpc:.3f}"
            rows.append(("Semantic diversity", detail, "diversity_report.json"))

    return rows


def detected_languages(dataset_dir: Path, pipeline_tag: str) -> list[str]:
    """ISO 639-1 codes actually present in this ONE dataset, not a hardcoded
    guess. The card's repo-wide `language:` field is the UNION over datasets.

    SDF runs get this from audit_report.json's own composition.language
    breakdown (evals/audit_sdf.py, keyed by the same full names
    LANGUAGE_CODES maps) — the culture matrix deliberately samples mostly
    non-English documents, so hardcoding "en" would misdeclare the card's
    language metadata for every real SDF run. DAD's audit_report.json has no
    such breakdown (dilemmas are English-only by the dad.language_distribution
    default), so DAD runs — and any SDF run missing the file or whose
    language names don't map — fall back to ["en"] rather than guessing.
    """
    if pipeline_tag != "sdf":
        return ["en"]
    d = _load_json(dataset_dir / "audit" / "audit_report.json")
    names = _get(d, "composition", "language", default={}) if d else {}
    codes = sorted({LANGUAGE_CODES[n] for n in names if n in LANGUAGE_CODES})
    return codes or ["en"]


PIPELINE_NAMES = {"sdf": "Synthetic documents", "dad": "Difficult advice Q&A"}
# What one record counts AS, per pipeline — SDF ships documents, DAD ships
# chat examples. Used for the "N <unit>." line under each section heading.
PIPELINE_UNITS = {"sdf": "documents", "dad": "chat examples"}
# Human-readable HF dataset-viewer config names (the tab labels), decoupled
# from the internal pipeline_tag used for staged directory paths, discovery
# tags, and --tag prefixes — those stay "sdf"/"dad".
CONFIG_LABELS = {"sdf": "synthetic documents", "dad": "difficult advice Q&A"}


def models_used(manifest: dict, pipeline_tag: str) -> tuple[str | None, list[str]]:
    """(default model, sorted per-stage override models) for a run.

    The manifest's top-level `model` alone misdescribes most real runs: both
    published corpora were generated with Opus on the stages that matter while
    that field still reads claude-sonnet-5, so a card showing only `model`
    would tell a reader it's a Sonnet dataset. The per-stage `*_model` knobs
    under config.<pipeline> are what actually generated the records.

    Only the pipeline's own top-level knobs are scanned, so DAD's nested
    baseline.model is excluded — the baseline arm is a control that is never
    trained on and never reaches the published corpus.
    """
    cfg = _get(manifest, "config", default={}) or {}
    default = manifest.get("model") or cfg.get("model")
    overrides = sorted({
        v for k, v in (_get(cfg, pipeline_tag, default={}) or {}).items()
        if k.endswith("_model") and isinstance(v, str) and v != default
    })
    return default, overrides


def code_state(manifest: dict) -> str:
    """Whether the run's working tree was clean at generation time.

    The git commit alone overstates what it pins down. shared.utils records
    git_dirty/git_dirty_files precisely because runs are typically generated
    with uncommitted changes on top of the recorded SHA — every DAD run
    published so far was dirty, several with pipeline code and prompt templates
    among the modified files. So the commit is the nearest committed ancestor
    of the generating code, not the generating code itself, and a card that
    prints a bare SHA implies a reproducibility it doesn't have. The counts are
    already uploaded per run in manifests/<run_id>.json — this only surfaces
    them for a human reader.

    A manifest predating the fields renders "unknown" rather than "clean":
    absent evidence is not evidence of a clean tree.
    """
    dirty = manifest.get("git_dirty")
    if dirty is None:
        return "unknown"
    if not dirty:
        return "clean"
    n = len(manifest.get("git_dirty_files") or [])
    return f"dirty ({n} uncommitted file{'s' if n != 1 else ''})" if n else "dirty"


def _dataset_section(ds: dict) -> list[str]:
    """The card's prose block for ONE dataset: heading, count, provenance,
    measured metrics, and pointers to its audit files."""
    tag, dataset_dir, staged = ds["pipeline"], ds["dir"], ds["staged"]
    content = ds.get("content") or {}

    heading = content.get("title") or PIPELINE_NAMES.get(tag, f"{tag.upper()} corpus")
    lines = ["", f"## {heading} (`{CONFIG_LABELS.get(tag, tag)}` config)"]
    if content.get("subtitle"):
        lines += ["", content["subtitle"]]

    n = staged.get("n_docs")
    if n:
        lines += ["", f"{n} {PIPELINE_UNITS.get(tag, 'records')}."]

    # Rendered from `content`, not from live git, so it describes the publish
    # that actually happened — and so the sibling's own stamp survives being
    # regenerated by the other pipeline's publish (see the sidecar write).
    #
    # ABOVE the layout branch below, which returns early for a combined publish:
    # placed after it, the warning would silently vanish from exactly the
    # multi-run corpora most likely to mix merged and unmerged runs. Being first
    # also keeps it out from under the per-run table.
    unmerged = content.get("unmerged") or {}
    if unmerged:
        lines += ["", "> **Unmerged code warning.**"]
        # Named per run, not collapsed. A reader CAN trace any row back to its
        # run — example_gid resolves to exactly one committed run dir in the
        # repo (see flatten_dad_corpus) — so naming the runs is what makes that
        # trace tell them WHICH rows came from unreviewed code. Collapsed into
        # one verdict, the warning would name no run and the trace would have
        # nothing to resolve against.
        for run in unmerged.get("runs") or []:
            run_id = run.get("run_id") or "unknown run"
            branch = run.get("branch") or "unknown"
            commit = run.get("commit") or "unknown"
            lines += [
                f"> Run `{run_id}` (branch `{branch}`, commit `{commit}`) was "
                "generated from code not verified to be in the repository's "
                "`main` branch, so it may not have been reviewed.",
            ]
        if unmerged.get("publish_branch"):
            lines += [
                f"> Published from branch `{unmerged['publish_branch']}`, which "
                "was not verified to be in `main` at publish time.",
            ]

    # A combined publish (several runs in one corpus — see stage_run) carries
    # per-run manifests under manifests/ instead of one run_manifest.json.
    # Checked on disk, not via staged["runs"], so a sibling fetched from the
    # Hub (which has no staging metadata) renders the same way.
    manifests_dir = dataset_dir / "manifests"
    if manifests_dir.is_dir():
        counts = {r["run_id"]: r["n_docs"] for r in (staged.get("runs") or [])}
        lines += ["", "Combined from several runs. The rows carry no run "
                      "column; this table is the provenance record for what "
                      "went into the corpus."]
        lines += ["", "| run | examples | default model | per-stage models "
                      "| backend | git commit | code state |",
                  "| --- | --- | --- | --- | --- | --- | --- |"]
        run_manifests = [_load_json(p) or {}
                         for p in sorted(manifests_dir.glob("*.json"))]
        for m in run_manifests:
            rid = m.get("run_id", "unknown")
            default_model, overrides = models_used(m, tag)
            lines.append(
                f"| `{rid}` | {counts.get(rid, '—')} "
                f"| `{default_model or 'unknown'}` "
                f"| {', '.join(f'`{x}`' for x in overrides) or '—'} "
                f"| `{_get(m, 'config', 'backend', default='unknown')}` "
                f"| `{m.get('git_commit', 'unknown')}` "
                f"| {code_state(m)} |")
        for m in run_manifests:
            rid = m.get("run_id", "unknown")
            rows = build_metrics_rows(dataset_dir, run_id=rid)
            if rows:
                lines += ["", f"**`{rid}`** — "
                          + "; ".join(f"{label}: {value}" for label, value, _ in rows)
                          + f". Audit files under `{tag}/audit/{rid}/`."]
        return lines

    manifest = _load_json(dataset_dir / "run_manifest.json") or {}
    if manifest:
        default_model, overrides = models_used(manifest, tag)
        lines += [
            "",
            f"- **run_id**: `{manifest.get('run_id', 'unknown')}`",
            f"- **label**: `{manifest.get('label', 'unknown')}`",
            f"- **git commit**: `{manifest.get('git_commit', 'unknown')}`",
            f"- **code state**: {code_state(manifest)}",
            f"- **default model**: `{default_model or 'unknown'}`",
        ]
        if overrides:
            lines += ["- **per-stage models**: "
                      + ", ".join(f"`{m}`" for m in overrides)]
        lines += [
            f"- **backend**: `{_get(manifest, 'config', 'backend', default='unknown')}`",
        ]

    metrics_rows = build_metrics_rows(dataset_dir)
    if metrics_rows:
        lines += ["", "| Metric | Value |", "| --- | --- |"]
        lines += [f"| {label} | {value} |" for label, value, _source in metrics_rows]

    audit_files = staged.get("audit_files") or []
    if "corpus_report.html" in audit_files:
        lines += ["", f"See `{tag}/audit/corpus_report.html` for the full interactive "
                       "report (self-contained; open directly in a browser)."]
    summarized = {source for _, _, source in metrics_rows}
    extra = [f for f in audit_files
             if f != "corpus_report.html" and f not in summarized]
    if extra:
        lines += ["", f"Additional machine-readable audit files under `{tag}/audit/`: "
                      + ", ".join(f"`{f}`" for f in sorted(extra)) + "."]
    return lines


def build_card(datasets: list[dict], license_id: str, pretty_name: str) -> str:
    """The repo-level README.md, covering EVERY dataset in `datasets`.

    `datasets` is an ordered list of {pipeline, dir, staged, content} — the
    first entry becomes the viewer's default config. The card is regenerated
    whole on each publish, so every dataset that should survive must be
    present here (see fetch_sibling).
    """
    # language: is repo-wide, so it must be the union across datasets — SDF
    # spans 16 languages while DAD is English-only, and declaring either one
    # alone would misdescribe the repo.
    languages = sorted({
        code
        for ds in datasets
        for code in detected_languages(ds["dir"], ds["pipeline"])
    }) or ["en"]

    configs = []
    for i, ds in enumerate(datasets):
        entry = {
            "config_name": CONFIG_LABELS.get(ds["pipeline"], ds["pipeline"]),
            "data_files": [
                {"split": "train",
                 "path": f"{ds['pipeline']}/{ds['staged']['corpus_file']}"},
            ],
        }
        if i == 0:
            # Sets which subset the viewer opens on and which data libraries
            # load by default; without it the order is default-then-alphabetical.
            entry["default"] = True
        configs.append(entry)

    # Dump the whole block through a real YAML emitter rather than hand-built
    # lines. Hand-built lines silently corrupt values that look like other
    # types to a YAML parser — a bare `- no` (Norwegian's ISO 639-1 code) is
    # read back as the boolean False, which published a malformed language
    # list. safe_dump quotes it as 'no'. sort_keys=False keeps config order
    # meaningful (first entry is the default).
    body = yaml.safe_dump(
        {
            "pretty_name": pretty_name,
            "license": license_id,
            "language": languages,
            "tags": ["synthetic-data", "ai-alignment", "animal-welfare",
                     "sentient-beings", *[ds["pipeline"] for ds in datasets]],
            "configs": configs,
        },
        default_flow_style=False, allow_unicode=True, sort_keys=False,
    ).rstrip("\n")
    frontmatter = ["---", body, "---", ""]

    lines = [f"# {pretty_name}"]
    for ds in datasets:
        lines += _dataset_section(ds)
    lines += [
        "",
        "## Source",
        "",
        "Generated by [animal-welfare-data-pipeline]"
        "(https://github.com/sentfutures/animal-welfare-data-pipeline).",
    ]

    return "\n".join(frontmatter + lines) + "\n"


def _create_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _upload_folder(folder_path: str, repo_id: str, commit_message: str,
                   delete_patterns: list[str]) -> str:
    from huggingface_hub import HfApi
    return HfApi().upload_folder(
        folder_path=folder_path, repo_id=repo_id, repo_type="dataset",
        commit_message=commit_message,
        # delete_patterns: republishing a run must not leave a PREVIOUS run's
        # audit files (e.g. one with realism_ablation.json alongside a later
        # one without it) lingering next to the new corpus/card. The caller
        # scopes this to the pipeline being published — a bare "audit/*" would
        # delete the SIBLING pipeline's audit files on every publish.
        delete_patterns=delete_patterns,
    )


def _list_repo_files(repo_id: str) -> list[str]:
    from huggingface_hub import HfApi
    return HfApi().list_repo_files(repo_id=repo_id, repo_type="dataset")


def _download_file(repo_id: str, filename: str, local_dir: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().hf_hub_download(
        repo_id=repo_id, filename=filename, repo_type="dataset",
        local_dir=local_dir,
    )


def fetch_sibling(repo_id: str, sibling_tag: str, dest_dir: Path) -> dict | None:
    """Pull the OTHER pipeline's metadata off the Hub so the regenerated card
    keeps its section, or None if that pipeline isn't published yet.

    `dest_dir` must be OUTSIDE the staging tree: hf_hub_download(local_dir=...)
    also writes a .cache/huggingface bookkeeping directory beside the files it
    fetches, and upload_folder has no default ignore for it, so downloading
    into the staged tree would upload that cache as dataset content. Keeping
    the fetch out of the tree also means the sibling's files are never
    re-uploaded at all.

    Downloads only run_manifest.json and audit/*.json — deliberately NOT the
    multi-MB corpus jsonl and not the HTML report. Those stay untouched on the
    Hub because upload_folder only adds/overwrites paths present in the staged
    folder.

    The sibling's record count therefore can't come from counting corpus
    lines; it's read from its own audit reports instead, and left absent
    rather than guessed if none of them carry it.
    """
    try:
        files = _list_repo_files(repo_id)
    except Exception as exc:  # repo may not exist yet on a first publish
        print(f"  (could not list {repo_id}: {type(exc).__name__} — "
              f"treating {sibling_tag} as absent)")
        return None

    prefix = f"{sibling_tag}/"
    if not any(f.startswith(prefix) for f in files):
        return None

    corpus_name = next(
        (f[len(prefix):] for f in files
         if f.startswith(prefix) and f[len(prefix):] in CORPUS_FILENAMES),
        None,
    )
    if corpus_name is None:
        # A partial/odd sibling dir with no corpus can't be declared as a
        # config; better to leave it out than emit a config pointing at nothing.
        print(f"  (no corpus file under {prefix} — treating {sibling_tag} as absent)")
        return None

    wanted = [
        f for f in files
        if f in (f"{prefix}run_manifest.json", f"{prefix}{CARD_META_FILENAME}")
        or (f.startswith(f"{prefix}manifests/") and f.endswith(".json"))
        or (f.startswith(f"{prefix}audit/") and f.endswith(".json"))
    ]
    # A transient download failure must NOT abort the publish: _create_repo has
    # already run and this pipeline's corpus is staged and valid, so raising
    # here would throw away good paid work over the sibling's prose. It must
    # also not drop the sibling entirely — its files stay on the Hub, so
    # removing its config entry would leave them present but unloadable. So
    # keep going with whatever landed: the config entry survives (that's what
    # makes the data loadable) and only the provenance/metrics detail degrades,
    # exactly how this module already treats any other missing audit input.
    failed = []
    for f in wanted:
        try:
            _download_file(repo_id, f, str(dest_dir))
        except Exception as exc:
            failed.append(f"{f} ({type(exc).__name__})")
    if failed:
        print(f"  WARNING: could not fetch {len(failed)} of {len(wanted)} "
              f"{sibling_tag} metadata file(s): {', '.join(failed)}")
        print(f"  The '{sibling_tag}' config entry is preserved, but its card "
              f"section will be missing the detail those files carry.")

    # audit_files reflects everything actually on the Hub (including the HTML
    # and .jsonl we didn't download), so the card's pointers stay accurate.
    audit_files = [f[len(prefix) + len("audit/"):]
                   for f in files if f.startswith(f"{prefix}audit/")]

    dataset_dir = dest_dir / sibling_tag
    n = None
    audit = _load_json(dataset_dir / "audit" / "audit_report.json")
    if audit:
        n = _get(audit, "n_docs") or _get(audit, "n_prompts")
    if n is None:
        div = _load_json(dataset_dir / "audit" / "diversity_report.json")
        n = _get(div, "n_records") if div else None

    return {
        "pipeline": sibling_tag,
        "corpus_file": corpus_name,
        "manifest_file": ("run_manifest.json"
                          if f"{prefix}run_manifest.json" in files else None),
        "audit_files": sorted(audit_files),
        "n_docs": n,
        # The sibling's curated heading/subtitle, so publishing this pipeline
        # doesn't downgrade the sibling's section to the generic name.
        "card_meta": _load_json(dataset_dir / CARD_META_FILENAME),
    }


def _create_tag(repo_id: str, tag: str) -> None:
    from huggingface_hub import HfApi
    # exist_ok: a retried publish with the same --tag (e.g. after fixing a
    # typo'd --input) must not die here after the corpus has already been
    # re-uploaded — that would leave the run in a partially-completed state.
    HfApi().create_tag(repo_id=repo_id, tag=tag, repo_type="dataset", exist_ok=True)


def merge_state(run_commit: str | None, *, fetch: bool = True) -> dict:
    """Seam over utils.merge_state, so tests can pin a run's merge status rather
    than depending on whatever branch the developer happens to be on."""
    return utils.merge_state(run_commit, fetch=fetch)


def _unmerged_summary(stamp: dict) -> str:
    """One-line description of an unmerged stamp, for the Hub commit message."""
    parts = []
    if runs := stamp.get("runs"):
        parts.append("unmerged run(s): "
                     + ", ".join(r.get("run_id") or "unknown" for r in runs))
    if stamp.get("publish_branch"):
        parts.append(f"published from unmerged branch {stamp['publish_branch']}")
    return "; ".join(parts) or "unmerged"


def check_merged(run_dirs: list[Path], *, dry_run: bool,
                 allow_unmerged: bool) -> dict | None:
    """Pre-flight provenance gate. Returns a stamp describing what is NOT backed
    by merged code (for the card), or None when everything checks out:

        {"publish_branch": <branch, only if HEAD isn't verified merged>,
         "runs": [{"run_id", "branch", "commit"}, ...]}

    Every input run is checked separately, and the stamp NAMES each unverified
    one. A combined corpus is only as merged as its least-merged run, and a row
    can be traced to the run — and therefore the code — that produced it, via
    the repo lookup example_gid supports (see flatten_dad_corpus). Naming each
    run is what connects that trace to a merge verdict; collapsing them into
    one would leave a reader able to identify a row's run but not whether that
    run's code was reviewed.

    Deliberately a warning-plus-confirmation rather than a refusal. The HF write
    token lives on contributors' laptops, so a hard block wouldn't prevent an
    unmerged publish — it would push it out of this script, which is the only
    thing that records provenance at all. What makes the check stick is the
    stamp on the public card, not this message.

    A dirty tree at run time is reported as context but is never itself a
    trigger: every real run so far has been dirty, and a warning that fires on
    every run is one people learn to type straight past.
    """
    # One fetch for the whole publish: merge_state would otherwise hit the
    # network once per run dir, and every check compares against the same
    # origin/main anyway.
    checked = []
    for i, run_dir in enumerate(run_dirs):
        manifest = _load_json(run_dir / "run_manifest.json") or {}
        state = merge_state(manifest.get("git_commit"),
                            fetch=(i == 0 and not dry_run))
        checked.append((run_dir, manifest, state))

    # head_merged describes the checkout doing the publishing, so it is the same
    # for every run — read it off the first.
    head_state = checked[0][2]
    unverified = [(rd, m, s) for rd, m, s in checked
                  if s["run_commit_merged"] is not True]
    if head_state["head_merged"] is True and not unverified:
        return None

    reasons = []
    if head_state["head_merged"] is False:
        ahead = head_state["ahead"]
        reasons.append(
            f"the current branch `{head_state['branch']}` has "
            f"{f'{ahead} commit(s)' if ahead else 'commits'} "
            f"not in {utils.MAIN_REF}")
    for run_dir, manifest, state in unverified:
        if state["run_commit_merged"] is not False:
            continue
        dirty = manifest.get("git_dirty_files")
        dirty_note = ""
        if dirty:
            dirty_note = f", plus {len(dirty)} uncommitted file(s) at run time"
        elif manifest.get("git_dirty"):
            dirty_note = ", plus uncommitted changes at run time"
        reasons.append(
            f"run {manifest.get('run_id') or run_dir.name} was generated from "
            f"commit {state['run_commit']}, which is not in "
            f"{utils.MAIN_REF}{dirty_note}")

    # "Not merged" only when something is definitely not merged. When every
    # check came back unknown, say THAT — overstating it teaches people the
    # warning is inaccurate, which is how a guardrail loses its authority.
    subject = "This run" if len(run_dirs) == 1 else "This publish"
    headline = (f"{subject} has NOT been merged into main." if reasons else
                f"{subject}'s provenance could NOT be verified against main.")
    bar = "=" * 68
    print(f"\n{bar}", file=sys.stderr)
    print(f"  {headline}", file=sys.stderr)
    for reason in reasons:
        print(f"    - {reason}", file=sys.stderr)
    # Notes explain why something is UNKNOWN; printed as caveats rather than
    # mixed in with the findings, which would read as reasons for the verdict.
    # Deduplicated: with several runs the same caveat (a stale origin/main, say)
    # would otherwise repeat once per run.
    for note in dict.fromkeys(n for _, _, s in checked for n in s["notes"]):
        print(f"    (note: {note})", file=sys.stderr)
    print("", file=sys.stderr)
    print("  Publishing anyway labels the dataset card as unmerged, publicly.",
          file=sys.stderr)
    print("  If this is meant to be a canonical snapshot, merge your pull "
          "request first", file=sys.stderr)
    print("  and re-run this on main.", file=sys.stderr)
    print(f"{bar}\n", file=sys.stderr)

    # Attribute the branch each run was GENERATED on, not the one it happens to
    # be published from — they differ, and the card's claim is about the code
    # behind the corpus. Only v3+ manifests record it, so fall back to the live
    # checkout for every run predating that.
    stamp: dict = {
        "runs": [
            {"run_id": m.get("run_id") or rd.name,
             "branch": m.get("git_branch") or s["branch"],
             "commit": s["run_commit"]}
            for rd, m, s in unverified
        ],
    }
    # A merged run can still be published from an unmerged checkout, which says
    # nothing about any individual run — so it is recorded separately.
    if head_state["head_merged"] is not True:
        stamp["publish_branch"] = head_state["branch"]
    if dry_run:
        # Nothing is published, so there is nothing to confirm — but the
        # preview still shows the stamp this run would carry.
        return stamp
    if allow_unmerged:
        print("Proceeding: --allow-unmerged was passed.", file=sys.stderr)
        return stamp
    if not sys.stdin.isatty():
        # A prompt nobody can see would hang an agent, a pipe, or a CI job
        # forever. Make the bypass an explicit, greppable flag instead.
        raise SystemExit(
            "Refusing to publish an unmerged run without confirmation. Re-run "
            "interactively, or pass --allow-unmerged to publish anyway.")
    try:
        answer = input("Type 'yes' to publish anyway: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\nAborted.")
    if answer != "yes":
        raise SystemExit("Aborted — nothing was published.")
    return stamp


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a run's final corpus + audit reports as a Hugging Face dataset."
    )
    parser.add_argument("--input", required=True, nargs="+",
                        help="Run directory (SDF or DAD). Several DAD run dirs "
                             "publish as ONE combined corpus, with each run "
                             "named in the card's provenance table; SDF takes "
                             "exactly one.")
    parser.add_argument("--repo-id", required=True,
                        help="e.g. sentientfutures/animal-welfare-training-dataset")
    parser.add_argument("--license", default="cc-by-4.0", dest="license_id")
    parser.add_argument("--tag", default=None,
                        help="Tag to create on the upload commit. Tags are repo-wide, so "
                             "prefix per pipeline (sdf-v1-..., dad-v1-...) once the repo "
                             "holds more than one dataset")
    parser.add_argument("--pretty-name", default=None,
                        help="Repo-level display name for the card (default: the "
                             "--repo-id's last path segment, verbatim)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage + build the card locally; make no Hub API calls")
    parser.add_argument("--staging-dir", default=None,
                        help="Where to stage files (default: a temp dir)")
    parser.add_argument("--allow-unmerged", action="store_true",
                        help="Publish even though this run's code is not in "
                             "origin/main, without the interactive "
                             "confirmation. The dataset card still records it "
                             "as an unmerged publish")
    args = parser.parse_args()

    resolved = [resolve_corpus_file(p) for p in args.input]
    run_dirs = [r[0] for r in resolved]
    corpus_names = {r[1] for r in resolved}
    if len(corpus_names) > 1:
        raise SystemExit("All --input run dirs must belong to the same pipeline "
                         f"(got {sorted(corpus_names)})")
    corpus_name = corpus_names.pop()
    if corpus_name == "sdf_corpus.jsonl" and len(run_dirs) > 1:
        # SDF corpora are copied verbatim rather than rewritten record by
        # record, and SDF has no cross-run stable id (doc_id is per-run), so a
        # concatenation would leave rows no way to be traced back to a run —
        # not even through the repo, the way DAD's example_gid allows.
        raise SystemExit("Combined publishing is DAD-only; pass one SDF run dir.")
    if len(set(run_dirs)) != len(run_dirs):
        raise SystemExit("Duplicate --input run dirs would double their rows "
                         "in the combined corpus.")
    pipeline_tag = "sdf" if corpus_name == "sdf_corpus.jsonl" else "dad"
    sibling_tag = "dad" if pipeline_tag == "sdf" else "sdf"
    pretty_name = args.pretty_name or args.repo_id.rsplit("/", 1)[-1]

    # Before staging (which wipes a directory) and before any Hub call, so an
    # aborted publish leaves nothing behind. Runs in --dry-run too: a preview
    # that hid the warning would be the wrong preview.
    unmerged = check_merged(run_dirs, dry_run=args.dry_run,
                            allow_unmerged=args.allow_unmerged)

    import contextlib
    import tempfile

    if args.staging_dir:
        # Explicitly requested — never ours to delete.
        staging_ctx = contextlib.nullcontext(args.staging_dir)
    elif args.dry_run:
        # --dry-run's whole point is to let a human inspect the staged output
        # afterward, so this directory must outlive the process.
        staging_ctx = contextlib.nullcontext(tempfile.mkdtemp(prefix="publish_hf_"))
    else:
        staging_ctx = tempfile.TemporaryDirectory()

    with staging_ctx as tmp:
        staging_dir = Path(tmp) if args.staging_dir else Path(tmp) / "staged"
        # stage_run wipes the staging root, so it must run BEFORE any sibling
        # metadata is fetched into a neighbouring subdirectory.
        staged = stage_run(run_dirs, corpus_name, staging_dir, pipeline_tag)

        # report_content.json is excluded from the upload (already baked into
        # corpus_report.html) but its title/subtitle are still reused for this
        # dataset's section heading — read in-memory, never staged. Only SDF
        # runs produce one, and SDF publishes are single-run, so runs[0] is
        # the only place it could live.
        content = _load_json(run_dirs[0] / "audit" / "report_content.json")

        run_names = ", ".join(r["run_id"] for r in staged["runs"])
        print(f"Staged {pipeline_tag}/{corpus_name} ({staged['n_docs']} records "
              f"from {len(staged['runs'])} run(s): {run_names}), "
              f"{len(staged['audit_files'])} audit file(s): {', '.join(staged['audit_files']) or '(none)'}")

        dataset_dir = staging_dir / pipeline_tag
        # Persist the two strings this dataset's section heading uses, so the
        # NEXT publish of the other pipeline can restore them instead of
        # falling back to the generic name (report_content.json itself is
        # never uploaded). Only what the card already shows publicly.
        card_meta = {k: v for k, v in (content or {}).items()
                     if k in ("title", "subtitle") and v}
        # The unmerged stamp rides in the same sidecar so it SURVIVES the next
        # publish of the other pipeline: the card is regenerated whole from the
        # sibling's Hub metadata, so a stamp derived from live git at render
        # time would both mislabel the sibling and silently erase its own
        # warning. delete_patterns already clears this file, so a later merged
        # publish of this pipeline drops the stamp on its own.
        if unmerged:
            card_meta["unmerged"] = unmerged
        if card_meta:
            (dataset_dir / CARD_META_FILENAME).write_text(
                json.dumps(card_meta, ensure_ascii=False, indent=2), encoding="utf-8")

        # The section reads the stamp out of `content` — the same key the
        # sibling's arrives under, from its downloaded sidecar.
        section_content = dict(content or {})
        if unmerged:
            section_content["unmerged"] = unmerged

        datasets = [{"pipeline": pipeline_tag, "dir": dataset_dir,
                     "staged": staged, "content": section_content}]

        if args.dry_run:
            # No network: the sibling (if any) is deliberately NOT fetched, so
            # say so rather than letting the preview imply it was dropped.
            card = build_card(datasets, args.license_id, pretty_name)
            (staging_dir / "README.md").write_text(card, encoding="utf-8")
            print(f"\n--dry-run: no Hub API calls made. Staged at {staging_dir} "
                  f"(left on disk for inspection).")
            print(f"NOTE: a '{sibling_tag}' dataset already on the Hub is not fetched in "
                  f"--dry-run, so it is missing from this preview; a real publish "
                  f"regenerates its section from the Hub.")
            print("\n--- README.md ---\n")
            print(card)
            return

        _create_repo(args.repo_id)

        # Regenerate the sibling's card section from the Hub — the card is
        # rewritten whole, so without this a publish would silently drop the
        # other dataset's section and config entry. Fetched OUTSIDE the staging
        # tree so neither its files nor hf_hub_download's .cache bookkeeping dir
        # end up in the upload.
        with tempfile.TemporaryDirectory(prefix="publish_hf_sibling_") as sib_tmp:
            sibling = fetch_sibling(args.repo_id, sibling_tag, Path(sib_tmp))
            if sibling:
                print(f"Preserving existing '{sibling_tag}' dataset "
                      f"({sibling['n_docs']} records) in the card.")
                entry = {"pipeline": sibling_tag, "dir": Path(sib_tmp) / sibling_tag,
                         "staged": sibling, "content": sibling.get("card_meta")}
                # sdf first so it stays the viewer's default config regardless
                # of which pipeline this invocation is publishing.
                datasets = ([entry] + datasets if sibling_tag == "sdf"
                            else datasets + [entry])

            # Card built inside the context: the sibling's downloaded metadata
            # must still exist on disk when build_card reads it.
            card = build_card(datasets, args.license_id, pretty_name)
            (staging_dir / "README.md").write_text(card, encoding="utf-8")

        # run_names (plural) is main's combined-publish naming; the unmerged
        # marker rides on the end of it rather than replacing it.
        commit_message = f"Publish {pipeline_tag}: {run_names}"
        if unmerged:
            # Visible in the repo's commit history, not just this terminal.
            commit_message += f" ({_unmerged_summary(unmerged)})"

        commit = _upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            commit_message=commit_message,
            # Scoped to THIS pipeline — a bare "audit/*" would delete the
            # sibling's audit files on every publish.
            #
            # card_meta.json is listed too even though it lives outside audit/:
            # it is only written when this run HAS a curated title, so without
            # the pattern a later run of the same pipeline that lacks one would
            # leave the earlier run's sidecar on the Hub, and the next sibling
            # publish would restore a title that is no longer what's published.
            # Safe to delete unconditionally because upload_folder drops any
            # deletion whose path is also being added, so a freshly staged
            # sidecar survives while a no-longer-produced one is cleared.
            # run_manifest.json and manifests/* are both listed so a publish
            # that switches layout (single-run <-> combined) clears the OTHER
            # layout's manifest file(s) — upload_folder drops any deletion
            # whose path is also being added, so the layout actually staged
            # always survives its own pattern.
            delete_patterns=[f"{pipeline_tag}/audit/*",
                             f"{pipeline_tag}/run_manifest.json",
                             f"{pipeline_tag}/manifests/*",
                             f"{pipeline_tag}/{CARD_META_FILENAME}"],
        )
        if args.tag:
            _create_tag(args.repo_id, args.tag)
        print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")
        print(f"Commit: {commit}")
        if args.tag:
            print(f"Tag: {args.tag}")


if __name__ == "__main__":
    main()
