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
  REPO=sentientfutures/animal-welfare-mid-training-datasets
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


def stage_run(run_dir: Path, corpus_name: str, staging_dir: Path,
              pipeline_tag: str) -> dict:
    """Copy the publishable subset of a run dir into staging_dir/<pipeline_tag>/.

    The per-pipeline subdirectory is what lets one repo hold both corpora as
    separate HF configs. Returns a manifest dict of what was staged, used both
    for the dataset card and for logging what --dry-run would have uploaded.
    """
    # Refuse a --staging-dir that equals or contains run_dir, OR either of the
    # two specific subtrees this function reads from (final/, audit/): rmtree
    # below would delete data we're about to read before the copy even runs.
    # Checking only run_dir itself isn't enough — a --staging-dir pointing at
    # run_dir/final or run_dir/audit directly (an easy typo, since those are
    # real, well-known subdirectory names on every run) would slip past a
    # run_dir-only check while still destroying the corpus or audit reports.
    staging_real = staging_dir.resolve()
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
    staged: dict = {"pipeline": pipeline_tag, "corpus_file": None,
                    "manifest_file": None, "audit_files": [], "n_docs": 0}

    corpus_src = run_dir / "final" / corpus_name
    corpus_dst = dataset_dir / corpus_name
    shutil.copy2(corpus_src, corpus_dst)
    staged["corpus_file"] = corpus_name
    with open(corpus_dst, encoding="utf-8") as f:
        staged["n_docs"] = sum(1 for _ in f)

    manifest_src = run_dir / "run_manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, dataset_dir / "run_manifest.json")
        staged["manifest_file"] = "run_manifest.json"

    audit_src = run_dir / "audit"
    if audit_src.is_dir():
        audit_dst = dataset_dir / "audit"
        utils.ensure_dir(audit_dst)
        # *.jsonl too: evals/audit_dad.py writes audit/tic_candidates.jsonl and
        # audit/reason_failures.jsonl for DAD runs — a fixed *.json/*.html
        # pattern silently dropped both.
        for pattern in ("*.json", "*.jsonl", "*.html"):
            for f in sorted(audit_src.glob(pattern)):
                if f.name == "report_content.json":
                    continue  # editorial input, already baked into corpus_report.html
                shutil.copy2(f, audit_dst / f.name)
                staged["audit_files"].append(f.name)

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


def build_metrics_rows(dataset_dir: Path) -> list[tuple[str, str, str]]:
    """(label, value, source_filename) rows, one per known audit file that's
    present in this ONE dataset's dir AND has the fields this function expects.

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
    audit_dir = dataset_dir / "audit"
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


PIPELINE_NAMES = {"sdf": "SDF corpus", "dad": "DAD corpus"}
# What one record counts AS, per pipeline — SDF ships documents, DAD ships
# chat examples. Used for the "N <unit>." line under each section heading.
PIPELINE_UNITS = {"sdf": "documents", "dad": "chat examples"}


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


def _dataset_section(ds: dict) -> list[str]:
    """The card's prose block for ONE dataset: heading, count, provenance,
    measured metrics, and pointers to its audit files."""
    tag, dataset_dir, staged = ds["pipeline"], ds["dir"], ds["staged"]
    content = ds.get("content") or {}

    heading = content.get("title") or PIPELINE_NAMES.get(tag, f"{tag.upper()} corpus")
    lines = ["", f"## {heading} (`{tag}` config)"]
    if content.get("subtitle"):
        lines += ["", content["subtitle"]]

    n = staged.get("n_docs")
    if n:
        lines += ["", f"{n} {PIPELINE_UNITS.get(tag, 'records')}."]

    manifest = _load_json(dataset_dir / "run_manifest.json") or {}
    if manifest:
        default_model, overrides = models_used(manifest, tag)
        lines += [
            "",
            f"- **run_id**: `{manifest.get('run_id', 'unknown')}`",
            f"- **label**: `{manifest.get('label', 'unknown')}`",
            f"- **git commit**: `{manifest.get('git_commit', 'unknown')}`",
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
    # yaml.safe_dump, not an f-string: pretty_name and each section heading can
    # come from report_content.json — editorial content this script doesn't
    # control — and a raw quote or newline would corrupt hand-built YAML.
    pretty_name_line = yaml.safe_dump(
        {"pretty_name": pretty_name}, default_flow_style=False, allow_unicode=True
    ).rstrip("\n")

    # language: is repo-wide, so it must be the union across datasets — SDF
    # spans 16 languages while DAD is English-only, and declaring either one
    # alone would misdescribe the repo.
    languages = sorted({
        code
        for ds in datasets
        for code in detected_languages(ds["dir"], ds["pipeline"])
    }) or ["en"]

    frontmatter = [
        "---",
        pretty_name_line,
        f"license: {license_id}",
        "language:",
        *[f"  - {code}" for code in languages],
        "tags:",
        "  - synthetic-data",
        "  - ai-alignment",
        "  - animal-welfare",
        "  - sentient-beings",
        *[f"  - {ds['pipeline']}" for ds in datasets],
        "configs:",
    ]
    for i, ds in enumerate(datasets):
        frontmatter += [
            f"  - config_name: {ds['pipeline']}",
            "    data_files:",
            "      - split: train",
            f"        path: {ds['pipeline']}/{ds['staged']['corpus_file']}",
        ]
        if i == 0:
            # Sets which subset the viewer opens on and which data libraries
            # load by default; without it the order is default-then-alphabetical.
            frontmatter += ["    default: true"]
    frontmatter += ["---", ""]

    lines = [f"# {pretty_name}"]
    for ds in datasets:
        lines += _dataset_section(ds)
    lines += [
        "",
        "## Source",
        "",
        "Generated by [alignment-data-pipeline]"
        "(https://github.com/sentfutures/alignment-data-pipeline).",
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
        if f == f"{prefix}run_manifest.json"
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
    }


def _create_tag(repo_id: str, tag: str) -> None:
    from huggingface_hub import HfApi
    # exist_ok: a retried publish with the same --tag (e.g. after fixing a
    # typo'd --input) must not die here after the corpus has already been
    # re-uploaded — that would leave the run in a partially-completed state.
    HfApi().create_tag(repo_id=repo_id, tag=tag, repo_type="dataset", exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a run's final corpus + audit reports as a Hugging Face dataset."
    )
    parser.add_argument("--input", required=True, help="Run directory (SDF or DAD)")
    parser.add_argument("--repo-id", required=True,
                        help="e.g. sentientfutures/animal-welfare-mid-training-datasets")
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
    args = parser.parse_args()

    run_dir, corpus_name = resolve_corpus_file(args.input)
    pipeline_tag = "sdf" if corpus_name == "sdf_corpus.jsonl" else "dad"
    sibling_tag = "dad" if pipeline_tag == "sdf" else "sdf"
    pretty_name = args.pretty_name or args.repo_id.rsplit("/", 1)[-1]

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
        staged = stage_run(run_dir, corpus_name, staging_dir, pipeline_tag)

        # report_content.json is excluded from the upload (already baked into
        # corpus_report.html) but its title/subtitle are still reused for this
        # dataset's section heading — read in-memory, never staged.
        content = _load_json(run_dir / "audit" / "report_content.json")

        print(f"Staged {pipeline_tag}/{corpus_name} ({staged['n_docs']} records), "
              f"{'with' if staged['manifest_file'] else 'without'} run_manifest.json, "
              f"{len(staged['audit_files'])} audit file(s): {', '.join(staged['audit_files']) or '(none)'}")

        datasets = [{"pipeline": pipeline_tag, "dir": staging_dir / pipeline_tag,
                     "staged": staged, "content": content}]

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
                         "staged": sibling, "content": None}
                # sdf first so it stays the viewer's default config regardless
                # of which pipeline this invocation is publishing.
                datasets = ([entry] + datasets if sibling_tag == "sdf"
                            else datasets + [entry])

            # Card built inside the context: the sibling's downloaded metadata
            # must still exist on disk when build_card reads it.
            card = build_card(datasets, args.license_id, pretty_name)
            (staging_dir / "README.md").write_text(card, encoding="utf-8")

        commit = _upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            commit_message=f"Publish {pipeline_tag}: {run_dir.name}",
            # Scoped to THIS pipeline — a bare "audit/*" would delete the
            # sibling's audit files on every publish.
            delete_patterns=[f"{pipeline_tag}/audit/*"],
        )
        if args.tag:
            _create_tag(args.repo_id, args.tag)
        print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")
        print(f"Commit: {commit}")
        if args.tag:
            print(f"Tag: {args.tag}")


if __name__ == "__main__":
    main()
