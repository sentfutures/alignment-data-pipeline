#!/usr/bin/env python3
"""Publish a run's final corpus + audit reports as a Hugging Face dataset.

Stages a run directory into a flat dataset layout — the final corpus jsonl
at the repo root, run_manifest.json for provenance, and (if present) every
audit/*.json + audit/*.html file, globbed rather than named so a future run's
eval additions/omissions are picked up or skipped automatically — then writes
a dataset card (README.md) and uploads the lot in one commit.

The card is built entirely from measured fields already sitting in the audit
JSONs — no interpretive prose is authored here. Each of the six known audit
files contributes one metrics-table row (cited by source filename); any file
that's absent just omits its row, the same "sections whose inputs are absent
are omitted" convention evals/report_sdf.py already uses. report_content.json
(the one editorial input among the audit files — curated excerpts/translations
and report-section prose) is excluded from the upload: it's already fully
baked into corpus_report.html, so nothing in it would be invisible to a Hub
visitor. Its title/subtitle strings, already curated by the pipeline for this
run, are reused verbatim as the card header when present.

Usage:
  python evals/publish_hf.py --input outputs/sdf/latest --repo-id sentientfutures/sdf-corpus
  python evals/publish_hf.py --input outputs/sdf/runs/<run_id> --repo-id sentientfutures/sdf-corpus \
      --tag v1-fullscale-500-opus5
  python evals/publish_hf.py --input outputs/sdf/latest --repo-id sentientfutures/sdf-corpus --dry-run

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

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared import utils

load_dotenv()

CORPUS_FILENAMES = ("sdf_corpus.jsonl", "dad_corpus.jsonl")


def resolve_run_dir(input_arg: str) -> tuple[Path, str]:
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


def stage_run(run_dir: Path, corpus_name: str, staging_dir: Path) -> dict:
    """Copy the publishable subset of a run dir into staging_dir (flattened).

    Returns a manifest dict of what was staged, used both for the dataset
    card and for logging what --dry-run would have uploaded.
    """
    utils.ensure_dir(staging_dir)
    staged: dict = {"corpus_file": None, "manifest_file": None, "audit_files": [], "n_docs": 0}

    corpus_src = run_dir / "final" / corpus_name
    corpus_dst = staging_dir / corpus_name
    shutil.copy2(corpus_src, corpus_dst)
    staged["corpus_file"] = corpus_name
    with open(corpus_dst, encoding="utf-8") as f:
        staged["n_docs"] = sum(1 for _ in f)

    manifest_src = run_dir / "run_manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, staging_dir / "run_manifest.json")
        staged["manifest_file"] = "run_manifest.json"

    audit_src = run_dir / "audit"
    if audit_src.is_dir():
        audit_dst = staging_dir / "audit"
        utils.ensure_dir(audit_dst)
        for pattern in ("*.json", "*.html"):
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


def build_metrics_rows(staging_dir: Path) -> list[tuple[str, str, str]]:
    """(label, value, source_filename) rows, one per known audit file that's
    present AND has the fields this function expects.

    Every value is a measured field lifted verbatim from the file's own
    summary — no thresholds, verdicts, or causal claims added here.
    """
    audit_dir = staging_dir / "audit"
    rows: list[tuple[str, str, str]] = []

    d = _load_json(audit_dir / "audit_report.json")
    if d is not None and _get(d, "n_docs") is not None:
        rows.append(("Documents (offline audit)", str(_get(d, "n_docs")), "audit_report.json"))

    d = _load_json(audit_dir / "compliance_report.json")
    if d is not None:
        judged, clean, frac = _get(d, "judged"), _get(d, "clean_documents"), _get(d, "clean_frac")
        if judged is not None:
            rows.append((
                "Constitutional compliance",
                f"{clean} of {judged} judged clean ({frac:.1%})" if frac is not None
                else f"{clean} of {judged} judged clean",
                "compliance_report.json",
            ))

    d = _load_json(audit_dir / "card_fidelity_report.json")
    if d is not None:
        judged = _get(d, "judged")
        by_card = _get(d, "by_card_frac", default={})
        if judged is not None and by_card:
            per_axis = ", ".join(f"{axis} {frac:.1%}" for axis, frac in by_card.items())
            rows.append((
                "Card fidelity (plan honours dealt card)",
                f"{per_axis} — n={judged}",
                "card_fidelity_report.json",
            ))

    d = _load_json(audit_dir / "diversity_report.json")
    if d is not None:
        n = _get(d, "n_records")
        vendi_score, vendi_ratio = _get(d, "vendi", "score"), _get(d, "vendi", "ratio")
        mpc = _get(d, "mean_pairwise_cosine")
        if vendi_score is not None:
            detail = f"Vendi {vendi_score:.1f} effective docs of {n} (ratio {vendi_ratio:.3f})"
            if mpc is not None:
                detail += f", mean pairwise cosine {mpc:.3f}"
            rows.append(("Semantic diversity", detail, "diversity_report.json"))

    d = _load_json(audit_dir / "realism_ablation.json")
    if d is not None:
        n, l5, blind, drop = (_get(d, "n"), _get(d, "layer5_mean"),
                               _get(d, "blind_same_rubric_mean"), _get(d, "mean_drop"))
        if n is not None:
            rows.append((
                "Realism, spec-blind rescore (n={})".format(n),
                f"{l5:.2f} (in-spec) vs {blind:.2f} (spec hidden), drop {drop:.2f}",
                "realism_ablation.json",
            ))

    d = _load_json(audit_dir / "vendi_curve.json")
    if d is not None:
        proj_1000 = _get(d, "proj", "1000")
        proj_5000 = _get(d, "proj", "5000")
        if proj_1000 or proj_5000:
            def _fmt_proj(p):
                return f"power {p['power']:.1f}, log {p['log']:.1f}"
            parts = []
            if proj_1000:
                parts.append(f"n=1000: {_fmt_proj(proj_1000)}")
            if proj_5000:
                parts.append(f"n=5000: {_fmt_proj(proj_5000)}")
            rows.append(("Projected effective-document count", "; ".join(parts), "vendi_curve.json"))

    return rows


def build_card(
    staging_dir: Path,
    staged: dict,
    license_id: str,
    pipeline_tag: str,
    content: dict | None = None,
) -> str:
    manifest = _load_json(staging_dir / "run_manifest.json") or {}
    n_docs = staged["n_docs"]

    title = (content or {}).get("title") or f"{pipeline_tag.upper()} corpus"
    subtitle = (content or {}).get("subtitle")

    frontmatter = [
        "---",
        f'pretty_name: "{title}"',
        f"license: {license_id}",
        "language:",
        "  - en",
        "tags:",
        "  - synthetic-data",
        "  - ai-alignment",
        "  - animal-welfare",
        "  - sentient-beings",
        f"  - {pipeline_tag}",
        "configs:",
        "  - config_name: default",
        "    data_files:",
        "      - split: train",
        f"        path: {staged['corpus_file']}",
        "---",
        "",
    ]

    lines = [f"# {title}"]
    if subtitle:
        lines += ["", subtitle]
    lines += ["", f"{n_docs} documents."]

    if manifest:
        lines += [
            "",
            "## Provenance",
            "",
            f"- **run_id**: `{manifest.get('run_id', 'unknown')}`",
            f"- **label**: `{manifest.get('label', 'unknown')}`",
            f"- **git commit**: `{manifest.get('git_commit', 'unknown')}`",
            f"- **model**: `{manifest.get('model', 'unknown')}`",
            f"- **backend**: `{_get(manifest, 'config', 'backend', default='unknown')}`",
            "- **source**: [alignment-data-pipeline]"
            "(https://github.com/Mycelium-tools/alignment-data-pipeline)",
        ]

    metrics_rows = build_metrics_rows(staging_dir)
    if metrics_rows:
        lines += ["", "## Measured metrics", "", "| Metric | Value |", "| --- | --- |"]
        lines += [f"| {label} | {value} |" for label, value, _source in metrics_rows]

    have_html = "corpus_report.html" in staged["audit_files"]
    summarized = {source for _, _, source in metrics_rows}
    extra_files = [f for f in staged["audit_files"]
                  if f != "corpus_report.html" and f not in summarized]
    if have_html:
        lines += ["", "See `audit/corpus_report.html` for the full interactive report "
                       "(self-contained; open directly in a browser)."]
    if extra_files:
        lines += ["", "Additional machine-readable audit files included under `audit/`: "
                       + ", ".join(f"`{f}`" for f in sorted(extra_files)) + "."]

    return "\n".join(frontmatter + lines) + "\n"


def _create_repo(repo_id: str) -> None:
    from huggingface_hub import HfApi
    HfApi().create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)


def _upload_folder(folder_path: str, repo_id: str, commit_message: str) -> str:
    from huggingface_hub import HfApi
    return HfApi().upload_folder(
        folder_path=folder_path, repo_id=repo_id, repo_type="dataset",
        commit_message=commit_message,
    )


def _create_tag(repo_id: str, tag: str) -> None:
    from huggingface_hub import HfApi
    HfApi().create_tag(repo_id=repo_id, tag=tag, repo_type="dataset")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Publish a run's final corpus + audit reports as a Hugging Face dataset."
    )
    parser.add_argument("--input", required=True, help="Run directory (SDF or DAD)")
    parser.add_argument("--repo-id", required=True, help="e.g. sentientfutures/sdf-corpus")
    parser.add_argument("--license", default="cc-by-4.0", dest="license_id")
    parser.add_argument("--tag", default=None, help="Git tag to create on the upload commit")
    parser.add_argument("--dry-run", action="store_true",
                        help="Stage + build the card locally; make no Hub API calls")
    parser.add_argument("--staging-dir", default=None,
                        help="Where to stage files (default: a temp dir)")
    args = parser.parse_args()

    run_dir, corpus_name = resolve_run_dir(args.input)
    pipeline_tag = "sdf" if corpus_name == "sdf_corpus.jsonl" else "dad"

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        staging_dir = Path(args.staging_dir) if args.staging_dir else Path(tmp) / "staged"
        staged = stage_run(run_dir, corpus_name, staging_dir)

        # report_content.json is excluded from the upload (already baked into
        # corpus_report.html) but its title/subtitle are still reused for the
        # card header — read in-memory, never written into the staged tree.
        content = _load_json(run_dir / "audit" / "report_content.json")

        card = build_card(staging_dir, staged, args.license_id, pipeline_tag, content=content)
        (staging_dir / "README.md").write_text(card, encoding="utf-8")

        print(f"Staged {corpus_name} ({staged['n_docs']} docs), "
              f"{'with' if staged['manifest_file'] else 'without'} run_manifest.json, "
              f"{len(staged['audit_files'])} audit file(s): {', '.join(staged['audit_files']) or '(none)'}")

        if args.dry_run:
            print(f"\n--dry-run: no Hub API calls made. Staged at {staging_dir}")
            print("\n--- README.md ---\n")
            print(card)
            return

        _create_repo(args.repo_id)
        commit = _upload_folder(
            folder_path=str(staging_dir),
            repo_id=args.repo_id,
            commit_message=f"Publish {run_dir.name}",
        )
        if args.tag:
            _create_tag(args.repo_id, args.tag)
        print(f"\nPublished to https://huggingface.co/datasets/{args.repo_id}")
        print(f"Commit: {commit}")
        if args.tag:
            print(f"Tag: {args.tag}")


if __name__ == "__main__":
    main()
