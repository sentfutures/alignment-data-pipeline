# `report/` — the standalone DAD story report

One self-contained HTML page explaining the DAD pipeline to an external technical
reader (someone who runs evals at a lab): the problem, a worked example, the
method, how it is measured, the results, what the data would teach a model, where
it is weak, and how to reproduce it.

This is **not** the Streamlit corpus-audit page. That page is an internal review
tool organised by what the eval measured; this is organised by what a reader needs
to believe, in order, and it is a single file you can hand over or host anywhere.

## Build

```bash
python report/build_report.py --run outputs/dad/runs/<run_id>
# -> report/dad_report.html   (defaults: --content report/content.md)
```

The current build comes from `outputs/dad/runs/2026-07-20_20-51_bedrock-40`.

To get the full report, the run needs its paid audit pass — without it the
delivery, Pareto and showcase blocks say "not measured on this run" and the
weaknesses table gains a BAD row:

```bash
python evals/audit_dad.py --input outputs/dad/runs/<run_id> --reasons
python evals/diversity.py --input outputs/dad/runs/<run_id>
```

To run the evals on the shared AWS Bedrock credits instead of an Anthropic API
key, add `--config config.bedrock.yaml` (identical to `config.yaml` but
`backend: bedrock`, which reads `CHAD_AWS_BEDROCK_KEY`).

## Files

| File | Role |
|---|---|
| `content.md` | **All the prose.** The file to iterate on. |
| `build_report.py` | Loading, the pure `build_report()` seam, the section builders. |
| `render.py` | CSS + inline-SVG chart primitives. No pipeline knowledge. |
| `dad_report.html` | The built artefact. |

## The two rules

**1. No number is ever typed into `content.md`.** Prose interpolates
`{{placeholders}}` resolved from the run's own audit JSON; an unknown one fails the
build. Only facts every run has are available to prose (`n`, `gen_models`,
`judge_model`, `backend`, `cost_total`, `cost_per_example`) — run-conditional
figures belong to the charts, so a run missing the paid pass degrades instead of
shipping a stale sentence.

**2. The weaknesses section is derived, not written.** Every BAD/OK verdict in the
audit, plus fixed provenance rules (non-`api` backend, dirty git tree, length
inflation, unmeasured sections, arm asymmetry, a delivery-quality regression),
emits its own row whether or not anyone wrote it up. `content.md` adds to that
floor; it cannot replace it. `test_weaknesses_render_without_any_editorial_prose`
pins this.

Section ids in `content.md` must exactly match `build_report.CONTENT_IDS` — a
missing or unknown id is a build error, so a typo can never silently drop a
section. `example_pick` holds the prompt_id of the worked example (or `auto`), so
a rebuild reproduces the same case without a command-line flag.

## Constraints

- **Self-contained**: no external CSS, JS, fonts or images. It must open offline
  from the filesystem, and artifact hosts' CSP blocks external origins. Charts are
  inline `<svg>`; the only JS is a tooltip handler. Enforced by
  `test_is_self_contained`.
- **stdlib only**, and no imports from `viewer/` or `shared/` — the report has to
  build where the pipeline's dependencies are not installed, which is also what
  makes it portable. Cost: the row-building helpers in `viewer/rendering.py` are
  re-implemented here, so a schema change to `audit_report.json` can drift.
- Both schemas render: modern (`valuable_welfare_considerations`) and legacy
  (reconstructed from `moral_patient_reasons` + `moves.alternatives`, exactly as
  `evals/audit_dad.py` does).

The CSS and the `esc`/`hbar`/`histogram`/`stat`/`table` primitives are adapted
from `evals/report_sdf.py` on the unmerged branch
`origin/aidan/sdf-500-run-and-report`, which is deliberately left untouched so it
still merges. Rewiring it onto `render.py` is a follow-up once it lands — that is
also the route to an SDF companion report.

## Tests

`pytest tests/test_dad_report.py` (48 tests, offline).
