# `report/` — the standalone report pages

A small hand-over site explaining these corpora to an external technical reader
(someone who runs evals or midtraining at a lab). Each page is one self-contained HTML
file you can host anywhere, email, or open offline from the filesystem.

| Page | File | What it is |
|---|---|---|
| Hub | `index.html` | What the two corpora are and why there are two. Everything true of both, met once: why this data is missing, the *Teaching Claude Why* grounding, the two routes and the one we turned down, the shared measurement philosophy, the shared limits, how to read a provenance line. |
| DAD | `dad.html` | The dilemma corpus in detail: the gap, one worked example in full, what the numbers say, how it is built, its stylistic footprint, how it is measured, where it is weak, how to run it, an appendix. |
| SDF | *not built yet* | The document corpus. The hub already introduces it and says the report is in preparation. `report/sdf.py` + `content_sdf.md` are the shape it would take; see the notes at the bottom of this file. |

These are **not** the Streamlit corpus-audit page. That is an internal review tool
organised by what the eval measured; these are organised by what a reader needs to
believe, in order.

## Build

```bash
python report/build_report.py --dad-run outputs/dad/runs/<run_id>
# -> report/index.html, report/dad.html

python report/build_report.py --page dad --dad-run outputs/dad/runs/<run_id>
python report/build_report.py --page index --dad-run <dir> --sdf-run <dir>
```

`--run` still works as an alias for `--dad-run`, which keeps the command printed in the
DAD page's own "Run it yourself" section true. `--content` (repeatable) overrides a
page's prose files. The current build comes from
`outputs/dad/runs/2026-07-20_20-51_bedrock-40`.

To get the full DAD page, the run needs its paid audit pass. Without it the delivery,
Pareto and showcase blocks say "not measured on this run", the lede's delivery clause
degrades to the same, and the weaknesses table gains a BAD row:

```bash
python evals/audit_dad.py --input outputs/dad/runs/<run_id> --reasons
python evals/diversity.py --input outputs/dad/runs/<run_id>
```

To run the evals on the shared AWS Bedrock credits instead of an Anthropic API key, add
`--config config.bedrock.yaml` (identical to `config.yaml` but `backend: bedrock`, which
reads `CHAD_AWS_BEDROCK_KEY`).

## Files

| File | Role |
|---|---|
| `content_shared.md` | **Hub prose.** |
| `content_dad.md` | **DAD prose.** The file to iterate on for that page. |
| `common.py` | Loading, prose parsing, `fill()`, cost aggregation, the provenance warnings, the warnings table, the CLI parser. Everything both pages use. |
| `dad.py` | The DAD page: `CONTENT_IDS`, `TOC`, `facts()`, the section builders, `derived_warnings()`. |
| `hub.py` | The hub page. |
| `render.py` | CSS + inline-SVG chart primitives + the `document()` shell. No pipeline knowledge. |
| `build_report.py` | The CLI. |

Each pipeline module exposes `body()` (returning sections, rail entries and header
fields) and `build()`. `document()` is called once, by `build()`. That is what would make
a single combined page a short function rather than a refactor, and it costs nothing
today.

## The rules

**1. No number is ever typed into a prose file.** Prose interpolates `{{placeholders}}`
resolved from the run's own audit JSON, and an unknown one fails the build.
Run-conditional figures are available to prose only as **pre-composed clauses** carrying
an explicit degraded string — `{{substance_clause}}`, `{{delivery_clause}}`,
`{{library_clause}}`, `{{footprint_regressions}}`. A run missing the paid pass renders
"no measured delivery comparison on this run" where the finding would be, so the sentence
survives and its claim does not. Do not add a bare conditional number to prose; add a
clause to `facts()`. The hub interpolates nothing at all — it has no run of its own, so a
placeholder there is a build error.

**2. The weaknesses section is derived, not written.** Every BAD/OK verdict in the audit,
plus provenance rules (non-`api` backend, uncommitted changes, small n) and DAD-specific
rules (a delivery regression, per-measure arm asymmetry, length inflation, an unmeasured
delivery pass), emits its own row whether or not anyone wrote it up. `content_dad.md`
adds to that floor; it cannot replace it. `warnings_table()` may **collapse** rows into a
drawer, and the drawer states how many it holds — collapsing is a view, never a filter.
`test_weaknesses_render_without_any_editorial_prose` pins this.

Section ids in each prose file must exactly match that module's `CONTENT_IDS` — a missing
or unknown id is a build error, so a typo can never silently drop a section. Two prose
files may not both define an id. `example_pick` holds the prompt_id of the DAD worked
example (or `auto`), so a rebuild reproduces the same case without a command-line flag.

## Constraints

- **Self-contained**: no external CSS, JS, fonts or images. Each page must open offline
  from the filesystem, and artifact hosts' CSP blocks external origins. Charts are inline
  `<svg>`; the only JS is a tooltip handler and a rail scroll-spy. Enforced by
  `test_is_self_contained`.
- **One theme, light.** `render.py` declares `color-scheme:only light` and emits a
  matching `<meta>`. `only` is load-bearing: it opts the page out of Chrome-Android and
  Samsung Internet's auto-darkening, which `prefers-color-scheme` does not cover. These
  are printed and screenshotted artefacts, and a viewer's OS preference is not a signal
  about how a published document should look. `test_is_light_mode_only` pins it.
- **Reserved status colours.** `--good`/`--warn`/`--bad` are not series hues. They used to
  be — `--good` was byte-identical to `--series-3`, the pipeline's own colour, so the
  palette quietly editorialised "pipeline = good". `test_status_colors_are_not_series_colors`
  pins the separation. Direction is also carried by a labelled chip rather than by
  colouring a numeral, so a status colour never travels alone.
- **Arm colours follow the arm.** `hbar(color=...)` takes a sequence; pass `R.ARM_PAIR`
  for any (control, pipeline) chart. Without it `hbar` falls back to `PAL[i]`, which
  colours bars by row order — that is how the headline chart came to paint the pipeline in
  the control's own colour while every other chart used green.
- **stdlib only**, and no imports from `viewer/` or `shared/` — the pages have to build
  where the pipeline's dependencies are not installed, which is also what makes them
  portable. Cost: the row-building helpers in `viewer/rendering.py` are re-implemented
  here, so a schema change to `audit_report.json` can drift.
- Both audit schemas render: modern (`valuable_welfare_considerations`) and legacy
  (reconstructed from `moral_patient_reasons` + `moves.alternatives`, exactly as
  `evals/audit_dad.py` does).

## Tests

```bash
pytest tests/test_report_common.py tests/test_dad_report.py tests/test_report_hub.py
```

102 tests, offline. `test_report_common.py` covers the shared plumbing (prose ids, the
placeholder contract, the provenance floor, the warnings table); `test_dad_report.py`
covers the DAD page along four risk axes — degradation, self-containment, candour, colour
integrity; `test_report_hub.py` covers the hub, whose distinctive risk is a link to a page
nobody has built.

## Adding the SDF page

`report/render.py` and `report/common.py` are already pipeline-agnostic, so an SDF page is
a new `report/sdf.py` + `report/content_sdf.md`, plus `sdf_href="sdf_report.html"` passed
to the hub. Two things to know before starting:

- **`derived_warnings()` cannot be shared.** `evals/audit_dad.py` records its verdicts
  into `sections[].rows[]`; `evals/audit_sdf.py` only prints them. So
  `common.audit_verdict_warnings()` returns `[]` for an SDF audit, and that page has to
  compute its own thresholds (top-type share, truncation fraction, near-dup rate,
  formulaic openings, score-distribution degeneracy). Teaching `audit_sdf.py` to record
  rows the way `audit_dad.py` does would give future runs the shared floor for free;
  committed runs predate it either way.
- **`evals/report_sdf.py` on `origin/aidan/sdf-500-run-and-report` is not portable.**
  Roughly half of its 853 lines is editorial prose welded to a 477-document run ("across
  477 documents", "nineteen drifted — 95%"), which is exactly what rule 1 exists to
  prevent. Lift its `histogram()` and `excerpt_block()`; write the rest. `render.py`
  already has `histogram()`.
- The only committed SDF run with a full `audit/audit_report.json` is
  `outputs/sdf/runs/2026-07-11_20-06_matrix100-cli` (100 docs). The newer
  `2026-07-13_13-18_al-gap-fixes-100docs` has only `diversity_report.json` and would need
  `python evals/audit_sdf.py --input <run> --patterns` re-run.
