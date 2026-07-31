# `report/` — the handoff page

One self-contained HTML file covering both datasets, written for one reader: someone who
runs midtraining at another lab, has no context on this project, and has about forty
seconds to decide whether to keep reading. It publishes to GitHub Pages as it stands,
emails, and opens offline from the filesystem.

The two datasets are **Difficult advice** (`dad`) and **Synthetic documents** (`sdf`) —
"corpus" and "corpora" are not words this page uses.

This is **not** the Streamlit corpus-audit page. That is an internal review tool
organised by what the eval measured; this is organised by what a reader needs, in order.

## Build

```bash
python report/build_report.py \
  --dad-run outputs/dad/runs/2026-07-20_20-51_bedrock-40 \
  --sdf-run outputs/sdf/runs/2026-07-11_20-06_matrix100-cli
# -> report/index.html
```

Those two runs are the pinned ones behind the current build. `--run` still works as an
alias for `--dad-run`, which keeps the command printed in the page's own "Running it
yourself" block true. `--content` (repeatable) overrides the prose files, `--example`
overrides the worked example, `--out-dir` writes elsewhere.

`--sdf-run` is optional. Without it the synthetic documents' column says "not published
yet" and its report says no audit output was supplied — the page still builds, and
carries no dead links.

To get the full difficult-advice report, the DAD run needs its paid audit pass. Without it the
delivery, Pareto and showcase blocks say "not measured on this run" and the weaknesses
table gains a BAD row:

```bash
python evals/audit_dad.py --input outputs/dad/runs/<run_id> --reasons
python evals/diversity.py --input outputs/dad/runs/<run_id>
```

To run the evals on the shared AWS Bedrock credits instead of an Anthropic API key, add
`--config config.bedrock.yaml` (identical to `config.yaml` but `backend: bedrock`, which
reads `CHAD_AWS_BEDROCK_KEY`).

## The page

| Anchor | What it is |
|---|---|
| hero | The illustration, the title, and the three lines that follow from it (*Teaching Claude Why*, and the two datasets built on it) — centred, and carrying the `#intro` id. Nothing else: no lede, no provenance, no tiles, and no "Intro" heading over a paragraph that needs no introducing. |
| `#datasets` | The comparison. No heading over it: the two column mastheads (name in serif, one line on what each dataset *is*) are the heading. Four rows — what it is for, what a record is, how many prompt templates, how many records — because the reader is deciding whether to run the pipeline, not shopping for a dataset. Dates, model ids and the composition spread live in the report that goes into them. The two figure rows carry the way to what they count: the figure at the column's left edge, an outline button at its right (the prompts on GitHub, the published sample on Hugging Face). Labels are right-aligned, one line each, vertically centred. |
| `#explore` | "Walk through a dataset generation" — a walkthrough, not results, because roughly half of each report is the worked example and the pipeline that produced it. Two buttons carrying each dataset's name and nothing else, 40rem centred so each sits under its own column. |
| `#sdf` | Synthetic documents (`report/sdf.py`) — a placeholder while its full report is written. Hidden until chosen. |
| `#dad` | Difficult advice, in full (`report/dad.py`). Hidden until chosen. |
| footer | Repo and both viewers as buttons, one provenance line per run, and the build claim. |

Both reports take the same skeleton, so a reader learns it once: **what it is + headline
figures / one example end to end / what we measured / how it is built / where it is weak
/ appendix**. Each beat is an `<h3>` with its own id (`#dad-weak`, `#sdf-what`).

There is no contents rail: the whole navigation of the page is one choice.

## The chooser

Neither report is open on load — the choice is the point. Three things make that safe,
and all three are pinned by `TestChooser`:

- **`#dad` and `#sdf` in the URL open that report**, on load and on `hashchange`, so the
  dataset card's deep links land where they say they will. A hash naming anything *inside*
  a report (`#dad-weak`, from a quoted finding) opens the report it lives in and scrolls
  to it — that is what `closest('.panel')` in the inline JS is for.
- **Each report ends with a button offering the other**, which switches panels and
  scrolls to the top of the new one. The dataset a reader did not pick is one click from
  the end of the one they did.
- **Printing expands both**, so a PDF of the page is the whole thing.

The cost is real and was accepted deliberately: Cmd-F cannot see a closed report.
`.panel[hidden]{display:none}` is load-bearing — a panel is a `<section>`, and
`section{display:grid}` beats the browser's own `[hidden]` rule.

## Files

| File | Role |
|---|---|
| `content_page.md` | **Page prose**: title, intro, the comparison's cells, the synthetic documents' placeholder text. `*_desc` are the mastheads' subtitles (what each dataset *is*, also used under each chooser button); `*_use` are what each is *for*. The page's own prose interpolates nothing — a `{{placeholder}}` in it is a build error. |
| `content_dad.md` | **Difficult-advice prose.** The file to iterate on for that report. |
| `page.py` | The page: hero, comparison, chooser, footer, and the one `document()` call. |
| `dad.py` | The `#dad` beats: `facts()`, the block builders, `derived_warnings()`. |
| `sdf.py` | The `#sdf` beats — small on purpose; see "Finishing the second report". |
| `common.py` | Loading, prose parsing, `fill()`, cost aggregation, the provenance warnings, the warnings table, `editorial_words()`, the CLI parser. |
| `render.py` | CSS + inline-SVG chart primitives + the `document()` shell. No pipeline knowledge. |
| `build_report.py` | The CLI. |

Each report module exposes `blocks()`, returning its body as one flat string; `page.py`
wraps that in `render.panel()`, which is the `<section>`. Blocks stay flat because a
figure has to be a direct child of the section for the CSS grid to bleed it past the
text measure.

## The rules

**1. No number is ever typed into a prose file.** Prose interpolates `{{placeholders}}`
resolved from the runs' own output, and an unknown one fails the build. Run-conditional
figures reach prose only with an explicit degraded string — `{{library_clause}}`,
`{{near_dup_pct}}`, `{{length_pct}}` — so a run missing the paid pass renders "an
unmeasured share" where the figure would be and the sentence survives. The page's own
prose has exactly two facts available, `{{gen_models}}` and `{{judge_models}}`, both of
The page's own prose has no facts at all (`PAGE_FACTS = {}`), so a placeholder in
content_page.md is a build error. Do not add a bare conditional number to prose; add a
clause to the owning module's `facts()`.

**2. The weaknesses beats are derived, not written.** Every BAD/OK verdict the DAD audit
recorded, plus provenance rules (non-`api` backend, uncommitted changes, small n) and
DAD-specific rules (a delivery regression, per-measure arm asymmetry, length inflation,
an unmeasured delivery pass), emits its own row whether or not anyone wrote it up.
`evals/audit_sdf.py` only *prints* its verdicts, so `sdf.derived_warnings()` re-applies
the eval's own thresholds instead. `warnings_table()` may **collapse** rows into a
drawer, and the drawer states how many it holds — collapsing is a view, never a filter.

**3. The delivery regression is stated once.** In prose, in the results, by
`dad._delivery_statement()`. The hero tile, the scoreboard row and the derived weakness
carry the same number as data; a prose file that says it again is the hedging this page
was rebuilt to remove. `TestSayingItOnce` pins it.

**4. Synthetic documents comes first**, in the comparison, the chooser and the panel
order, so the page reads in one order throughout.

**5. Both datasets are for midtraining.** "SFT" names the *format* of the difficult-advice
data — chat transcripts, consumed as supervised fine-tuning — not a different training
phase; the documents are consumed as continued pretraining. The "what it is for" row says
both halves, because internal shorthand has the two sounding like different phases.

**6. Prose has a budget.** The build prints `editorial_words()` for the page it just
wrote, and `test_the_prose_has_a_ceiling` fails if the shipped prose files grow past it.
Deks — the aphoristic line under a heading — are rationed to two for the whole page.

Section ids in each prose file must exactly match the owning module's `CONTENT_IDS`; a
missing or unknown id is a build error, and two files may not both define one, so moving
a block between prose files is a rename. `example_pick` holds the prompt_id of the DAD
worked example (or `auto`), so a rebuild reproduces the same case without a flag.

## The hero illustration

`report/assets/hero.png` is inlined as a `data:` URI at build time (`build_report.
data_uri()`), because the page must open offline and survive an artifact host's CSP: a
file reference, even a relative one, breaks the "one file" guarantee and
`test_is_self_contained` with it. `render.illustration()` raises on anything that is not
a `data:` URI, and renders a marked-TODO placeholder at the right proportions if the
asset is missing.

`report/assets/hero.png` is the artwork as supplied, unedited — an RGBA PNG, so the line
art sits straight on the cream with no background of its own. It is 2.1 MB, which
makes the built page ~3 MB — fine for a page you open or publish, worth knowing
before you email it.

## Constraints

- **Self-contained**: no external CSS, JS, fonts or images. Charts and the two link
  marks are inline `<svg>`, the hero is a data URI, and the only JS is a tooltip handler
  and the chooser. Enforced by `test_is_self_contained`, which allows a `data:` src and
  nothing else off-page.
- **One accent, `--accent:#3b2fa0`.** The page's only interaction colour: the text
  selection and every link. Indigo because it cannot collide with anything the palette
  reserves — far from `--good`, `--warn` and `--bad`, so a selection can never read as a
  verdict, and deeper than `--series-7`, which only appears inside charts. There is no
  separate `--link` token; two names for one hex is how a palette drifts.
- **A link is a typographic object**: `var(--mono)` at `.92em`, weight 600, accent
  coloured, with a 2px accent underline. Buttons are not links — `.lbtn`, `.choice` and
  `.cta` each set their own `font:` shorthand, which beats the bare `a` rule.
- **One filled button.** `.cta` — accent ground, cream text — is the end-of-report call
  to the other dataset, the one action the page asks for. Everything else is an outline
  button (`.lbtn`, `.choice`), a plain icon link (`.ilink`, in the footer), or prose.
- **Two CSS traps, both hit and both commented in place.** `section` must use
  `minmax(0,1fr)`, never a bare `1fr`: a child with a definite width wider than the
  column grows the track past the page, and every percentage resolved against that grid
  area then points right of centre (measured: the comparison landed 116px off). And
  `.cmp th` sets the rule, alignment and padding for every cell, so a `.cmp-k` override
  has to out-specify it — `.cmp th.cmp-k` — not merely follow it.
- **A link that leaves the page says so**, with an arrow that is *drawn* — `EXT_ARROW`,
  an inline SVG at `stroke-width:2` in `currentColor`. As a glyph (U+2197) it is a
  hairline in most faces and a different shape in every one, and this page is printed and
  screenshotted. `inline_md()` adds it to any absolute link automatically, and
  `linkbutton()` carries it too.
- **One theme, aged paper.** `render.py` declares `color-scheme:only light` and emits a
  matching `<meta>`. `only` is load-bearing: it opts the page out of Chrome-Android and
  Samsung Internet's auto-darkening, which `prefers-color-scheme` does not cover.
- **The palette is contrast-verified, not eyeballed.** The page is `#f7f4ea` warm cream,
  panels `#f1ebdd`, code and table heads `#e9e1cd`, rules `#cec3a6`.
  `test_text_contrast_meets_wcag_aa` recomputes WCAG ratios from the CSS tokens
  themselves and fails if any text-on-surface pair drops below 4.5:1. Cream is much less
  forgiving than white: the pale chip washes only reach ~1.15:1 against the page, so
  every chip carries a tinted `--*-edge` hairline, and `segbar()` draws no text inside
  its bars.
- **Reserved status colours.** `--good`/`--warn`/`--bad` are not series hues.
  `test_status_colors_are_not_series_colors` pins the separation. Direction is carried by
  a labelled chip, so a status colour never travels alone.
- **Arm colours follow the arm.** `hbar(color=...)` takes a sequence; pass `R.ARM_PAIR`
  for any (control, pipeline) chart. Without it `hbar` colours bars by row order — that
  is how the headline chart came to paint the pipeline in the control's own colour.
- **British English in prose, American in code.**
- **stdlib only**, and no imports from `viewer/` or `shared/` — the page has to build
  where the pipeline's dependencies are not installed, which is also what makes it
  portable. Cost: the row-building helpers in `viewer/rendering.py` are re-implemented
  here, so a schema change to `audit_report.json` can drift.
- Both DAD audit schemas render: modern (`valuable_welfare_considerations`) and legacy
  (reconstructed from `moral_patient_reasons` + `moves.alternatives`, exactly as
  `evals/audit_dad.py` does).

## Checking it renders

The generator's tests assert on the HTML it emits; they cannot tell you where anything
lands on screen. Two layout bugs got through that way — a `1fr` grid track silently
grown past the page by the comparison's wrapper, and a deep link scrolling before the
hero image had claimed its space — so if you are changing layout, measure it:

```bash
apt-get install -y chromium && npm install puppeteer   # chromium must match your arch
node -e "const p=require('puppeteer');(async()=>{
  const b=await p.launch({executablePath:'/usr/bin/chromium',args:['--no-sandbox']});
  const pg=await b.newPage(); await pg.setViewport({width:1440,height:1000});
  await pg.goto('file://\$PWD/report/index.html',{waitUntil:'load'});
  console.log(await pg.evaluate(()=>{
    const th=[...document.querySelectorAll('.cmp thead th')].map(e=>e.getBoundingClientRect());
    return {centre:innerWidth/2, pairMid:(th[0].left+th[1].right)/2};}));
  await pg.screenshot({path:'/tmp/page.png',fullPage:true}); await b.close();})()"
```

`pairMid` must equal `centre`: the two dataset columns straddle the page centre and the
field labels hang off their left, outside the pair.

## Tests

```bash
pytest tests/test_report_common.py tests/test_dad_report.py tests/test_report_page.py
```

172 tests, offline. `test_report_common.py` covers the shared plumbing (prose ids, the
placeholder contract, the provenance floor, the warnings table, the prose count);
`test_dad_report.py` covers the dilemma section along five risk axes — degradation,
self-containment, candour, saying the regression once, colour integrity;
`test_report_page.py` covers the page itself, whose distinctive risks are a report that
cannot be reached, a column that shows nothing when a run is missing, and prose growing
back.

## Finishing the second report

`report/sdf.py` ships the chooser entry, the comparison-table figures and a derived
provenance floor. The full section is a matter of filling in the same beats `dad.py` uses —
`R.sub("sdf-example", ...)` and so on — plus a `content_sdf.md` that takes over the
`sdf_what` / `sdf_soon` ids from `content_page.md` (a rename: the build fails if both
files define one). Three things to know before starting:

- **`derived_warnings()` cannot be shared.** `evals/audit_dad.py` records its verdicts
  into `sections[].rows[]`; `evals/audit_sdf.py` only prints them, so
  `common.audit_verdict_warnings()` returns `[]` for an SDF audit. `sdf.derived_warnings()`
  mirrors the eval's own thresholds instead. Teaching `audit_sdf.py` to record rows the
  way `audit_dad.py` does would give future runs the shared floor for free.
- **`evals/report_sdf.py` on `origin/aidan/sdf-500-run-and-report` is not portable.**
  Roughly half of its 853 lines is editorial prose welded to a 477-document run ("across
  477 documents", "nineteen drifted — 95%"), which is exactly what rule 1 exists to
  prevent. Lift its `excerpt_block()`; write the rest. `render.py` already has
  `histogram()`.
- The only committed SDF run with a full `audit/audit_report.json` is
  `outputs/sdf/runs/2026-07-11_20-06_matrix100-cli` (100 docs). The newer
  `2026-07-13_13-18_al-gap-fixes-100docs` has only `diversity_report.json` and would need
  `python evals/audit_sdf.py --input <run> --patterns` re-run. Per-document layer-5
  scores live in `layer5/scores.jsonl` and in each corpus record's `scores` field.
