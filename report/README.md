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
alias for `--dad-run`. `--content` (repeatable) overrides the prose files, `--example`
overrides the worked example, `--out-dir` writes elsewhere.

**The page does not document how to run the pipeline.** No install, no invocation, no
costs, no per-stage model table — that is this repository's own README and `CLAUDE.md`.
What the page carries is the process, one record's whole trail through it, and caveats.

`--sdf-run` is optional. Without it the synthetic documents' column says "not published
yet" and its report says no audit output was supplied — the page still builds, and
carries no dead links.

The paid audit pass only affects the appendix. Without it, the judged drawer says no paid
pass ran and the derived-flags drawer gains a BAD row; the four beats above the appendix
are unchanged, because none of them depends on a judge:

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
| `#datasets` | The comparison. No heading over it: the two column mastheads (name in serif, one line on what each dataset *is*) are the heading. Four rows — what it is for, what a record is, how many prompt templates, where an example dataset can be read — because the reader is deciding whether to run the pipeline, not shopping for a dataset. **The record count is deliberately not here**: how many records exist is a property of one run, and this section describes the pipelines. Dates, model ids, the composition spread and the counts all live in the report that goes into them. The last two rows carry the way to what they name: the figure (if any) at the column's left edge, an outline button at its right — the templates on GitHub, the published sample on Hugging Face. The `example dataset` row is button-only, so its cells keep an empty first flex item and its buttons line up under the row above. Labels are right-aligned, one line each, vertically centred. |
| `#explore` | "Walk through a dataset generation" — a walkthrough, not results, because roughly half of each report is the worked example and the pipeline that produced it. Two buttons carrying each dataset's name and nothing else, 40rem centred at rest so each sits under its own column, in a bar that pins to the top of the screen while a report is read and tightens as it goes. Both reports are *inside* this section, in `.explore-body` — see "The chooser". |
| `#sdf` | Synthetic documents (`report/sdf.py`) — a placeholder while its full report is written. Hidden until chosen. |
| `#dad` | Difficult advice, in full (`report/dad.py`). Hidden until chosen. |
| footer | Repo and both viewers as buttons, one provenance line per run, and the build claim. |

Both reports take the same skeleton, so a reader learns it once: a one-line lede under the
report's own `<h2>`, then **how it is built / one example end to end / where it is weak /
appendix**. Each beat is an `<h3>` with its own id (`#dad-weak`, `#dad-example`).

What the dataset *is* has no beat: the `<h2>` says "Difficult advice" and the lede says
what that means, in the same sentence the comparison's masthead uses — a reader who
arrived on `#dad` from a deep link never saw that table. A "What it is" heading over one
sentence only names what the reader can already see. (`report/sdf.py` still has an
`sdf-what` heading; its report is a placeholder and takes this shape when it is written.)

Three beats are open and the fourth is drawers, and the line between them is what a reader
has to read:

- **Open**: the process, one record's whole trail through it, and caveats that hold for
  *any* run of the pipeline.
- **Appendix**: everything specific to one run. The judged comparison, the regression
  statement, every chart, every check, the diversity numbers, and the derived floor of
  what this run's audit flagged.

The stages come *before* the example that walks through them, because that is what the
chooser above promises. There is no "what we measured" beat: this is not a results report.

**The page has no contents rail; a report has one.** A column of page-wide links beside a
hero and a comparison is furniture, and it stays gone. But a report is ~2,700 visible words
of records with four beats and seven stages in it, and from inside one a reader could see
neither its shape nor a way past the worked example — so each report carries its own
contents, in the column to its left, sticky under the bar, with the stages nested under
their beat. See "The chooser" below. An earlier revision hung those links as a second row
under the bar instead; it read as clutter on the control and came out.

The type scale is the other thing that makes a report skimmable, and it had none: `h3` (a
beat) was `1.1rem` against a `1.0625rem` body and `h4` (a stage) was `.82rem`, *smaller*
than the prose under it. It steps 2 / 1.4 / 1.12rem now, each level clear of the body text,
and every beat is chunked off the one before it by a hairline above its `<h3>`.
`TestTypeScale` keeps it monotonic. `h4` doubles as a label over a block in exactly one
place — the two halves of a side-by-side — and `h4.pane-h` keeps the old small sans there.

## The chooser

Neither report is open on load — the choice is the point. Three things make that safe,
and all three are pinned by `TestChooser`:

- **`#dad` and `#sdf` in the URL open that report**, on load and on `hashchange`, so the
  dataset card's deep links land where they say they will. A hash naming anything *inside*
  a report (`#dad-weak`, from a quoted finding) opens the report it lives in and scrolls
  to it — that is what `closest('.panel')` in the inline JS is for.
- **The way across is on screen throughout**, because the bar the tabs live in is pinned.
  A report used to end with a filled button offering the other dataset, from when the
  chooser scrolled away behind the reader; a second way across at the foot of every report
  was then a button the page did not need.
- **Printing expands both**, so a PDF of the page is the whole thing.

The cost is real and was accepted deliberately: Cmd-F cannot see a closed report.
`.panel[hidden]{display:none}` is load-bearing — a panel is a `<section>`, and
`section{display:grid}` beats the browser's own `[hidden]` rule.

**The bar is pinned while you read**, and pressing a tab scrolls it to the top of the
screen. `TestStickyBar` pins the six things that make that work:

- **The panels live inside `#explore`**, wrapped with the bar and the rails in
  `.explore-body` (`render.explore_body`). A sticky box travels only inside its containing
  block, and the containing block of a *grid item* is its own grid area — one row, as tall
  as the buttons — so a sticky bar left loose in `#explore`'s grid has nowhere to go. The
  wrapper is the travel: the bar pins for the length of the open report. It is two columns
  and two rows — the bar across the top, then `.railcol` beside `.panels` — and the panels
  are wrapped as **one grid item** on purpose: a grid item stretches to its row's height, so
  the rail's column is as tall as the open report. Left as loose siblings, each panel would
  start a row of its own and the rail would have one panel's worth of travel.
- **`.choicebar` carries the background, `.choices` the buttons.** The band is the full
  column in `var(--surface-0)`, the page's own paper, so the report scrolls under it and
  out of sight; the pair stays centred inside it. A sticky box the width of the buttons
  would let a figure scroll up either side.
- **The rail is the open report's own contents.** `.rail` is a column of jump links to that
  report's `<h3 id>`s with its `<h4 id>`s nested under them, hidden with the panel it
  belongs to and toggled by the same handler (`[data-rail]` in the inline JS), so what a
  reader sees is always the contents of what they are reading. It is **read back off the
  built panel** — `render.outline()` over the markup `report/page.py` just assembled, not a
  module's `BEATS` list — because the beats are conditional: the document report only earns
  `sdf-weak` when its run's audit flagged something, and
  `test_every_rail_link_lands_on_a_heading_that_rendered` builds a clean run to prove a link
  can't advertise a beat that isn't there. A stage becomes a rail item **by having an id**
  (`render.substep()`), which is why the appendix's `<h4>`s deliberately have none: they sit
  inside closed drawers, and a link to a collapsed heading goes nowhere.
- **The room for it came out of the shell, not the report.** `.shell` is 67rem rather than
  the 53rem the page was built at: 12rem of rail plus a 2rem gutter, so the reading column
  keeps its 38rem measure and the figure track its 792px. A rail taken out of the reading
  side would have shrunk the figure track, and every chart is drawn at 800px — an 11px label
  in a 600px track is no longer 11px. `test_the_room_for_the_rail_did_not_come_out_of_the_report`
  recomputes that from the tokens.
- **Where the reader is: ink and a left edge, never a fill.** The current beat or stage takes
  `aria-current`, and the line for "arrived at" is the heading's **own
  `scroll-margin-top`**, read off the element: the CSS already states how far below the top
  of the screen a linked heading lands, so the same number decides whether it has been
  reached. Measured — with the bar's own bottom as the line instead, the marker sat one
  heading behind every jump. Marking runs inside the rAF-throttled scroll callback the bar's
  flag already uses; there is no second listener and no IntersectionObserver.
- **The script measures `.explore-body`, never the bar.** Once sticky takes hold, the
  bar's own `getBoundingClientRect()` and `offsetTop` report where it is *painted*, so
  scrolling to it means scrolling to wherever the reader already was. `.explore-body`'s
  top is the bar's flow top, and that is also the sticky threshold, so nothing jumps as
  the bar pins. Nothing else in the script queries the bar either.
- **The headroom is CSS, not arithmetic.** The bar measures 5.21rem, so `h3[id]`, `h4[id]`
  and `.panel` take `scroll-margin-top:7rem` and a linked beat or stage lands clear of it
  (60px, measured); a native fragment jump reads the same value, and so does the
  current-item pass. `_bar_rem()` in the test file recomputes the height from the six
  tokens, so retuning the bar without revisiting the headroom — or the rail's `top` —
  fails there rather than in a browser. `scrollIntoView()` carries no
  `behavior`, so `html{scroll-behavior}` — and therefore `prefers-reduced-motion` — still
  owns the smoothness.
- **Below 900px there is no beside.** The rail becomes a static wrapped block at the head of
  the report, held to the reading measure so it reads as part of the document. Between 900px
  and the shell's own 67rem the reading column simply narrows, which needs no rule.
- **It has two sizes and crosses between them once.** Loose it is 83px tall and 40rem wide,
  lined up under its heading with the two dataset columns; tight it is 52px and 30rem, with
  the arrow faded out — `↓` means "the report is below", which is stale once the reader is
  in it. It tightens 96px past its own top and loosens again at 24px, animated by a 200ms
  transition. **Two thresholds, not one**, because a reader parked on a single boundary
  flips a layout change back and forth; and a trigger rather than a size that tracks the
  scroll, because tracking meant the bar moved whenever the page did, which reads as
  distraction beside prose. `--t` is a flag (0 loose, `.explore-body.tight` sets 1 — it
  lives on the wrapper because the rail's `top` reads it too, so a tightening bar does not
  leave a growing gap above the contents) and every
  dimension is one interpolation off it, so both states are one set of numbers and a
  breakpoint restates only the six tokens. The 30rem floor is measured, not chosen: below
  27.5rem "Synthetic documents" wraps and the tight bar is *taller* than the loose one, and
  `test_the_bar_has_two_sizes_and_the_tight_one_is_smaller` recomputes it from `--w` and the
  factor. The transition sits on `padding`, `width`, `gap` and `font-size` rather than on
  `--t`, which is both necessary (a custom property is discrete unless registered) and what
  lets the page's own reduced-motion rule turn the animation off with the `transition:none`
  it already applies to everything — measured: 83 → 52 with no frames in between.
- **`overflow-anchor:none` on the wrapper.** Shrinking the pinned bar moves the report
  under it, so scroll anchoring corrects the scroll by the same amount — moving the element
  `--p` is computed from. Measured with anchoring on: the bar settled at 52px while sitting
  31px *below* the top of the screen, or flipped between its two sizes depending on where
  the reader stopped. Resizing every frame costs nothing measurable otherwise (120 scroll
  frames: 16.6ms mean, no frame over 20ms, same as with the driver off).

`#explore>h2` is a child combinator on purpose: every panel opens with its own `<h2>`, so
a descendant selector centres and stretches both report titles too.

The bar stays pinned below 760px, so it is kept to **one row** there — two columns with
tighter tokens (57px at rest, 37px shrunk), and no arrow below 620px. Stacked, the two
buttons are ~10rem of permanent chrome, a quarter of a phone screen.

## Files

| File | Role |
|---|---|
| `content_page.md` | **Page prose**: title, intro, the comparison's cells, the synthetic documents' placeholder text. `*_desc` are the mastheads' subtitles (what each dataset *is*, also used under each chooser button); `*_use` are what each is *for*. The page's own prose interpolates nothing — a `{{placeholder}}` in it is a build error. |
| `content_dad.md` | **Difficult-advice prose.** The file to iterate on for that report. |
| `page.py` | The page: hero, comparison, chooser, footer, and the one `document()` call. |
| `dad.py` | The `#dad` beats: `facts()`, the block builders, `read_lineage()`, `judged_drawer()`, `derived_warnings()`. |
| `sdf.py` | The `#sdf` beats — small on purpose; see "Finishing the second report". |
| `common.py` | Loading, prose parsing, `fill()`, cost aggregation, the provenance warnings, the warnings table, `editorial_words()`, the CLI parser. |
| `render.py` | CSS + inline-SVG chart primitives + the `document()` shell. No pipeline knowledge. |
| `build_report.py` | The CLI. |

Each report module exposes `blocks()`, returning its body as one flat string; `page.py`
wraps that in `render.panel()`, which is the `<section>`, and both panels go inside
`#explore` via `render.explore_body()` so the chooser bar has something to stick to.
Blocks stay flat because a figure has to be a direct child of the section for the CSS grid
to bleed it past the text measure — nesting the panels does not change that: a panel is
still their parent, and `.explore-body` spans the full column, so a panel resolves to the
same width it had as a child of `<main>`.

## The rules

**1. No number is ever typed into a prose file.** Prose interpolates `{{placeholders}}`
resolved from the runs' own output, and an unknown one fails the build. Run-conditional
figures reach prose only with an explicit degraded string — `{{library_clause}}` and
`{{judge_arms_clause}}` — so a run missing the paid pass renders "not measured on this
run" where the figure would be and the sentence survives. The page's own prose has no
facts at all (`PAGE_FACTS = {}`), so any placeholder in `content_page.md` is a build
error. Do not add a bare conditional number to prose; add a clause to the owning module's
`facts()`.

Two blocks are stricter than that and carry **no figure at all**: the `caveats` list,
which is about the method rather than a run, and `dad_what`. Both are pinned by tests
against the shipped prose, because a fixture cannot prove it.

**2. The caveats a reader sees are general; the run's own findings are derived, and in the
appendix.** Two separate things, and the split is deliberate. `caveats` is authored, holds
for any run of this pipeline, and takes no `audit` argument at all, so a run number cannot
get into it. Everything the run's own audit flagged — every BAD/OK verdict, plus
provenance rules (non-`api` backend, uncommitted changes, small n) and DAD-specific rules
(a delivery regression, per-measure arm asymmetry, length inflation, an unmeasured
delivery pass) — is still emitted by `derived_warnings()` whether or not anyone wrote it
up, and renders in the appendix's "What this run's audit flagged" drawer.

Generalising the visible caveats must not lose that floor, and
`test_the_derived_floor_is_still_on_the_page` builds with the caveats prose *emptied* and
asserts every derived row is still there. `evals/audit_sdf.py` only *prints* its verdicts,
so `sdf.derived_warnings()` re-applies the eval's own thresholds instead.
`warnings_table()` may **collapse** rows into a drawer, and the drawer states how many it
holds — collapsing is a view, never a filter.

**3. The judged comparison does not lead, and the delivery regression is stated once.**
The whole comparison against the plain model — considerations, delivery, the scatter, the
scoreboard, retention — is one drawer in the appendix, headed with why it is there. It was
demoted because the delivery pass lost 19 of its 80 judgements on the pinned run, leaving
its two means over 33 pipeline against 26 control answers: different sets of records. A
page that led with that would rest on its least sound measurement.

Demoted is not deleted, and both halves are pinned by tests. The regression is written in
prose exactly once, by `dad._delivery_statement()`, *inside* that drawer — next to the
comparison it is about, because the caveats beat is generalised and a figure from one run
cannot live there. The scoreboard row and the derived weakness carry the same number as
data. No figure of any kind appears outside the appendix —
`test_no_figure_appears_outside_the_appendix` is the restructure in one assertion.

The judged drawer reads **either** audit schema: the old
`valuable_welfare_considerations` + `delivery`, or the `delivery` + `welfare_impact` +
`composite` that PR #107 replaced it with upstream. A run with neither says so.

**4. Synthetic documents comes first**, in the comparison, the chooser and the panel
order, so the page reads in one order throughout.

**5. Both datasets are for midtraining.** "SFT" names the *format* of the difficult-advice
data — chat transcripts, consumed as supervised fine-tuning — not a different training
phase; the documents are consumed as continued pretraining. The "what it is for" row says
both halves, because internal shorthand has the two sounding like different phases.

**6. Prose has a budget, and two ceilings.** The build prints `editorial_words()` for the
page it just wrote. `test_the_prose_has_a_ceiling` bounds the whole page;
`test_the_report_a_reader_reads_has_its_own_ceiling` bounds the difficult-advice beats
*before* the appendix, which is the part that is open when a reader arrives — the
whole-page number is dominated by drawers nobody has to read. The second is the one that
matters: it came down from 1,199 words to under 800 over two rounds of cutting, by dropping
the results narrative, then the cost tiles, the commands and the run-specific caveats. Deks
— the aphoristic line under a heading — are rationed to two for the whole page.

`editorial_words()` counts what a person *wrote*, so it skips corpus text, chart internals,
every table and every `<nav>`. The rails are excluded because their labels are the
document's own headings, already counted where they are written; counting them twice would
spend the ceiling on navigation and let real prose in underneath it.

Section ids in each prose file must exactly match the owning module's `CONTENT_IDS`; a
missing or unknown id is a build error, and two files may not both define one, so moving
a block between prose files is a rename. `example_pick` holds the prompt_id of the DAD
worked example (or `auto`) and `example_extra` the ids in its carousel, so a rebuild
reproduces the same cases without a flag. A pinned id the run never shipped says so on the
page and falls back, rather than failing the build.

## The worked example

`#dad-example` is one record's whole trail through the run, and every block in it is
verbatim from a file in the run directory — the dealt cards, the scenario the planner
wrote from them, the message that shipped, the scope and the library entries stage 2
pulled, the answer, and the three largest things stage 3 changed. Its `<h4>`s reuse the
stage headings from "How it is built" rather than inventing a second vocabulary.

`dad.read_lineage()` assembles it at load time. Two things about the join: only step 1 is
keyed by `scenario_id` and everything downstream by `prompt_id`, so `step1/dilemmas.jsonl`
is the join table, with `audit.gid_map[pid]["scenario"]` as the fallback for a run that
kept no dilemmas file. And `step2/scopes.jsonl` is trimmed on the way in — 725 KB of it is
the reasoning library's prose repeated per case, and the page shows an entry's id,
category and claim.

A missing artefact **names the file it wanted** rather than disappearing, because a step
that silently vanishes reads as a step the pipeline does not have. A key that is not
available is left *absent* rather than set to `None`, so renderers test membership; null
values in the dealt cards are dropped, since rendering an axis with "None" in it is a bug
that reads as data.

Below it, `render.tabs()` puts the ids in `example_extra` behind one set of buttons, using
the chooser's own mechanism — `data-pane`, `aria-selected`, the same inline JS. The first
pane renders *without* `hidden`, so with JS off the carousel degrades to one example
rather than to none, and the print rule expands the rest.

**The carousel is inside a closed drawer.** That visible first pane is a second full
transcript — ~1,250 words on the pinned run — sitting under the pinned record's own trail,
which is what the beat is for; the drawer's summary counts what is behind it, and
`<details>` prints open, so nothing is lost on paper.

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
  marks are inline `<svg>`, the hero is a data URI, and the only JS is a tooltip handler,
  the chooser and the example carousel. Enforced by `test_is_self_contained`, which allows
  a `data:` src and nothing else off-page — and which now looks for `url(` and `@import`
  *outside* the run's own text, because the page quotes three records verbatim and a
  dilemma that happened to contain a CSS snippet would fail a test about the generator.
- **One accent, `--accent:#3b2fa0`.** The page's only interaction colour: the text
  selection and every link. Indigo because it cannot collide with anything the palette
  reserves — far from `--good`, `--warn` and `--bad`, so a selection can never read as a
  verdict, and deeper than `--series-7`, which only appears inside charts. There is no
  separate `--link` token; two names for one hex is how a palette drifts.
- **A link is a typographic object**: `var(--mono)` at `.92em`, weight 600, accent
  coloured, with a 2px accent underline. Buttons are not links — `.lbtn`, `.choice` and
  `.tab` each set their own `font:` shorthand, which beats the bare `a` rule.
- **Filled means selected, not important.** Every control is an outline button (`.lbtn`,
  `.choice`, `.tab`), a plain icon link (`.ilink`, in the footer), or prose; the accent
  ground with cream text appears only on the tab or pane that is currently open. There is
  no primary button — the end-of-report `.cta` went with the sticky bar, which offers the
  other dataset from anywhere in the one being read.
- **Four CSS traps, all hit and all commented in place.** `section` must use
  `minmax(0,1fr)`, never a bare `1fr`: a child with a definite width wider than the
  column grows the track past the page, and every percentage resolved against that grid
  area then points right of centre (measured: the comparison landed 116px off). And
  `.cmp th` sets the rule, alignment and padding for every cell, so a `.cmp-k` override
  has to out-specify it — `.cmp th.cmp-k` — not merely follow it. And a `position:sticky`
  grid item is confined to its own grid area, which is why the chooser bar needs
  `.explore-body` around it and both panels. And a *stuck* sticky element's own
  `getBoundingClientRect()`/`offsetTop` report where it is painted, not where it sits, so
  anything scrolling to it has to measure a static element instead.
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
  is how the considerations chart came to paint the pipeline in the control's own colour.
- **British English in prose, American in code.**
- **stdlib only**, and no imports from `viewer/` or `shared/` — the page has to build
  where the pipeline's dependencies are not installed, which is also what makes it
  portable. Cost: the row-building helpers in `viewer/rendering.py` are re-implemented
  here, so a schema change to `audit_report.json` can drift.
- Every DAD audit schema renders: `valuable_welfare_considerations`, the legacy
  reconstruction from `moral_patient_reasons` + `moves.alternatives` (exactly as
  `evals/audit_dad.py` did), and the `delivery` + `welfare_impact` + `composite` that
  PR #107 replaced both with. Only the appendix's judged drawer reads any of them, so a
  schema change cannot take the report down — the beats above it read the step files.

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
  // The bar: choosing a report puts it at the top, and it stays there while you read.
  await pg.evaluate(()=>document.getElementById('choose-dad').click());
  await new Promise(r=>setTimeout(r,1200));
  const probe=()=>{const b=document.querySelector('.choicebar');
    const r=b.getBoundingClientRect();
    const c=document.querySelector('.choices').getBoundingClientRect();
    const f=document.querySelector('.explore-body').getBoundingClientRect();
    return {tight:b.classList.contains('tight'), barTop:r.top, barH:Math.round(r.height),
            pairW:Math.round(c.width), past:Math.round(-f.top)};};
  const at=async past=>{await pg.evaluate(async past=>{
    const f=document.querySelector('.explore-body');
    window.scrollTo(0,f.getBoundingClientRect().top+scrollY+past);
    await new Promise(r=>setTimeout(r,400));},past);       // let the transition finish
    console.log(await pg.evaluate(probe));};
  await at(0);    // loose  83px x 640px
  await at(95);   // loose  — still, one pixel short of the trigger
  await at(97);   // TIGHT  52px x 480px
  await at(25);   // TIGHT  — still, coming back up: the second threshold
  await at(23);   // loose  again
  await pg.screenshot({path:'/tmp/page.png',fullPage:true}); await b.close();})()"
```

`pairMid` must equal `centre`: the two dataset columns straddle the page centre and the
field labels hang off their left, outside the pair. `barTop` must be `0` in every probe
after the click; the bar must be loose (83px × 640px) at 95px past and tight (52px × 480px)
at 97px, and coming back up it must stay tight to 25px and loosen at 23px — one threshold
each way means a reader stopped on the boundary flips it repeatedly. `past` must equal
exactly what you asked for; if it does not, scroll anchoring is back and the size change is
fighting itself. A deep link (`…/index.html#dad-weak`, `waitUntil:'load'`) must leave that
`<h3>`'s `top` greater than `barH` (measured 82 against the 52px bar), and on a `390x844`
viewport the two buttons must stay on one row (57px loose, 37px tight).

Sample the bar's height over consecutive `requestAnimationFrame`s just after the trigger:
it must pass through intermediate values (measured `83 83 81 76 70 …`). Under
`emulateMediaFeatures([{name:'prefers-reduced-motion',value:'reduce'}])` it must jump
`83 → 52` with nothing in between. Note that `requestAnimationFrame` does not fire in a
backgrounded tab, so call `bringToFront()` on any page you sample this way.

The worked example put two wide tables and six new blocks inside that same grid, which is
the class of change the `1fr` trap bit last time, so measure the overflow too:

```js
console.log(await pg.evaluate(()=>{
  document.querySelectorAll('.panel').forEach(p=>p.hidden=false);
  const s=document.querySelector('#dad');
  const over=[...s.querySelectorAll('*')].filter(e=>e.getBoundingClientRect().right>innerWidth+1)
                                         .map(e=>e.className||e.tagName);
  return {panel:s.getBoundingClientRect().width, vw:innerWidth, overflowing:over.slice(0,5),
          beats:[...s.querySelectorAll('h3[id]')].map(h=>h.id)};}));
// then the carousel: clicking a tab swaps the pane, and only one is visible
await pg.evaluate(()=>document.querySelectorAll('.tab')[1].click());
console.log(await pg.evaluate(()=>[...document.querySelectorAll('.pane-x')].map(p=>p.hidden)));
```

`overflowing` must be empty, `panel <= vw`, and `beats` must read in skeleton order. The
one thing no assertion can check is whether the lineage scans as a walk or as a wall of
`<h4>`s — screenshot it with `#dad` open and read it.

## Tests

```bash
pytest tests/test_report_common.py tests/test_dad_report.py tests/test_report_page.py
```

219 tests, offline. `test_report_common.py` covers the shared plumbing (prose ids, the
placeholder contract, the provenance floor, the warnings table, the prose count);
`test_dad_report.py` covers the difficult-advice section along six risk axes —
degradation, self-containment, candour, not leading with the judge, the lineage naming
what it could not find, colour integrity; `test_report_page.py` covers the page itself,
whose distinctive risks are a report that cannot be reached, a column that shows nothing
when a run is missing, a chooser bar with nowhere to stick or a beat hidden under it, and
prose growing back.

Three of them are the boundary this page keeps being pulled across, and are worth knowing
by name: `test_the_page_does_not_explain_how_to_run_the_pipeline`,
`test_the_caveats_carry_no_run_figures` and `test_the_derived_floor_is_still_on_the_page`.
Slice a beat with `beat(html, anchor)` rather than by `index("id='dad-weak'")` — the naive
slice keeps the next beat's `<h3` and its stray `3` breaks any assertion about digits.

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
