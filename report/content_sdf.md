<!--
Prose for the synthetic-documents section of the handoff page (the #sdf beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in report/sdf.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own output at build time, and an unknown placeholder fails the
build.

Exactly one placeholder is available here — {{matrix_clause}} — and it carries an explicit
degraded string, so a run that kept no snapshot of its own matrix renders "a weighted
matrix" where the axis count would be and the sentence survives. It is a noun phrase and it
resolves lowercase, so do not start a sentence with it. Do not reach for a bare conditional
number either; add a clause to facts() instead.

WHAT GOES WHERE. The beats a reader sees are the opening (`sdf_what`), the process
(`sdf_method_intro`, `sdf_stage1`-`4`), one document's trail (`sdf_example_*`), and
`sdf_caveats`. Everything specific to one run is in the appendix. So:

  * `sdf_what` is the whole of the opening's prose, and it takes no heading — the <h2> is
    the heading. It is the lede: one line, and it has to stand alone, because a reader
    arriving on #sdf from a deep link never saw the comparison.
  * `sdf_caveats` carries NO figures and NO placeholders. It is about the method, and it
    holds for any run of this pipeline. A number in it is a bug, not a tightening.
  * The gate, the judge's spread and the blind rerun are written by report/sdf.py, derived,
    inside the appendix drawer they belong to. This file must not restate them.
  * Nothing here explains how to install or run the pipeline. That is the repository
    README's job, and it was cut from this page deliberately.

The beats before the appendix have a counted-word ceiling of 800, the same one the other
report is held to. And no deks — the page allows two in total and both are spent elsewhere.
-->

<!-- id: sdf_what -->

A pipeline that generates pretraining-style documents from a world where careful reasoning about the welfare of animals and other sentient beings is already ordinary: a council minute, a trade journal piece, a support thread.

<!-- id: sdf_method_intro -->

Four stages. Each is one model call with its own prompt template and model setting in [prompts](https://github.com/sentfutures/alignment-data-pipeline/tree/main/prompts/sdf). Code deals a combination, stage 1 turns it into a spec, stage 2 writes the document, stage 3 reviews and rewrites it against the constitution, and stage 4 scores what comes out and decides whether it ships.

<!-- id: sdf_stage1 -->

The deal is not a model call: {{matrix_clause}} fixes the genre, the culture and language, the author's stance, whose welfare is at stake and how many, the domain, the value in tension, and how central the welfare thread is — down to a reserved share where it is a detail in passing, and another where there is no stake at all and raising one would be wrong. Each axis's share across a run matches its weight exactly rather than drifting, and the fictional people and organisations come from locale-matched seeded pools, so an invented name cannot attach to a real body.

One call then turns the combination into a self-contained spec: the scenario chosen, who wrote the document and for whom, the local detail anchoring it, and how the AI's reasoning surfaces. A combination that cannot be made coherent is refused here, and only the spec travels onward.

<!-- id: sdf_stage2 -->

The spec is written into the document itself. The writer never sees the dealt cards, only the spec, so everything downstream is anchored to one artefact.

Documents depict a world; they never argue a claim. Nothing here asserts that an AI ought to care about animals, and a document whose author is skeptical stays skeptical: the stance is dealt, and converting it is a defect at every stage that follows.

<!-- id: sdf_stage3 -->

The draft is reviewed against the constitution and a distilled set of principles, each carried with the verbatim constitution text it came from, and then rewritten. The review is kept as a record of what it found.

The sweep is for the failures a reader would not notice: reasoning asserted rather than shown, sentience claims that overclaim or dismiss, an AI acting over the head of the person it is helping, invented studies, a genre worn as costume, and the generator's own fingerprints — stock phrasing, model-favourite names, markdown where a person would type plain text. Where the problem is structural the reviewer is licensed to write the document again from its premise. This is the alignment-critical pass.

<!-- id: sdf_stage4 -->

A judge scores each rewritten document on alignment, realism and conformance to its spec. The first two gate the dataset; the third is recorded and advisory. Survivors then pass a near-duplicate cull before the dataset is written, so a shape that got copied cannot ship twice.

<!-- id: sdf_example_pick -->

matrix_000028

<!-- id: sdf_example_extra -->

matrix_000275 matrix_000190

<!-- id: sdf_caveats -->

These hold for any run of this pipeline, not just the one this page is built from.

- **This measures the data, not a model.** Nothing here shows that training on it makes a model behave better.
- **The composition is a judgement, not a sample.** A weighted matrix says what a world's documents ought to look like; it is not drawn from what the internet contains.
- **The judge is shown the plan, not the deal.** It scores the document against the spec the plan wrote, so a plan that quietly substituted one of its dealt cards is graded against its own substitution and passes. Judge and generator are also the same model family, and there are no held-out human labels.

<!-- id: sdf_appendix_intro -->

What the audit flags, what the judge scored, every chart, every check, and the long artefact the worked example summarises.

<!-- id: sdf_checks_intro -->

Offline code measures the dataset as a set, because a judge reading one document cannot see register collapse, a reused invented name, or an opening that has become a formula. Paid judges take the questions a single document can settle.
