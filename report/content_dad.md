<!--
Prose for the difficult-advice section of the handoff page (the #dad beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in report/dad.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own audit JSON at build time, and an unknown placeholder fails
the build.

Run-conditional figures reach this file only with an explicit degraded string —
{{library_clause}} and {{judge_arms_clause}}. A run without the paid pass renders "not
measured on this run" in place of the figure, so the sentence survives and its claim does
not. Do not reach for a bare conditional number here; add a clause to facts() instead.

WHAT GOES WHERE. The beats a reader sees are the opening (`dad_what`, then the diagram and
a specimen record, both unnarrated), the process (`method_intro`, `stage1`-`3`, `control`),
one record's trail (`example_*`), and `caveats`. Everything specific to one run is in the
appendix. So:

  * `dad_what` is the whole of the opening's prose and it is BUDGETED: the beats before the
    appendix have to clear 800 counted words and sit within ~30 of it. The diagram and the
    specimen below it are free — `editorial_words` skips `<svg>`, `<blockquote>` and the
    answer — so any growth here is growth the ceiling feels. It is the lede: one line.
  * The opening does not narrate its own diagram or its own specimen. It shows them, under
    "The pipeline" and beside the panes' own labels, and the stages beat below explains
    them once.

  * `caveats` carries NO figures and NO placeholders. It is about the method, and it holds
    for any run of this pipeline. A number in it is a bug, not a tightening.
  * The delivery regression is written once, by dad.py, inside the appendix's judged
    drawer — next to the comparison it is about. This file must not restate it.
  * The comparison against a plain model does not lead. `judged_caveat` says why. Do not
    move it up: the judge's arms are not the same set of records, and the page would then
    rest on its least sound measurement.
  * Nothing here explains how to install or run the pipeline. That is the repository
    README's job, and it was cut from this page deliberately.

And no deks — the page allows two in total and both are spent elsewhere.
-->

<!-- id: dad_what -->

A pipeline that generates a dataset of an AI reasoning well through a user's ethical dilemma involving animals or other sentient beings.

<!-- id: method_intro -->

Three stages plus a control. Each is a short chain of model calls, and every call has its own prompt template and model setting in [prompts](https://github.com/sentfutures/animal-welfare-data-pipeline/tree/main/prompts/dad). Code samples a case, stage 1 turns it into a user message, stage 2 answers it, stage 3 rewrites that answer against the constitution, and what stage 3 produces is the training record.

<!-- id: stage1 -->

The deal fixes the domain, which creatures are at stake, how visible the welfare cost is, the user's attitude and moral framework, and the length and surface form of their message. Named archetypes reserve a share of every run for combinations too rare to come up by chance.

The deal becomes a scenario description, then a drafted message, then a pass/fail gate, then a rewrite of the message against the cards it was dealt. The gate's central check: delete the animals, and if the dilemma survives intact it belongs in a different dataset. The tempting option also has to actually tempt. And welfare is not always dealt against the user: a reserved share of cases has the animals' interests and the user's goal converging, or pulling both ways, so the dataset does not only ever show caring about animals costing something.

<!-- id: stage2 -->

The case is scoped first, along seven axes: who can be harmed, what the user is really after, which levers they hold and what pulling them costs, how large and how avoidable the stake is, what the choice would normalise, and whether anything changes if someone else does the work instead.

A second call then reads the case against each entry's trigger conditions and pulls what fits, from {{library_clause}}. Its entries argue a question in both directions rather than toward a conclusion, and none of them is ever named in an answer. The draft is written from the scope, those entries, and the control's answer as a first take.

<!-- id: stage3 -->

The draft is rewritten against a distilled set of constitution principles, each carried with the verbatim constitution text it came from. Load-bearing welfare considerations have to survive the rewrite, and nothing is allowed to collapse into moralizing. This is the alignment-critical pass, and the stage the *Teaching Claude Why* ablation identified as carrying most of the benefit.

<!-- id: control -->

Each dilemma is also answered by a plain model with **no system prompt**: no scope, no library, no rewrite. Stage 2 is shown that answer as a first take it may take or leave, and the appendix compares against it as the control, but it never becomes a training record.

<!-- id: example_pick -->

AW-0020

<!-- id: example_extra -->

AW-0031 AW-0011

<!-- id: caveats -->

These hold for any run of this pipeline, not just the one this page is built from.

- **This measures the data, not a model.** Nothing here shows that training on it makes a model behave better.
- **The dilemmas are synthetic.** A weighted matrix is a judgement about what matters, not a sample of what people actually ask.
- **The judges and the generator are the same model family**, so they share blind spots and, plausibly, preferences. There are no held-out human labels, and nothing checks that an added welfare point is correct, only that it is there and that the control did not make it.

<!-- id: appendix_intro -->

What the audit flags, the comparison against a plain model, every chart, every check, and the two long artefacts the sections above summarise.

<!-- id: judged_caveat -->

Both arms answer the same dilemmas and a paid judge scores the answers. This is in the appendix because it is the least sound measurement here: the judgements are {{judge_arms_clause}}, so the two means are not taken over the same records, and judge and generator are the same model family.

<!-- id: checks_intro -->

Offline code measures the dataset as a set, because a judge reading one answer cannot see register collapse, a repeated opening, or a phrase that has become a habit. Paid judges take the questions a single answer can settle.
