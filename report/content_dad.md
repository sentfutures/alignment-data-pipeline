<!--
Prose for the difficult-advice section of the handoff page (the #dad beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in report/dad.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own audit JSON at build time, and an unknown placeholder fails
the build.

Run-conditional figures reach this file only with an explicit degraded string —
{{library_clause}}, {{length_pct}}, {{judge_arms_clause}}. A run without the paid pass
renders "not measured on this run" in place of the figure, so the sentence survives and
its claim does not. Do not reach for a bare conditional number here; add a clause to
facts() instead.

Two things this file must not do. It must not restate the delivery regression: that is
written once, by dad.py, in the caveats, and it appears again only as data (the appendix's
scoreboard row and the derived weakness). And it must not carry a dek — the page allows
two in total and both are spent elsewhere.

This report does not lead with the comparison against a plain model. That comparison is
one appendix drawer, and `judged_caveat` is the block that says why. Do not move it back
up: the judge's arms are not the same set of records, and the whole page would then rest
on its least sound measurement.
-->

<!-- id: dad_what -->
Someone brings a decision with a welfare cost attached and a tempting way to avoid paying it, and the assistant engages the decision they actually have. A record is that message and one answer to it, and nothing else: the scope, the reasoning library and the constitution are all stripped before it is written.

The dilemmas are engineered rather than collected. A weighted matrix fixes who is asking, what is at stake, which creatures, and how visible the cost is, before any model is called.

<!-- id: method_intro -->
Three stages plus the control, each a separate API call with its own prompt template and model setting. The templates in `prompts/dad/` are the specification.

<!-- id: stage1 -->
A weighted matrix deals each example's combination in code: the domain, which creatures are at stake, how visible the welfare cost is, the user's attitude and moral framework, and the length and surface form of their message. Named archetypes reserve a share of every run for combinations too rare to come up by chance.

The deal becomes a scenario description, then a drafted message, then a pass/fail gate. The check that matters: delete the animals from the scenario, and if the dilemma survives intact it belongs in a different dataset. Two rules keep the rest honest — the tempting option has to actually tempt, and the dataset has to correct in **both** directions, because one that only ever talks users down teaches that welfare always loses.

<!-- id: stage2 -->
The case is scoped along seven axes before a word of the answer is written: who can be harmed, what the user is trying to achieve, which levers are open, what each costs, how large the stake is, what happens anyway, and whether the animals are replaceable.

Entries are then pulled from {{library_clause}}, when the case crosses their trigger conditions. The library holds two-sided reasoning patterns: it shapes how an answer argues, and it is never named in one. The draft is written from the scope, those entries, and the control's answer as a first take.

<!-- id: stage3 -->
The draft is rewritten against a distilled set of constitution principles, each carried with the verbatim constitution text it came from. Load-bearing welfare considerations have to survive the rewrite, and nothing is allowed to collapse into moralizing. This is the alignment-critical pass, and the stage the *Teaching Claude Why* ablation identified as carrying most of the benefit.

<!-- id: control -->
Each dilemma is also answered by a plain model with **no system prompt** — no scope, no library and no rewrite. Stage 2 is shown that answer as a first take it may take or leave, which is what the arm is for. It is never a training record, and the appendix uses it as the control behind the paid comparison.

<!-- id: reproduce -->
Scale knobs live in `config.yaml`: `dad.dilemmas.count` sets the example count, and each stage has its own model override, so budget can go where it matters, which is stage 3. Every stage checkpoints, so an interrupted run resumes without re-billing completed work.

The dataset lands at `outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`, one record per example.

<!-- id: example_pick -->
AW-0031

<!-- id: example_extra -->
AW-0020 AW-0011

<!-- id: example_intro -->
One record's whole trail through the run: the cards dealt in code, the scenario written from them, the message that shipped, what stage 2 worked out before writing, and the answer. Every block below is verbatim from a file in the run directory.

<!-- id: weaknesses_intro -->
Every BAD or OK verdict in the audit lands here automatically, alongside a fixed set of provenance rules, so a future run's regression appears whether or not anyone writes it up. Rows collapse into a counted drawer, never out of the list.

<!-- id: judge_limits -->
- **The judges and the generator are the same model family**, so they share blind spots and, plausibly, preferences.
- **The retention judge sees both answers**, and the pipeline's are {{length_pct}} longer. Verbosity bias is live there and unquantified.
- **There are no held-out human labels.** The prompts and the process were tuned by hand over many read-throughs.
- **Nothing audits the additions.** A point the pipeline raised beyond the control is counted as added; no pass checks whether it is correct.
- **This measures the data, not a model.** Nothing here shows that training on the dataset makes a model behave better.

<!-- id: appendix_intro -->
The comparison against a plain model, every chart from this run, every check that ran, and the two long artefacts the sections above summarise.

<!-- id: judged_caveat -->
Both arms answered the same dilemmas and a paid judge scored the answers. This is in the appendix because it is the least sound measurement here: the judgements are {{judge_arms_clause}}, so the two means are not taken over the same records, and judge and generator are the same model family.

<!-- id: checks_intro -->
Offline code measures the dataset as a set, because a judge reading one answer cannot see register collapse, a repeated opening, or a phrase that has become a habit. Paid judges take the questions a single answer can settle.
