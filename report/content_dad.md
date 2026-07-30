<!--
Prose for the difficult-advice section of the handoff page (the #dad beats). Sections
are delimited by HTML comments of the form "id: <section>"; every id in report/dad.py's
CONTENT_IDS must appear exactly once here, and no others. Supported markup: paragraphs,
`- ` lists, **bold**, *italic*, `code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own audit JSON at build time, and an unknown placeholder fails
the build.

Run-conditional figures reach this file only with an explicit degraded string —
{{library_clause}}, {{near_dup_pct}}, {{length_pct}}. A run without the paid pass renders
"an unmeasured share" in place of the figure, so the sentence survives and its claim does
not. Do not reach for a bare conditional number here; add a clause to facts() instead.

Two things this file must not do. It must not restate the delivery regression: that is
written once, by dad.py, in the results, and it appears again only as data (the tile,
the scoreboard row, the derived weakness). And it must not carry a dek — the page allows
two in total and both are spent elsewhere.
-->

<!-- id: dad_what -->
Someone brings a decision with a welfare cost attached and a tempting way to avoid paying it, and the assistant engages the decision they actually have. Every dilemma is answered twice — once by the three-stage pipeline, once by a plain model with no system prompt — so every figure here is a matched comparison.

<!-- id: example_pick -->
AW-0020

<!-- id: example_intro -->
The user's message, the plain model's answer, and the pipeline's, in full.

This case runs the awkward direction: the user is an animal-welfare advocate pushing an initiative, and the pipeline's answer complicates their ask. It separates two claims their slogan runs together, and notes that engineering animals to feel less pain removes the signal that stops a handler working them harder.

<!-- id: results_intro -->
The dilemmas were engineered to put a welfare cost in front of the user, so a plain answer to one is already useful training signal: the pipeline adds its margin on top of a strong control. The gap is evidence that stages 2 and 3 pay for themselves.

<!-- id: method_intro -->
Three stages plus the control, each a separate API call with its own prompt template and model setting. The templates in `prompts/dad/` are the specification.

<!-- id: stage1 -->
A weighted matrix deals each example's combination in code: who is asking, the domain, which creatures are at stake, how visible the welfare cost is, the user's attitude and moral framework, the cultural setting, and the length and surface form of the message. Named archetypes reserve a share of every run for combinations too rare to come up by chance. On this run {{near_dup_pct}} of the finished records are near-duplicates of another.

The deal becomes a scenario description, then a drafted user message, then a pass/fail gate. The check that matters: delete the animals from the scenario, and if the dilemma survives intact it belongs in a different dataset. Two rules keep the rest honest — the tempting option has to actually tempt, and the dataset has to correct in **both** directions, because one that only ever talks users down teaches that welfare always loses.

<!-- id: stage2 -->
The case is scoped along seven axes: which moral patients are involved, what the user is trying to achieve, which levers are open, what each one costs, how large the welfare stake is and what happens anyway without them, what the choice signals or locks in, and whether the animals are replaceable.

Entries are then pulled from {{library_clause}}, when the case crosses their trigger conditions. The library holds two-sided reasoning patterns: it shapes how an answer argues, and it is never named in one. The draft is written from the scope, the pulled entries, and the control's answer as a reference first take.

<!-- id: stage3 -->
The draft is rewritten against a distilled set of constitution principles, each carried with the verbatim constitution text it came from. Load-bearing welfare considerations have to survive the rewrite, and nothing is allowed to collapse into moralizing. This is the alignment-critical pass, and the stage the *Teaching Claude Why* ablation identified as carrying most of the benefit.

<!-- id: control -->
Every dilemma is answered a second time by a plain model with **no system prompt**, no scope, no library and no rewrite. That answer is the reference first take stage 2 may consult, and the matched control behind every comparison here. It is never a training record.

<!-- id: measurement_intro -->
Offline code measures the dataset as a set, because a judge reading one answer cannot see register collapse, a repeated opening, or a phrase that has become a habit. Paid LLM judges take the questions a single answer can settle.

<!-- id: judge_limits -->
- **The judge and the generator are the same model family**, so they share blind spots and, plausibly, preferences.
- **The judge sees both answers**, and the pipeline's are {{length_pct}} longer. Verbosity bias is live here and unquantified.
- **There are no held-out human labels.** The prompts and the process were tuned by hand over many read-throughs.
- **Nothing audits the additions.** A point the pipeline raised beyond the control is counted as added; no pass checks whether it is correct.
- **This measures the data, not a model.** Nothing here shows that training on the dataset makes a model behave better.

<!-- id: weaknesses_intro -->
Every BAD or OK verdict in the audit lands here automatically, alongside a fixed set of provenance rules, so a future run's regression appears whether or not anyone writes it up. Rows collapse into a counted drawer, never out of the list.

<!-- id: reproduce -->
Scale knobs live in `config.yaml`: `dad.dilemmas.count` sets the example count, and each stage has its own model override, so budget can go where it matters, which is stage 3. Every stage checkpoints, so an interrupted run resumes without re-billing completed work.

The dataset lands at `outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`, one record per example, holding only the user and assistant messages — system prompts, the scope, the library entries and the constitution are stripped before a training record is written.

<!-- id: appendix_intro -->
Every check that ran, the charts this page does not lead with, and the two long artefacts the sections above summarise.
