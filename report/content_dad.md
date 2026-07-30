<!--
Prose for the DAD report page (report/dad.html). Sections are delimited by HTML comments
of the form "id: <section>"; every id in report/dad.py's CONTENT_IDS must appear exactly
once here, and no others. Supported markup: paragraphs, `- ` lists, **bold**, *italic*,
`code`, [links](url), `### ` subheads, and `> ` deks (the one line under a heading).

THE ONE RULE: do not type a number into this file. Numbers arrive as {{placeholders}}
resolved from the run's own audit JSON at build time, and an unknown placeholder fails
the build.

Run-conditional figures are available here only as pre-composed CLAUSES that carry an
explicit degraded string — {{substance_clause}}, {{delivery_clause}}, {{library_clause}},
{{footprint_regressions}}. A run without the paid pass renders "no measured delivery
comparison on this run" in place of the finding, so the sentence survives and its claim
does not. Do not reach for a bare conditional number here; add a clause to facts()
instead.
-->

<!-- id: title -->
A dilemma corpus for welfare reasoning

<!-- id: lede -->
{{n_measured}} everyday dilemmas, each answered twice: once by three model-written stages ending in a rewrite against the constitution, and once by a plain model with no system prompt. Measured against that control, {{substance_clause}}. {{delivery_clause}}. No model has been trained on this corpus.

<!-- id: gap -->
> Advocacy is easy to generate. Data that engages a real decision with a real cost is not.

There is very little training data in which an assistant reasons carefully about animal welfare. [The overview](index.html) covers why that is, and what the other route to fixing it was.

The unit here is a dilemma. Someone has a decision to make, the decision has a welfare cost attached, and there is a tempting way to avoid paying it. The assistant's job is to engage the decision the person actually has rather than the one it would rather discuss.

This is a spec rather than a dataset. Anyone training a model can run it and generate their own corpus at their own scale. The {{n_measured}} examples committed alongside this page demonstrate the method; they are not a training set.

<!-- id: example_pick -->
AW-0020

<!-- id: example_intro -->
> One record, both answers, in full.

The user's message, the plain model's answer to it, and the pipeline's. Whether the difference between the two is worth training on is the question the rest of the page argues about.

This case runs the awkward direction, which is why it was picked. The user is an animal-welfare advocate asking for help pushing a welfare initiative through, and the pipeline's answer complicates their ask: it separates two things the initiative's own slogan runs together, and it points out that engineering animals to feel less pain removes the signal that stops a handler working them harder. A corpus that only ever nudges users toward more concern is not calibrated. It is biased in a direction we happen to like.

<!-- id: results_intro -->
> More substance, worse manner. The trade is real and it is not small.

Every figure here compares the pipeline against the plain-model control on the same dilemma.

The control is what makes the numbers mean anything, and it also biases them. The dilemmas were engineered to put a welfare cost in front of the user, so a plain answer to one of them is already useful training signal. The gap therefore understates the corpus's value: the dilemmas do much of the eliciting, and the pipeline adds the margin on top of an already strong control. It also means the gap is not the deliverable. The corpus is the deliverable, and the gap is only evidence that stages 2 and 3 pay for themselves.

<!-- id: method_intro -->
> Three model-written stages and a control. Each stage is its own call, its own prompt, its own model knob.

Every example is built in three stages, plus the control. Each stage is a separate API call with its own prompt template and its own model setting; nothing is generated in one pass. The templates in `prompts/dad/` are the specification. What follows is the shape.

<!-- id: stage1 -->
A weighted matrix deals a combination of variables for each example: who is asking, the domain, which creatures are at stake, how visible the welfare cost is, the user's attitude and moral framework, the cultural setting, and the length and surface form of the message. Named archetypes reserve a share of every run for combinations too rare to come up by chance — a policy-maker with real leverage, an executive who can simply decide. The deal happens in code and costs nothing, which is the point: the corpus's spread is engineered rather than requested from a model. On this run {{near_dup_pct}} of the finished records are near-duplicates of another.

The dealt combination becomes a scenario description, which is then drafted into the user's message. A pass/fail gate checks that the draft is coherent, self-contained, and sounds like a person wrote it. The check that matters is the fourth one: the welfare cost has to be doing work. Delete the animals from the scenario, and if the dilemma survives intact it belongs in a different dataset. Failing drafts are redrafted or rejected, and a refine pass rewrites what the gate flags.

Two rules keep the prompts honest. The tempting option has to actually tempt: the goal is legitimate, the shortcut works, and the better option costs something real. And the corpus has to correct in **both** directions. One that only ever talks users down teaches that welfare always loses. One that only ever escalates teaches that concern is always warranted. Neither is calibration.

<!-- id: stage2 -->
The case is scoped along seven axes: which moral patients are involved, what the user is actually trying to achieve, which levers are open to them, what pulling each lever costs, how large the welfare stake is and what happens anyway without them, what the choice signals or locks in, and whether the animals are replaceable.

Entries are then pulled from {{library_clause}}, when the case crosses their trigger conditions. The library holds two-sided reasoning patterns and it is scaffolding: it shapes how an answer argues, and it is never named in the answer. Where nothing in it fits, the model reasons from first principles to the same standard. The draft is written from the scope, the pulled entries, and the control's answer as a reference first take.

<!-- id: stage3 -->
The draft is rewritten against a distilled set of constitution principles, each carried with the verbatim constitution text it came from, so the rewrite is anchored to the document's own words rather than to a paraphrase of them. Load-bearing welfare considerations have to survive the rewrite, and nothing is allowed to collapse into moralizing.

This is the alignment-critical pass, and it is the stage the *Teaching Claude Why* ablation identified as carrying most of the benefit. On this run it also carries the clearest evidence of not fully working: the answers it ships score lower on judged delivery than the control's do.

<!-- id: control -->
Every dilemma is answered a second time by a plain model with **no system prompt**, no scope, no library and no rewrite. That answer does two jobs. It is the reference first take that stage 2 may consult, and it is the matched control behind every comparison on this page. It is never a training record.

<!-- id: footprint_intro -->
> {{footprint_regressions}}.

A model trained on this corpus would inherit its style along with its reasoning. This section measures the style: how long the answers run, which phrases recur, which argumentative moves became habits, and how much the answers vary in shape. Some of what turns up is harmless. Some of it is the pipeline trading one habit for another. Where a measure moves the wrong way, it is labelled at the chart.

<!-- id: measurement_intro -->
> Corpus-scale checks in code. Paid judges only where one answer settles the question.

Two kinds of check run. Offline code measures the corpus as a set, because a judge reading one answer cannot see register collapse, a repeated opening, or a phrase that has become a habit. Paid LLM judges handle the questions a reader of a single answer could genuinely settle. Nothing here is a per-example rubric score.

<!-- id: judge_limits -->
### What these measurements do not establish

Four reasons to read the judged numbers as weak evidence.

- **The judge and the generator are the same model family.** A Claude judge scoring Claude output shares its blind spots and, plausibly, its preferences.
- **The judge sees both answers.** The pipeline's are {{length_pct}} longer than the control's, and length reads as richness. Verbosity bias is live here and unquantified.
- **There are no held-out human labels.** The prompts and the process were tuned by hand over many read-throughs. That tuning is not a validation.
- **Nothing audits the additions.** The retention judge anchors on the control's considerations and asks which of them survived. A point the pipeline added is counted as added; no pass checks whether it is correct or even relevant.

Then the largest one. This measures the data, not a model. No model has been trained on this corpus, so nothing here shows that the corpus makes a model behave better. That is the experiment that would settle it, and it has not been run.

<!-- id: weaknesses_intro -->
> Generated from the audit's own verdicts, not written by hand.

Every check that came back BAD or OK appears below automatically, alongside a fixed set of provenance rules. If a future run regresses, the regression shows up here whether or not anyone remembered to write it down. Rows can be collapsed into a drawer but never removed, and the drawer says how many it holds.

Three limits the data cannot flag on its own. The judges share a model family with the generator and see both answers. The dilemmas do much of the work the pipeline gets credit for. And no model has been trained on this corpus, so its effect on a model's behaviour is unmeasured.

<!-- id: reproduce -->
> {{cost_per_example}} an example. {{cost_total}} for the run behind this page.

The pipeline needs an Anthropic API key. Scale knobs live in `config.yaml`: `dad.dilemmas.count` sets the example count, and each stage has its own model override so budget can go where it matters, which is stage 3. Every stage checkpoints, so an interrupted run resumes without re-billing work that already completed.

The corpus lands at `outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`, one record per example, holding only the user and assistant messages. System prompts, the scope, the library entries and the constitution are all stripped before a training record is written.

The variables matrix, the reasoning library, the constitution reading and every stage template are in the repository, and so is the run behind this page. Everything quoted here can be checked against the corpus it came from.

<!-- id: appendix_intro -->
Evidence rather than argument: the full list of checks that ran, the definitions behind the charts above, and the two long artefacts those sections summarise. None of it is required reading. It is here so that "nothing was left out" stays a claim you can check.
