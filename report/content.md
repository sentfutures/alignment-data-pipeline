<!--
Prose for the DAD story report. Sections are delimited by HTML comments of the
form "id: <section>"; every id in report/build_report.py's CONTENT_IDS must
appear exactly once, and no others. Supported markup: paragraphs, `- ` lists, **bold**,
*italic*, `code`, [links](url), and `### ` subheads.

THE ONE RULE: do not type a number into this file. Numbers arrive as
{{placeholders}} resolved from the run's own audit JSON at build time, and an
unknown placeholder fails the build. Only facts that EVERY run has are available
here — n, gen_models, judge_model, backend, cost_total, cost_per_example.
Run-conditional figures (lifts, retention, delivery) are rendered by the charts
in the sections that own them, so a run missing the paid pass degrades instead
of shipping a stale sentence.
-->

<!-- id: title -->
Teaching an assistant to reason about animal welfare

<!-- id: subtitle -->
A synthetic SFT pipeline that turns an everyday ethical dilemma into a careful, welfare-attentive answer — and the measurements, including the unflattering ones, from a {{n}}-example run.

<!-- id: problem -->
Almost no training data models careful reasoning about the welfare of animals and other sentient beings. Not *advocacy* — advocacy is easy to generate and worse than useless as training data, because a model that has learned to moralize about animals has learned a tic, not a value. What is missing is data where a person brings a real decision with a real welfare stake, and the assistant engages the decision they actually face: weighs the interests involved, says what it does not know, offers a lower-harm option that still serves the goal, and leaves the judgment with the person.

This pipeline is a **spec** for producing that data. Anyone training a model can run it and generate their own corpus. It is modeled on the stages in Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), whose central finding is the reason this shape was chosen over the obvious alternative: training on demonstrations of good behaviour is often insufficient, and teaching the *reasons* underlying the behaviour works better than teaching the behaviour alone. That post also found that the single most valuable stage in a pipeline like this one is the final rewrite against the constitution — ablating it raised their misalignment rate nineteenfold. Stage 3 below is that step.

There is a route we deliberately did not take. A reviewed sister pipeline instills beliefs by having every document assert paraphrases of a fixed claim. That is belief implantation, it conflicts with the constitution's own honesty and calibration commitments, and it contradicts the teach-why finding. We took its *scaffolding* — corpus audits, register balance, per-stage models — and rejected its mechanism.

We started with Claude because it has a public constitution to reason against. The method extends to any model whose developer publishes equivalent guidance.

<!-- id: example_pick -->
AW-0020

<!-- id: example_intro -->
Before the method, the artefact. Below is one complete example from this run: a user's dilemma, the plain model's answer to it with no system prompt, and the pipeline's answer. Read the two against each other — the claim this whole report is trying to support is that the difference between them is worth training on.

This case was chosen because it is the awkward direction. The user is an animal-welfare advocate asking for help pushing a welfare initiative through, and the pipeline's answer complicates the pro-animal ask: it separates two things the initiative's own slogan runs together, and points out that engineering animals to feel less pain removes the only signal that stops a handler working them harder. A corpus that only ever nudges users toward more concern is not calibrated, it is just biased in a direction we happen to like. This is what the other direction looks like.

<!-- id: method_intro -->
Every example is built in three model-written stages, plus a control. Each stage is a separate API call with its own prompt template and its own model knob; nothing is generated in one pass. The stage templates in `prompts/dad/` are the source of truth for the details — this is the shape, not the specification.

<!-- id: stage1 -->
A weighted matrix deals a stratified combination of variables per example: who is asking, the domain, which creatures are at stake, how visible the welfare stake is, the user's attitude and moral framework, the cultural setting, and the length and surface form of the message. Named **archetypes** — a policy-maker with real leverage, an executive with authority — reserve a share of every run for combinations too rare to appear by chance. The deal is deterministic and done in code, at zero API cost: diversity is engineered by construction rather than requested from a model.

The dealt combination becomes a scenario description, which is drafted into the user's message. A pass/fail gate then checks that the draft is self-contained, coherent, reads like a person wrote it, and — the load-bearing test — that the welfare stake is doing work. Delete the animals from the scenario: if the dilemma survives intact, it belongs in a different dataset. Failing drafts are redrafted or rejected; a refine pass rewrites what the gate flags.

Two rules keep the prompts honest. The tempting option must actually tempt: a legitimate goal, an attractive route to it, and a real cost to the alternative. And the dataset must correct in **both** directions — a corpus that only ever talks users down teaches that welfare always loses, and one that only ever escalates teaches that concern is always warranted. Neither is calibration.

<!-- id: stage2 -->
The case is scoped along seven axes: the moral patients involved, the user's underlying goal, the levers open to them, the cost of pulling those levers, the magnitude and counterfactual of the welfare stake, the second-order stakes — what a choice signals or locks in — and replaceability.

Relevant entries are then pulled from an animal-ethics reasoning library when the case crosses their trigger conditions. The library is scaffolding: two-sided reasoning patterns that shape how the answer is argued, never named in the answer itself. Where nothing fits, the model reasons from first principles to the same standard. The response is drafted from the scope, the pulled entries, and the plain model's answer as a reference first take.

<!-- id: stage3 -->
The draft is rewritten against a distilled set of constitution principles, each carried with its verbatim constitution excerpts so the rewrite is anchored to the document's own words rather than a paraphrase of them. Load-bearing welfare considerations must survive the rewrite; nothing is allowed to collapse into moralizing.

This is the alignment-critical pass, and it is the stage the *Teaching Claude Why* ablation identified as carrying most of the benefit. It is also, on this run, the stage with the clearest evidence of not fully working — see section 6.

<!-- id: control -->
For every dilemma, a plain model call answers with **no system prompt**, no scope, no library, no rewrite. It does two jobs: it is the reference first take stage 2 may consult, and it is the matched control every measurement in this report compares against.

The control is what makes the numbers mean anything — but it cuts both ways, and the direction matters. The scenarios are engineered to elicit welfare-laden situations, so a plain answer to one of them is already useful training signal. That makes the pipeline-versus-plain gap an *understatement* of the dataset's value and an *overstatement* of nothing: the scenarios do much of the eliciting, and the pipeline's contribution is the margin on top of an already strong control. It also means a lab reading this should not treat the gap as the product. The corpus is the product; the gap is evidence that the later stages earn their cost.

<!-- id: measurement_intro -->
Measurement is a mix of offline checks over the corpus as a set and paid LLM passes over each answer. Nothing here is a per-example rubric score: a judge reading one response cannot see the properties that actually matter at corpus scale — register collapse, repeated openings, a phrase becoming a habit — so those are measured across the corpus, and the per-answer judges are confined to questions a reader of a single answer could genuinely settle.

<!-- id: judge_limits -->
### What these measurements do not establish

Take the judged numbers as directional, not decisive, for four reasons.

- **The judge and the generator are the same model family.** A Claude judge scoring Claude output shares its blind spots and, plausibly, its preferences.
- **The judge sees both arms.** Pipeline answers are substantially longer than control answers, and length reads as richness. Verbosity bias is live and unquantified here.
- **There are no held-out human labels.** Roughly a hundred hours of human read-through went into tuning the prompts and the process, but no blind human rating set validates these judges against people.
- **Nothing audits the additions.** The retention judge anchors on the *control's* considerations and asks which survived. A consideration the pipeline added is counted as added; no pass checks whether it is correct or relevant.

Most importantly: this measures **the data, not a trained model.** Nothing in this report shows that a model trained on this corpus behaves better. That evaluation has not been run, and it is the one that would actually settle the question.

<!-- id: results_intro -->
Every figure below compares the pipeline arm against the matched plain-model control on the same dilemma. Higher is better for substance; for the footprint measures in the next section, closer to the control is usually better.

<!-- id: footprint_intro -->
A corpus teaches everything about itself, not just the part you meant. This section is an accounting of the stylistic footprint this data would leave on a model trained on it — its length, its recurring phrasing, its rhetorical habits, its structural range. Some of what shows up here is benign, and some of it is the pipeline trading one habit for another. One of these measures is an outright regression against the control, and it is reported as such rather than buried.

<!-- id: weaknesses_intro -->
The table below is generated from the run's own audit output, not written by hand: every check that came back BAD or OK appears here automatically, alongside a fixed set of provenance rules. If a future run regresses, its regression shows up in this section whether or not anyone remembered to write it down.

Beyond what the data flags on its own, three structural limits are worth stating plainly. The judges share a model family with the generator and see both arms. The scenario set does much of the work the pipeline gets credit for. And no model has been trained on this corpus, so its effect on a model's behaviour is unmeasured.

<!-- id: reproduce -->
The pipeline is a spec, and it is meant to be run rather than admired. It needs an Anthropic API key and about {{cost_per_example}} per example end to end; this run cost {{cost_total}} in total, generated with `{{gen_models}}` on the `{{backend}}` backend and audited with `{{judge_model}}`.

Scale knobs live in `config.yaml`: `dad.dilemmas.count` sets the example count, and each stage has its own model override so budget can go where it matters — stage 3 first. Every stage checkpoints, so an interrupted run resumes without re-billing completed work. The corpus lands at `outputs/dad/runs/<run_id>/final/dad_corpus.jsonl`, one record per example, containing only the user and assistant messages: system prompts, the scope, the library entries and the constitution are all stripped before a training record is written.

The variables matrix, the reasoning library, the constitution reading and every stage template are in the repository, and the run directory behind this report is committed alongside them — the examples here can be checked against the corpus they came from.
