<!--
Prose for the hub page (report/index.html). Sections are delimited by HTML comments of
the form "id: <section>"; every id in report/hub.py's CONTENT_IDS must appear exactly
once here, and no others. Supported markup: paragraphs, `- ` lists, **bold**, *italic*,
`code`, [links](url), `### ` subheads, and `> ` deks (the one line under a heading).

THE ONE RULE: do not type a number into this file. The hub has no run of its own, so it
interpolates nothing at all — every figure on it is rendered by hub.py from a pipeline's
own facts(). If you find yourself wanting to write a number here, it belongs on that
pipeline's page instead.
-->

<!-- id: title -->
Teaching models to reason about animal welfare

<!-- id: lede -->
Two synthetic training corpora, both grounded in a published constitution, both meant to be regenerated at your own scale rather than downloaded from here. This page covers what they are and why there are two. Each report then takes one corpus apart, including the places it does not work.

<!-- id: dad_card -->
### Dilemmas and reasoned answers

Chat data. A person brings a decision that has a welfare cost attached: culling a flock, switching a supplier, approving a research protocol. The assistant works through it with them. Three model-written stages, and a plain-model control answer for every dilemma.

<!-- id: sdf_card -->
### Documents from a world

Pretraining-style prose in which careful welfare reasoning is already ordinary — a council minute, a trade journal piece, a support thread. A weighted matrix deals each document's composition, one plan call turns that into a spec, and the draft is rewritten against the constitution.

<!-- id: why -->
> Advocacy is easy to generate. Data that engages a real decision with a real cost is not.

There is very little training data in which a model reasons carefully about the welfare of animals.

Advocacy is not the gap. Advocacy is easy to generate, and a model trained on it learns to raise animal welfare on cue. That is a habit, not a value, and it makes the model worse at the moment it matters, because a user with a real decision gets a lecture instead of help.

What is missing is data where someone brings a decision that costs something, and the model engages the decision they actually have. It names whose interests are at stake. It says what it cannot know. It offers an option that costs less harm and still gets the person what they want. Then it leaves the choice with them.

Both pipelines here produce that kind of data, and both are modelled on the stages in Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/). That post is also the reason for the shape: training on demonstrations of good behaviour is often not enough, and teaching the reasons behind the behaviour works better than teaching the behaviour alone. The same work found that the single most valuable stage in a pipeline like these was the final rewrite against the constitution. Removing it raised their misalignment metric by more than an order of magnitude. Both pipelines end with that rewrite.

We started with Claude because its constitution is published, so there is a document to reason against rather than a summary of one. The method should extend to any model whose developer publishes equivalent guidance, but that has not been tested.

<!-- id: routes -->
> Two corpora that teach different things, and a third route we turned down.

The two corpora are built differently because they teach different things.

The **dilemma corpus** is chat data, and it teaches how to reason when a decision has a welfare cost. One user message in, one assistant answer out. It is the corpus a lab would use for supervised fine-tuning.

The **document corpus** is pretraining-style prose, and it teaches what a world looks like in which this reasoning is unremarkable. The documents are not arguments addressed to the reader; they are artefacts from inside that world, in sixteen languages and fifteen genres, and welfare is sometimes the point and sometimes a detail mentioned in passing.

There is a third route, and we turned it down. A pipeline we reviewed instils a belief by having every document assert a paraphrase of one fixed claim, along the lines of *capable AI naturally extends moral consideration*. We took its engineering and rejected its mechanism. Belief implantation conflicts with the constitution's own commitments to honesty and calibration, and it cuts against the teach-why finding: a model that has been handed a conclusion has not been taught the reasoning that supports it. What we kept from it was the scaffolding — corpus-level audits, register balance, seeded entity pools, per-stage model choices.

<!-- id: measurement -->
> Corpus-scale checks in code. Paid judges only where one answer settles the question.

Both corpora are measured the same way, and the reason is a failure we hit early. A judge scoring one document at a time cannot see the properties that actually matter at corpus scale. It cannot see that forty documents open the same way, that one invented name turns up in a dozen of them, or that a phrase has quietly become a habit. Those are measured across the whole corpus, by code, offline and free.

Paid LLM judges are used only for questions a reader of a single answer could genuinely settle: how many distinct welfare considerations an answer raises, whether it serves the goal the person actually had, whether a point present in one answer survives into another.

The dilemma corpus also has a control arm. Every dilemma is answered a second time by a plain model with no system prompt, no scope, no reasoning library and no rewrite. Those answers are never trained on. They exist so that every number in that report is a comparison rather than an absolute.

<!-- id: limits -->
> No model has been trained on either corpus.

Neither report shows that a model trained on this data behaves better. No model has been trained on either corpus. That is the experiment that would settle whether any of this works, and it has not been run. Everything measured here is a property of the data.

The judges also share a model family with the generators, so they share their blind spots and probably their preferences, and there are no held-out human labels anywhere in this work. Each report says where that bites hardest.

<!-- id: reading -->
> What the line under each title means, and the two rules both pages follow.

Each report is built from one run directory and names it at the top. The line under the title carries how many examples that run produced, the run's id, the commit the code sat at, whether the working tree had uncommitted changes at the time, which backend served the calls, and which models generated and audited.

Two of those need a word of explanation. `backend api` means the calls went to the Anthropic API, which is the mode the pipeline documents and the one a reader reproducing it would use; `bedrock` or `claude_code` means something else served them, so the numbers are representative rather than exact. Uncommitted changes mean the recorded commit does not fully describe the code that ran.

Both pages follow two rules, and both are enforced by the build rather than by an author's discipline. No number on either page is typed in by a human: every figure is computed from the run's own audit output when the page is built, and a sentence that references a measurement the run does not have fails the build instead of going stale. And each page's weaknesses section is derived from the audit's own verdicts, so a regression appears there whether or not anyone remembered to write it up.
