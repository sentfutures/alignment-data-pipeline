<!--
Prose for the landing page (report/index.html). Sections are delimited by HTML comments
of the form "id: <section>"; every id in report/hub.py's CONTENT_IDS must appear exactly
once here, and no others. Supported markup: paragraphs, `- ` lists, **bold**, *italic*,
`code`, [links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. The landing page has no run of its
own, so it interpolates nothing at all — every figure on it is rendered by hub.py from a
pipeline's own facts(). A {{placeholder}} here fails the build.

This is a way in, not a document. Hero, two buttons, then roughly 490 words: why the
data is missing, what the two corpora are, and what neither of them shows. Anything a
reader only needs while reading a report belongs on that report's page — which is where
the provenance conventions, the measurement philosophy and the build rules now live.

In `corpora`, the dek and the first paragraph render ABOVE the two cards and everything
after them renders below (see hub._split_around_cards), so the third-route paragraph
lands after the reader has seen what the two routes we did take actually are.
-->

<!-- id: title -->
Teaching models to reason about animal welfare

<!-- id: lede -->
Two synthetic training corpora, built against a published constitution, and honest measurements of what each one does and does not do.

<!-- id: why -->
> Advocacy is easy to generate. Data that engages a real decision with a real cost is not.

There is very little training data in which a model reasons carefully about the welfare of animals.

Advocacy is not the gap. Advocacy is easy to generate, and a model trained on it learns to raise animal welfare on cue. That is a habit rather than a value, and it fails at the moment that matters, because a user with a real decision gets a lecture instead of help.

What is missing is data where someone brings a decision that costs something and the model engages the decision they actually have: whose interests are at stake, what it cannot know, which option costs less harm and still gets the person what they want. Then it leaves the choice with them.

Both corpora are modelled on Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/), which found that teaching the reasons behind a behaviour beats teaching the behaviour, and that the most valuable stage was the final rewrite against the constitution. Both pipelines end with that rewrite. We started with Claude because its constitution is published, so there is a document to reason against rather than a summary of one.

<!-- id: corpora -->
> They are built differently because they teach different things.

The **dilemma corpus** teaches how to reason when a decision has a welfare cost: one user message in, one assistant answer out, ready for supervised fine-tuning. The **document corpus** teaches what a world looks like in which that reasoning is unremarkable, across sixteen languages and fifteen genres, with welfare sometimes the point and sometimes a detail mentioned in passing.

There is a third route, and we turned it down. A pipeline we reviewed instils a belief by having every document assert a paraphrase of one fixed claim, along the lines of *capable AI naturally extends moral consideration*. Belief implantation conflicts with the constitution's own commitments to honesty and calibration, and it cuts against the teach-why finding: a model handed a conclusion has not been taught the reasoning for it. We kept that pipeline's engineering — corpus-level audits, register balance, seeded entity pools, per-stage model choices — and rejected its mechanism.

<!-- id: dad_card -->
### Dilemmas and reasoned answers

A person brings a decision that has a welfare cost attached, and the assistant works through it with them. Three model-written stages, and a plain-model control answer for every dilemma to measure against.

<!-- id: sdf_card -->
### Documents from a world

Prose in which careful welfare reasoning is already ordinary: a council minute, a trade journal piece, a support thread. A weighted matrix deals each document's composition before any model is called.

<!-- id: limits -->
> What neither corpus can tell you yet.

Neither report shows that a model trained on this data behaves better. No model has been trained on either corpus. That is the experiment that would settle whether any of this works, and it has not been run. Everything measured is a property of the data.

The judges also share a model family with the generators, so they share their blind spots and probably their preferences, and there are no held-out human labels anywhere in this work. Each report says where that bites hardest, and each derives its own weaknesses section from its audit's verdicts rather than from anything written by hand.
