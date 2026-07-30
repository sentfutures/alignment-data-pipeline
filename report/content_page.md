<!--
Prose for the handoff page's own sections (report/index.html). Sections are delimited by
HTML comments of the form "id: <section>"; every id in report/page.py's CONTENT_IDS and
report/sdf.py's CONTENT_IDS must appear exactly once across the prose files, and no
others. Supported markup: paragraphs, `- ` lists, **bold**, *italic*, `code`,
[links](url), `### ` subheads, and `> ` deks.

THE ONE RULE: do not type a number into this file. Figures arrive as {{placeholders}}
resolved from the pinned runs' own output at build time, and an unknown placeholder
fails the build. The page's own prose has NO facts available at all — every figure on it
is rendered by a section from its run — so any {{placeholder}} here fails the build.

The hero is the illustration, the title and `intro`, centred — so `title` has to stand up
on its own, and `intro` reads as its second half rather than as a section. Two paragraphs
and it stops: the two datasets are named once, below, as the comparison's masthead. The
arrow on an outbound link is added by the renderer — do not type one here. The reader has
forty seconds: let the comparison do the comparing. Deks are rationed: the page carries
at most two.

`dad_desc` / `sdf_desc` are the mastheads' subtitles — what each dataset *is*, in one
sentence — and double as the line under each chooser button. `dad_use` / `sdf_use` are
what each is *for*: both are midtraining, and the difference is the format they are
consumed in. One short sentence each.

The licence is not prose: it renders from `page.LICENCE_TODO` as a chip in the comparison,
so it stays visible until someone sets one.

When the synthetic documents' full report lands, `sdf_what` and `sdf_soon` move to a
content_sdf.md of their own — moving an id between prose files is a rename, never a
copy, and the build fails if both files define it.
-->

<!-- id: title -->

Teaching models to reason about animal welfare

<!-- id: intro -->

Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) describes how teaching a model the reasons behind a behaviour beats teaching the behaviour itself, and that the final rewrite against the constitution carries most of the benefit.

We have built two training dataset generation pipelines based on their methods, for a neglected subject almost no training data covers well: the welfare consideration of nonhuman sentient beings.

<!-- id: dad_desc -->

Examples of an AI reasoning well in response to a user's ethical dilemma that concerns the welfare of animals and other sentient beings.

<!-- id: sdf_desc -->

Prose from a world in which animals and other sentient beings are reasoned about carefully.

<!-- id: dad_use -->

Midtraining, as supervised fine-tuning on chat transcripts.

<!-- id: sdf_use -->

Midtraining, as continued pretraining on documents.

<!-- id: dad_unit -->

One user dilemma in, one assistant answer out.

<!-- id: sdf_unit -->

One standalone document, no chat framing.

<!-- id: sdf_what -->

Prose in which careful welfare reasoning is already ordinary: a council minute, a trade journal piece, a support thread. A weighted matrix deals each document's composition in code before any model is called — language, genre, register, domain, and how central the welfare question is, down to a reserved share where it is a detail mentioned in passing.

<!-- id: sdf_soon -->

The full report for this dataset is in preparation and will take the same shape as the other. Until it lands, the run's audit output and every stage template are in the repository.
