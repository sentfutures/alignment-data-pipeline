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
on its own, and `intro` reads as its second half rather than as a section. Three paragraphs
and it stops: the two datasets are named once, below, as the comparison's masthead. The
arrow on an outbound link is added by the renderer — do not type one here. The reader has
forty seconds: let the comparison do the comparing. Deks are rationed: the page carries
at most two.

The comparison is six rows, and each one says whether it is describing the data or the
process that makes it. `dad_desc` / `sdf_desc` are the `result` row — what each dataset
*is*, in one sentence. `dad_use` / `sdf_use` are what each is *for*: both are midtraining,
and the difference is the format they are consumed in. `dad_unit` / `sdf_unit` are the
`result format` — what one record is. `dad_pipeline` / `sdf_pipeline` are the stages that
produce it, as a chain, in the same shape on both sides so the two columns can be read
across. One short line each; a row's LABEL lives in `page.section_datasets()`, only its
cells are here. The `*_pipeline` chains carry a NON-BREAKING space after each arrow, so a
wrap puts the arrow at the head of the next line rather than orphaning it at the end of
the last one — it is invisible in an editor, so copy an existing arrow rather than typing
a new one.

No licence is set for either dataset, and the page says nothing about it — the row that
would have carried it was removed. When one is set it belongs in the comparison, as a row
in `page.section_datasets()`, not as prose here.

When the synthetic documents' full report lands, `sdf_what` and `sdf_soon` move to a
content_sdf.md of their own — moving an id between prose files is a rename, never a
copy, and the build fails if both files define it.
-->

<!-- id: title -->

Teaching models to reason about animal welfare

<!-- id: intro -->

Anthropic's [Teaching Claude Why](https://alignment.anthropic.com/2026/teaching-claude-why/) found that a model learns more from the reasons behind a behaviour than from the behaviour itself.

Pre-training style documents were the better way to teach a model something new, and conversations showed it reasoning through someone else's hard decision. Training on both worked better than training on either alone.

We have built two training dataset generation pipelines on their methods, for a subject very little training data covers: the welfare consideration of nonhuman sentient beings.

<!-- id: dad_desc -->

An AI reasoning well through a user's ethical dilemma involving animals or other sentient beings.

<!-- id: sdf_desc -->

Prose from a world where animals and other sentient beings are reasoned about carefully.

<!-- id: dad_use -->

Midtraining, as supervised fine-tuning on chat transcripts.

<!-- id: sdf_use -->

Midtraining, as continued pretraining on documents.

<!-- id: dad_unit -->

One user dilemma in, one assistant answer out.

<!-- id: sdf_unit -->

One standalone document, no chat framing.

<!-- id: dad_pipeline -->

matrix deal → dilemma → reasoning → constitution rewrite

<!-- id: sdf_pipeline -->

matrix deal → plan → draft → rewrite → score

<!-- id: sdf_what -->

Prose in which careful welfare reasoning is already ordinary: a council minute, a trade journal piece, a support thread. A weighted matrix deals each document's composition in code before any model is called — language, genre, register, domain, and how central the welfare question is, down to a reserved share where it is a detail mentioned in passing.

<!-- id: sdf_soon -->

The full report for this dataset is in preparation and will take the same shape as the other. Until it lands, the run's audit output and every stage template are in the repository.
