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
*is*, in one sentence. `dad_unit` / `sdf_unit` are the `result format` — what one record
is. `dad_use` / `sdf_use` are what each is *for*: both are midtraining, and the difference
is the format they are consumed in. `dad_pipeline` / `sdf_pipeline` are the stages that
produce it, as a chain, in the same shape on both sides so the two columns can be read
across. One short line each; a row's LABEL lives in `page.section_datasets()`, only its
cells are here. The `*_pipeline` chains carry a NON-BREAKING space after each arrow, so a
wrap puts the arrow at the head of the next line rather than orphaning it at the end of
the last one — it is invisible in an editor, so copy an existing arrow rather than typing
a new one.

No licence is set for either dataset, and the page says nothing about it — the row that
would have carried it was removed. When one is set it belongs in the comparison, as a row
in `page.section_datasets()`, not as prose here.

Each report's own prose lives in its own file — `content_dad.md` and `content_sdf.md`.
Moving an id between prose files is a rename, never a copy: the build fails if both files
define one.
-->

<!-- id: title -->

Teaching models to reason about harm to animals

<!-- id: intro -->

Research on alignment midtraining ([1](https://alignment.anthropic.com/2026/teaching-claude-why/), [2](https://www.lesswrong.com/posts/GTYJRLhqztxKF2v5R/synthetic-document-finetuning-for-instilling-positive-traits)) finds that teaching AI models the reasons behind aligned behaviors is just as important as the behaviors themselves.

Two complementary techniques proved especially effective:

1. **Synthetic document finetuning** using pretraining-style documents from a world where the target model is *already* aligned to a wide variety of aligned behaviors. This reinforces the existence of an aligned persona for the model in training while increasing the number of aligned propensities associated with it.
2. **Difficult advice Q&A** depicting an AI assistant coaching users through ethical dilemmas analogous to those the target model might eventually encounter in deployment. This teaches ethical reasoning skills while teaching the model to identify with the responsible persona.

Following this research, we built pipelines for synthesizing training data on a subject currently absent from the training corpus: welfare considerations of nonhuman sentient beings. We developed scenarios and reasoning principles in consultation with leading animal ethicists to create a robust training set that mirrors situations where real AI systems will take actions beneficial or detrimental to animal welfare.

<!-- id: dad_desc -->

AI coaching users through ethical dilemmas involving disenfranchised third parties (e.g. animals).

<!-- id: sdf_desc -->

Diverse artifacts from a world where your model already reasons responsibly about animal welfare.

<!-- id: dad_use -->

Supervised fine-tuning QA

<!-- id: sdf_use -->

Midtraining

<!-- id: dad_unit -->

One user dilemma in, one assistant answer out.

<!-- id: sdf_unit -->

Blogs, interviews, encyclopedia entries, forum threads.

