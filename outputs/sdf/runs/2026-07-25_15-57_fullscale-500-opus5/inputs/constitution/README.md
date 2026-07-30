# Constitution documents

Three files, each with a distinct role. `shared/constitution_loader.py` is the single loader for all of them, and every pipeline run snapshots this whole directory into `runs/<run_id>/inputs/constitution/`, so a run stays reproducible even after these files change.

## `constitution_claude.md`

The original Claude constitution, verbatim. Never edited here.

## `constitution_sentient_beings.md`

A section-by-section *reading* of the constitution for animal-welfare implications — it quotes the constitution and explains what each section already implies for the treatment of animals and other sentient beings; it does not modify or extend it. One `## ` header per section.

The two markdown files are joined **in memory** (`load_full_constitution()`) wherever the full text is needed — the system prompt at SDF layers 4–5. They are never combined on disk.

## `constitution_principles.csv`

The distilled welfare-relevant principles, one row each:

- `number`, `principle` — a stable number and a one-line name.
- `welfare_application` — what the constitution commits a response to, in plain language.
- `constitution_excerpts` — the **verbatim constitution quotes** the principle distills, kept so the principles can always be traced back to (and referenced against) the constitution's own words.

This distillation was produced *from* the two documents above, with the full constitution in context; the pipeline then uses the distillation instead of the full text. `load_principles()` / `format_principles()` render each principle (name → welfare application → constitution excerpts) as the `CONSTITUTION PRINCIPLES` block in the DAD step-3 rewrite prompt. The full constitution itself is not sent at DAD generation time.
