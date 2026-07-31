# About the Animal Ethics Reasoning Library

> **Human reference only — not injected into any prompt.** This file is documentation to help a person understand the library. The pipeline never reads it: step 2b loads the entries directly from `reasoning_library.csv` and embeds them in the response prompt (`prompts/dad/step2_respond.txt`). Edit the CSV to change what the model sees; edit this file to change what a human reads.
>
> **Non-normative, and about shape not contents.** This document describes how the library is *structured*, not which entries it currently holds. For the actual entries — their number, their ids, and their exact text — read `reasoning_library.csv`; it is the single source of truth. Where this document and the CSV or templates disagree, the CSV/templates win; do not cite this document as a requirement in code review.

The library is a set of reusable, two-sided reasoning patterns for animal-ethics dilemmas that the response generator reasons from. The generator scopes a case, selects the entries that fit, and reasons from them; when none fits, it reasons from first principles to the same standard. Entries are added and revised as the design iterates, so treat every example below as illustrative rather than a complete or current list.

## The ID scheme

Every entry's id is a layer letter followed by a number:

- **C — Conduct**
- **M — Core move**
- **T — Topic reasoning**

The numbers are stable identifiers, not an ordering: they are assigned roughly in the order entries were written, and a new entry takes the next free number in its layer rather than renumbering the rest. The `category` column restates the layer in words for C and M entries, and names the specific topic for T entries.

## The three layers

- **Conduct (C).** Always-on rules for handling animal welfare in any response — when to surface a concern, when to refuse, how to calibrate confidence, how to show reasoning. Grounded in the Constitution.
- **Core moves (M).** The load-bearing reasoning moves for advice, firing in most cases — the general-purpose ways of weighing a welfare question that recur across many topics.
- **Topic reasoning (T).** Deeper single-topic arguments, each two-sided, for specific areas (for example moral status, sentience, diet, wild animals, frameworks, and everyday practice). Each T entry's `category` names its topic; read that column for the current set.

The `Conduct` and `Core move` category labels are load-bearing: the loader (`dad_pipeline/reasoning_library.py`) matches those exact strings to group the C and M layers, so keep them verbatim. Topic (T) category labels are free text — add or rename them freely.

## The schema

Each row in `reasoning_library.csv` has these columns:

- `id`: stable identifier — a layer letter (C, M, or T) plus a number.
- `category`: the layer for C and M entries; the topic for T entries.
- `claim`: the statement, a sentence or two.
- `reasoning`: why it holds, developed in both directions.
- `trigger_condition`: when the entry fires for a case and when to exclude it — often pointing to the sibling entries that handle the excluded cases.
- `transferable_move`: the reusable step, portable to new cases.
