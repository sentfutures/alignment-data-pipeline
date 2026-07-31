# code_quality/

Pre-delivery code-quality audit of this repo, produced 2026-07-29 (v2, at `main@464c5bd`),
superseding a 2026-07-10 audit (v1, at `cf8d91e`).

| File | What it is |
|---|---|
| `CODE_QUALITY_REPORT.md` | The report. Written to be self-contained for readers without codebase context: background/glossary, executive summary, the fate of every v1 finding, and prioritized findings (Part 2 fix-before-delivery, Part 3 structural, Part 4 process/docs). |
| `findings_v2_2026-07-29.json` | Machine-readable ledger behind the v2 report: all 50 findings with per-finding `evidence`, `failure_scenario`, `recommendation`, `instances`, severity/effort, and `verifier_notes` (what the independent adversarial verifier checked to confirm the claim), grouped by review dimension with a `summary` per dimension. |
| `findings_v1_2026-07-10.json` | Same format for the superseded July audit. Historical: its line numbers describe the pre-rebuild code, and the v2 report's Part 1 records each finding's current status. |

## For agents filing or resolving issues from this audit

- Work from the **v2 report's numbering** (findings 2.x / 3.x / 4.x); use the v2 JSON to pull
  the full `evidence` (exact file:line as of `464c5bd`), the concrete `failure_scenario`
  (useful as the issue's motivation), and the full `recommendation` (often more detailed than
  the report's summary — includes suggested helper names, test recipes, and PR sequencing).
- `verifier_notes` record what the verification pass actually read; re-verify cited lines
  against current `main` before acting — the code moves fast and line numbers drift.
- Findings with severity `low`/`info` were **not** adversarially verified (treat as probable,
  not proven); confirm them against the code before filing.
- The v1 ledger is only for history/context (e.g. explaining why a pattern exists); do not
  file issues from it — its still-relevant items are carried forward in the v2 report.
