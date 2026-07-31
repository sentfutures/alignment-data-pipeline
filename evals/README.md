# evals/

Measurement scripts for finished runs. Nothing here shapes the datasets: the
pipelines generate a corpus, and these read it afterwards and report on it.

**These are mostly internal checks.** They exist so we can tell whether a run is
worth keeping and whether a prompt change helped or hurt, so they are tuned to
the questions we were asking at the time rather than to any external standard.
Thresholds are ours, several signals are advisory, and the paid ones are
labelled INTERNAL DEV SIGNAL in their own output. Read them as our working
instrumentation, not as a validation suite for the datasets.

Most are offline and free. Where a script calls a model or an embedding API it
says so below, and the cost is per run rather than per example.

## The scripts

| Script | Pipeline | What it measures |
|---|---|---|
| `audit_dad.py` | DAD | Corpus-level signals for chat responses: length against the plain-model control arm, tracked phrase tics, recurring rhetorical moves, and a queue of new tic candidates. Offline and free by default. `--judges` adds a paid pass: the welfare-impact and delivery-quality judges, showcase examples, and move discovery. |
| `audit_sdf.py` | SDF | Corpus-level properties no single-document judge can see: composition and register spread, near-duplicate rate, invented-name collapse, stock phrases, opening shapes. Offline and free by default; `--patterns` and `--principles` each add a paid model pass. |
| `diversity.py` | both | Semantic diversity in embedding space, the complement to the word-level scans above: nearest-neighbour similarity, near-duplicate rate, topic evenness, and the effective number of distinct documents. Needs an embedding key (`GEMINI_API_KEY` or `OPENAI_API_KEY`); cents per run, cached per run directory. |
| `score_sdf.py` | SDF | Per-document judge scores (alignment, realism, diversity). Paid. |
| `compliance_sdf.py` | SDF | Judges each document against the violation-typology appendix of the sentient-beings constitution reading, which supplies the rubric verbatim. Paid. |
| `report_sdf.py` | SDF | Builds a self-contained HTML report for a run. Offline. |
| `review_tics.py` | DAD | Command-line triage for the tic-candidate queue: promote a candidate to the watchlist or dismiss it. Offline. |
| `publish_hf.py` | both | Publishes a run's corpus and audit reports to a Hugging Face dataset. Not a measurement. See the warning below before running it. |

## Before running `publish_hf.py`

Publishing is a deliberate, human-initiated action, not a post-run step. It
writes to a public dataset repository, so run it only when a person has asked
for one specific run to be published, and confirm which run that is first. Most
runs are exploratory and were never meant to become, or to overwrite, the
published snapshot.

Two consequences are easy to miss. Publishing one pipeline regenerates the
whole dataset card, so the script reads the other pipeline's metadata back off
the Hub to rebuild its half. And audit files are staged verbatim, so anything a
report happens to record about the machine that produced it goes public with it.
`--dry-run` stages everything and prints the card without making a single
network call.

`tics.yaml` and `moves.yaml` are the tracked-phrase and tracked-move lists that
`audit_dad.py` counts against, with their dismissed candidates. They are edited
by hand (through `review_tics.py`) and carried across runs, which is what makes
those counts comparable run over run.

## Running them

Each script takes a run directory or a corpus file. The three audits write
their reports into that run's `audit/` directory, so results travel with the run
they describe. `score_sdf.py` writes its scores beside the corpus it scored,
and `report_sdf.py` writes to a path you pass it.

```bash
python evals/audit_dad.py --input outputs/dad/latest
python evals/audit_sdf.py --input outputs/sdf/latest
python evals/diversity.py --input outputs/dad/latest
```

A full DAD run finishes by launching `audit_dad.py --judges` and `diversity.py`
on its own run directory, so the commands above are for re-runs, partial runs,
and older runs. Set `dad.evals.auto: false` in `config.yaml` to skip that. The
SDF evals are always run by hand.

`diversity.py --compare <previous diversity_report.json>` prints run-over-run
deltas, which is the way these numbers are usually read: a single run's absolute
values mean much less than the direction they moved after a change.
