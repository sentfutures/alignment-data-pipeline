"""Load and format the animal-ethics reasoning library.

Source of truth is prompts/dad/reasoning_library.csv. Rows are entries with
columns: id, category, claim, reasoning, trigger_condition,
transferable_move. Every entry — conduct (C*), core move (M*), and topic (T*)
alike — is conditional: a dedicated selection call after 2a scoping
(step2_select.txt) reads the trigger index (trigger_index_block) and flags
which entries fire for the case, and 2b injects only the flagged rows
(falling open to the whole library when the selection is unusable). Which
rows were injected is recorded per prompt in step2/scopes.jsonl and on each
response record's entry_ids.
"""

import csv
import io
from pathlib import Path

CSV_FILENAME = "reasoning_library.csv"

CONDUCT_CATEGORY = "Conduct"
CORE_MOVE_CATEGORY = "Core move"


def resolve_path(prompts_dir: str | Path) -> Path:
    """The library CSV in prompts_dir (may not exist; caller surfaces the
    error)."""
    return Path(prompts_dir) / CSV_FILENAME


def parse_text(text: str) -> dict:
    """Parse the library CSV into {"entries": [row, ...]}."""
    return {"entries": list(csv.DictReader(io.StringIO(text)))}


def load(prompts_dir: str | Path) -> dict:
    path = resolve_path(prompts_dir)
    return parse_text(path.read_text(encoding="utf-8"))


def _entries(library: dict) -> list[dict]:
    return library.get("entries") or []


def conduct_ids(library: dict) -> list[str]:
    """The C* conduct entries."""
    return [e["id"] for e in _entries(library)
            if e.get("category") == CONDUCT_CATEGORY]


def core_move_ids(library: dict) -> list[str]:
    """The M* core-move entries."""
    return [e["id"] for e in _entries(library)
            if e.get("category") == CORE_MOVE_CATEGORY]


def all_ids(library: dict) -> list[str]:
    return [e["id"] for e in _entries(library)]


def get_entries(library: dict, ids: list[str]) -> list[dict]:
    """The full rows for ids, in the order given; unknown ids are dropped."""
    by_id = {e["id"]: e for e in _entries(library)}
    return [by_id[i] for i in ids if i in by_id]


def trigger_index_block(library: dict) -> str:
    """One line per entry — id plus trigger condition — the lightweight index
    the 2a.5 select prompt evaluates instead of loading the whole library."""
    return "\n".join(f"- {e['id']}: {e.get('trigger_condition', '')}"
                     for e in _entries(library))


def format_library(library: dict) -> str:
    """The whole library formatted for the response prompt (all entries, in file
    order: conduct, core moves, then topic reasoning)."""
    return format_entries(library, all_ids(library))


def format_entries(library: dict, ids: list[str]) -> str:
    by_id = {e["id"]: e for e in _entries(library)}
    blocks = []
    for eid in ids:
        e = by_id.get(eid)
        if not e:
            continue
        # .get(): render what a row has rather than crashing the viewer's
        # lineage page on a hand-edited CSV with a missing field.
        blocks.append(
            f"[{e['id']}] {e.get('claim', '')}\n"
            f"Reasoning: {e.get('reasoning', '')}\n"
            f"Trigger condition: {e.get('trigger_condition', '')}\n"
            f"Transferable move: {e.get('transferable_move', '')}"
        )
    return "\n\n".join(blocks)


def system_prompt(library: dict) -> str:
    """LEGACY — the pipeline no longer sends this; the 2b template is
    self-contained. Kept for the viewer, which reconstructs the system prompt
    of runs recorded before that: the conduct rules (C*)."""
    return (
        "STANDING CONDUCT RULES (apply to every response, whether or not the user "
        "mentions animals):\n\n" + format_entries(library, conduct_ids(library))
    )
