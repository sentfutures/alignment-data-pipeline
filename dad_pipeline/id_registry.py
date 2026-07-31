"""Stable, content-keyed global ids for the things the pipeline produces.

The per-run ids — scenario_id (S-###) and the response_id / record_id uuids —
reset every run (or are unreadable), so they can't identify "the same thing"
across runs. This registry provides the *stable* ids, one kind per artifact,
each counting up globally and reused whenever the same content appears again:

- scenario_gid  S-####  the dealt scenario (categorical shape, pre-prompt)
- prompt_gid    P-####  the shipped user message (exact wording)
- response_gid  R-####  one pipeline response (step-2 draft text; the id
                        survives the step-3 rewrite — same response, revised)
- plain_gid     C-####  one plain-Claude control response (baseline text)
- example_gid   E-####  one finished training example (user message + final
                        rewritten response pair)

That lets the viewer and the audits align/sort the same artifact across runs.
Additive — the per-run ids are untouched. (A one-time `backfill_gids.py`
labeled the pre-registry historical runs; it was removed 2026-07-30 along
with those runs.)

The registry is a git-tracked JSON file shared across runs (one id space); in
tests it lives under the tmp output root, so it never touches the real one.
"""

import hashlib
import json
from pathlib import Path


def _fingerprint(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()[:16]


def scenario_fingerprint(scenario: dict) -> str:
    """Hash of a scenario's categorical shape — everything but its own ids."""
    return _fingerprint({k: v for k, v in scenario.items()
                         if k not in ("scenario_id", "scenario_gid")})


def prompt_fingerprint(user_message: str) -> str:
    """Hash of the whitespace-normalized user message."""
    return _fingerprint(" ".join((user_message or "").split()))


def response_fingerprint(text: str) -> str:
    """Hash of a whitespace-normalized response text (pipeline draft or plain
    baseline — the kind keeps their id spaces separate)."""
    return _fingerprint(" ".join((text or "").split()))


def example_fingerprint(user_message: str, assistant_message: str) -> str:
    """Hash of a whitespace-normalized (user, assistant) training pair."""
    return _fingerprint([" ".join((user_message or "").split()),
                         " ".join((assistant_message or "").split())])


def prompt_key(rec: dict) -> str:
    """The id that names a record's prompt: prompt_gid (P-####) on current
    records, falling back to the retired per-run prompt_id (AW-####) so
    stages, audits, and the viewer keep working on legacy run dirs."""
    return rec.get("prompt_gid") or rec.get("prompt_id") or ""


def prompt_keys(rec: dict) -> tuple[str, ...]:
    """Every id naming this record's prompt (prompt_gid, legacy prompt_id).
    Lookup tables register a record under all of them, so mixed-era runs —
    gid-era step-1 files joined by pre-gid later stages that carry only
    prompt_id — still join."""
    return tuple(str(v) for v in (rec.get("prompt_gid"), rec.get("prompt_id")) if v)


def registry_path(output_dir: Path) -> Path:
    """The registry lives at the dad-pipeline output root
    (<outputs>/dad/id_registry.json), found by walking up to the `runs` dir.
    Falls back to the output dir itself for non-standard layouts (e.g. tests
    passing a bare tmp stage dir), which keeps each test isolated."""
    for anc in Path(output_dir).parents:
        if anc.name == "runs":
            return anc.parent / "id_registry.json"
    return Path(output_dir) / "id_registry.json"


class IdRegistry:
    """Maps a content fingerprint to a stable integer per kind. New content
    gets max+1; seen content keeps its number; numbers never reset across
    runs. Persisted as JSON."""

    KINDS = ("scenario", "prompt", "response", "plain", "example")
    PREFIXES = {"scenario": "S", "prompt": "P", "response": "R",
                "plain": "C", "example": "E"}

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._data: dict[str, dict[str, int]] = {k: {} for k in self.KINDS}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                loaded = {}
            if isinstance(loaded, dict):
                for kind in self.KINDS:
                    table = loaded.get(kind)
                    if isinstance(table, dict):
                        self._data[kind] = {str(k): int(v) for k, v in table.items()}

    def assign(self, kind: str, fingerprint: str) -> int:
        """Return the stable number for this content, allocating the next one
        (global max + 1) the first time it's seen."""
        table = self._data.setdefault(kind, {})
        if fingerprint not in table:
            table[fingerprint] = max(table.values(), default=0) + 1
        return table[fingerprint]

    def gid(self, kind: str, fingerprint: str) -> str:
        """The formatted stable id for this content, e.g. 'R-0012'."""
        return f"{self.PREFIXES[kind]}-{self.assign(kind, fingerprint):04d}"

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
