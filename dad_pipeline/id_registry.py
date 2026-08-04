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
Additive — the per-run ids are untouched.

The registry is a git-tracked JSON file shared across runs (one id space); in
tests it lives under the tmp output root, so it never touches the real one.

Two properties the callers depend on:

- **Not safe to run concurrently.** Every run under one `outputs/dad` tree opens
  the *same* registry file, allocates from its own in-memory copy, and persists
  by full overwrite. Two DAD runs at once therefore hand the same number to
  different content, and the last save wins — silently. Run them one at a time.
  (No `fcntl` lock or per-run allocation journal: deliberately out of scope.)
- **Ordering invariant: save before you write.** A stage must call `save()`
  *before* appending any record that carries a freshly stamped gid. A crash may
  then leave a number allocated but unused — a harmless gap — but never a number
  that is already on a written record yet unallocated, which is what makes the
  next run hand that same number to different content.
"""

import hashlib
import json
import os
import stat
import sys
import tempfile
from collections import Counter
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
    records, falling back to the per-run prompt_id (AW-####) that older runs
    carry, so stages, audits, and the viewer keep working on those run dirs."""
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
        # A registry that does not exist yet is the normal first-run case: start
        # empty. A registry that exists but cannot be read as one is NOT — see
        # _die.
        if not self.path.exists():
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
        except OSError as exc:
            self._die(f"it could not be read ({exc})")
        try:
            loaded = json.loads(raw)
        except json.JSONDecodeError as exc:
            self._die(f"it is not valid JSON ({exc})", raw=raw)
        if not isinstance(loaded, dict):
            self._die(f"its top level is {type(loaded).__name__}, not an object", raw=raw)
        for kind, table in loaded.items():
            if not isinstance(table, dict):
                self._die(f"its {kind!r} table is {type(table).__name__}, not an object",
                          raw=raw)
            try:
                self._data[kind] = {str(k): int(v) for k, v in table.items()}
            except (TypeError, ValueError) as exc:
                self._die(f"its {kind!r} table holds a value that is not a number ({exc})",
                          raw=raw)
        self._warn_on_duplicate_numbers()

    def _die(self, why: str, raw: str = "") -> None:
        """Refuse to run on a registry we cannot read. Starting over would
        re-issue numbers that already name different artifacts in committed
        runs, audits, and the viewer — a silent, unrecoverable collision."""
        msg = [
            f"Cannot read the id registry at {self.path}: {why}.",
            "This file maps content fingerprints to the stable gids "
            "(S-/P-/R-/C-/E-####) that committed runs, audit reports, and the "
            "viewer are keyed by, so starting over would re-issue numbers that "
            "already name different artifacts.",
        ]
        if any(m in raw for m in ("<<<<<<<", "=======", ">>>>>>>")):
            msg.append(
                "It looks like an unresolved merge conflict: resolve it by "
                "keeping BOTH sides' entries (each fingerprint keeps the number "
                "it already has), not by picking one side.")
        msg.append(f"Otherwise restore it from git (`git checkout -- {self.path}`). "
                   "A registry that does not exist yet is fine — that starts empty.")
        raise SystemExit(" ".join(msg))

    def _warn_on_duplicate_numbers(self) -> None:
        """Two branches allocating from the same max independently can both mint
        the same number, so a merged registry may map two fingerprints to one
        gid. Warn loudly — but don't die: the only remedy is renumbering, which
        would move gids already stamped on committed runs. New ids still
        allocate above the maximum, so the damage never spreads."""
        for kind, table in sorted(self._data.items()):
            dupes = sorted(n for n, c in Counter(table.values()).items() if c > 1)
            if dupes:
                print(
                    f"  WARNING: {self.path} maps two different {kind} fingerprints "
                    f"onto the same number(s): {dupes}. A merge of two branches' "
                    "registries does this, and those artifacts now share a gid. "
                    "New ids still allocate above the maximum, so it does not "
                    "spread; renumbering would move gids already committed.",
                    file=sys.stderr,
                )

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
        """Persist atomically: a same-directory temp file, fsynced, then
        os.replace()d onto the target. An interrupted save leaves the previous
        registry intact rather than a truncated one the next run cannot read."""
        # Serialize first: a serialization error then cannot leave debris behind.
        payload = json.dumps(self._data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # mkstemp creates 0600 files; match the existing target's mode so an
        # atomic save never silently tightens permissions on a git-tracked file.
        try:
            mode = stat.S_IMODE(self.path.stat().st_mode)
        except OSError:
            mode = 0o644
        fd, tmp = tempfile.mkstemp(dir=self.path.parent,
                                   prefix=self.path.name + ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
                os.fchmod(f.fileno(), mode)
            os.replace(tmp, self.path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
