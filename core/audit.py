"""Hash-chained, append-only audit log (AGENTS.md §2.2, §8.5).

CPU only, no GPU implication. One append is a single line write with no fsync:
sub-millisecond, so it is safe to call on every model call and tool call without
distorting the latency numbers we quote in §13.

Chain rule (deviates from §8.5 as literally written, by decision):
    hash = sha256(canonical_json(record minus `hash`))
The spec's `sha256(prev_hash + canonical_json(payload))` leaves `seq`, `ts` and
`kind` outside the digest, so those fields could be rewritten without breaking
the chain. `prev_hash` is a field of the record, so hashing the whole record
still links each entry to its predecessor.

Intent/result rule: §2.2 wants a record before the action and the outcome after,
but §8.5 makes the log append-only — mutating a record would break the chain.
So an action emits two records: an intent record, then a result record whose
payload carries `ref: <intent seq>`.
"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

Kind = Literal[
    "model_call",
    "tool_call",
    "file_read",
    "file_write",
    "egress_attempt",
    "approval",
    "error",
]

GENESIS = "0" * 64


class ConcurrentAudit(RuntimeError):
    """Another process appended to this log. §2.2 makes this fatal, not a warning."""
DEFAULT_PATH = Path("workspace/.audit/audit.jsonl")


class AuditRecord(BaseModel):
    seq: int
    ts: str
    session_id: str
    kind: Kind
    payload: dict[str, Any]
    prev_hash: str
    hash: str = ""

    def digest(self) -> str:
        """sha256 over every field except `hash` itself."""
        body = self.model_dump(exclude={"hash"})
        canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class AuditLog:
    """Append-only JSONL chain. Resumes from an existing file on construction.

    One `AuditLog` instance is shared by every tool call in a run, and a tool
    runs inside `asyncio.to_thread` (core/agent.py) — a real OS thread, not
    just another coroutine. The operator kill switch (core/api/agent.py) can
    call `append()` from the main thread at the same instant a tool's worker
    thread is mid-`append()` of its own, and without a lock both read the
    same `_seq`/`_prev` before either had written, producing two records
    claiming the same seq — measured, not hypothetical: it broke the chain
    the first time a kill landed while a model call was in flight.
    """

    def __init__(self, path: Path = DEFAULT_PATH, session_id: str = "adhoc") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id
        self._seq, self._prev = self._resume()
        self._size = self.path.stat().st_size if self.path.exists() else 0
        self._lock = threading.Lock()

    def _resume(self) -> tuple[int, str]:
        last: str | None = None
        if self.path.exists():
            with self.path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        last = line
        if last is None:
            return 0, GENESIS
        rec = AuditRecord.model_validate_json(last)
        return rec.seq + 1, rec.hash

    def append(
        self, kind: Kind, payload: dict[str, Any], ref: int | None = None
    ) -> AuditRecord:
        """Write one record. `ref` links a result record back to its intent record.

        The whole method is one critical section: reading `_seq`/`_prev`,
        writing the line, and advancing that state must happen as a unit, or
        two threads can each read the pre-write state and both claim the same
        seq (see the class docstring). `threading.Lock`, not `asyncio.Lock` —
        the callers this actually races are OS threads (`asyncio.to_thread`
        workers), which an asyncio-level lock does not serialize at all.
        """
        with self._lock:
            body = dict(payload) if ref is None else {**payload, "ref": ref}
            rec = AuditRecord(
                seq=self._seq,
                ts=datetime.now(timezone.utc).isoformat(),
                session_id=self.session_id,
                kind=kind,
                payload=body,
                prev_hash=self._prev,
            )
            rec.hash = rec.digest()
            self._assert_sole_writer()
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(rec.model_dump_json() + "\n")
            self._seq += 1
            self._prev = rec.hash
            self._size = self.path.stat().st_size
            return rec

    def _assert_sole_writer(self) -> None:
        """Refuse to append if the file grew behind our back.

        `seq` and `prev_hash` are held in memory from construction, so two live
        processes on one file both resume from the same point and both write the
        same seq — which silently breaks the chain rather than failing. That is
        not hypothetical: starting a second orchestrator while the first held
        port 8080 produced a duplicate seq and a BROKEN verify. Comparing size is
        O(1) and catches exactly that, loudly, before the bad record is written.
        """
        actual = self.path.stat().st_size if self.path.exists() else 0
        if actual != self._size:
            raise ConcurrentAudit(
                f"{self.path} changed underneath this process "
                f"({self._size} -> {actual} bytes). Another writer — most likely a "
                f"second app instance — is appending to the same audit log. Stop it "
                f"before continuing; two writers corrupt the hash chain."
            )

    def tail(self, n: int = 50) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        # ponytail: reads the whole file; fine to ~100k records. Reverse-seek if
        # the log outgrows RAM, which it will not during a 6-minute demo.
        with self.path.open(encoding="utf-8") as fh:
            lines = [ln for ln in fh if ln.strip()]
        return [AuditRecord.model_validate_json(ln) for ln in lines[-n:]]

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain. Returns (ok, seq of first broken record)."""
        if not self.path.exists():
            return True, None
        prev = GENESIS
        expected = 0
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                try:
                    rec = AuditRecord.model_validate_json(line)
                except ValidationError:
                    return False, expected
                if (
                    rec.seq != expected
                    or rec.prev_hash != prev
                    or rec.hash != rec.digest()
                ):
                    return False, rec.seq
                prev = rec.hash
                expected += 1
        return True, None


def main() -> int:
    """`audit verify [path]` — walks the chain and reports the first break."""
    if len(sys.argv) < 2 or sys.argv[1] != "verify":
        print("usage: audit verify [path]", file=sys.stderr)
        return 2
    path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_PATH
    # verify() answers "is this chain broken", and a log that does not exist is
    # not broken — which is the right answer for the API (a fresh install has
    # nothing to verify) and the wrong one here. This CLI is the §10.5 pre-demo
    # assertion: a typo'd path printing "chain intact" asserts nothing while
    # looking like proof. Missing must fail loudly.
    if not path.exists():
        print(f"FAIL no audit log at {path} — nothing was verified", file=sys.stderr)
        return 1
    ok, broken = AuditLog(path).verify()
    if ok:
        print(f"OK  chain intact: {path}")
        return 0
    print(f"FAIL chain broken at seq {broken}: {path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
