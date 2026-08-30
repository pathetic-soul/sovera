"""Windows Firewall drop-log watcher (AGENTS.md §10.2).

§10 specifies nftables drop-chain counters. There is no Linux userland on the
demo machine (only Docker's WSL utility VM), so the enforcement layer is Windows
Firewall, which exposes no counter API. We tail its drop log instead — which is
strictly better for the demo: the panel shows the destination IP and port of the
packet that died, not just an integer going up.

Enable the log with `sovereignty/firewall.ps1 -Enable` (needs admin).

CPU only. Each poll reads just the bytes appended since the last poll.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

DEFAULT_LOG = Path(
    os.environ.get(
        "PFIREWALL_LOG",
        os.path.expandvars(r"%windir%\system32\LogFiles\Firewall\pfirewall.log"),
    )
)

# W3C header: date time action protocol src-ip dst-ip src-port dst-port size ...
# followed by tcp flags etc, then `path` (SEND / RECEIVE) as the last field.
_DATE, _TIME, _ACTION, _PROTO, _SRC, _DST, _SPORT, _DPORT = range(8)


class DropWatcher:
    """Incremental reader over pfirewall.log. Not thread-safe; one per app."""

    def __init__(self, path: Path = DEFAULT_LOG) -> None:
        self.path = Path(path)
        self.total = 0
        self._offset = 0
        self._seed()

    def _seed(self) -> None:
        """Start at end-of-file so the demo counts only drops we cause."""
        if self.path.exists():
            self._offset = self.path.stat().st_size

    def available(self) -> tuple[bool, str]:
        if not self.path.exists():
            return False, f"log not found at {self.path} - run firewall.ps1 -Enable"
        if not os.access(self.path, os.R_OK):
            return False, f"{self.path} unreadable — run the app as Administrator"
        return True, "ok"

    def poll(self) -> list[dict[str, Any]]:
        """Return outbound drops appended since the previous poll."""
        ok, _ = self.available()
        if not ok:
            return []
        size = self.path.stat().st_size
        if size < self._offset:  # log rotated or cleared
            self._offset = 0
        if size == self._offset:
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(self._offset)
            chunk = fh.read()
            self._offset = fh.tell()

        drops: list[dict[str, Any]] = []
        for line in chunk.splitlines():
            if not line or line.startswith("#"):
                continue
            f = line.split()
            if len(f) < 8 or f[_ACTION] != "DROP":
                continue
            if f[-1] == "RECEIVE":  # inbound drops are noise; we claim egress
                continue
            drops.append(
                {
                    "ts": f"{f[_DATE]} {f[_TIME]}",
                    "proto": f[_PROTO],
                    "dst": f[_DST],
                    "dport": f[_DPORT],
                }
            )
        self.total += len(drops)
        return drops


if __name__ == "__main__":
    pass
