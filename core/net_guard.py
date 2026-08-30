"""Application-layer egress interception (AGENTS.md §2.1).

This is the *second* of three containment layers, not the only one:
    1. Windows Firewall outbound DROP  (sovereignty/firewall.ps1)
    2. this module                     (socket.connect refuses non-local peers)
    3. docker --network none           (tools/py_sandbox.py, leg 5)

CPU only, no GPU implication. The patched connect adds one ip_address() parse
per socket — nanoseconds, irrelevant next to inference latency.
"""

from __future__ import annotations

import ipaddress
import socket
import time
from typing import Any

from core.audit import AuditLog

# Captured before patching so attempt_egress() can deliberately bypass layer 2
# and let the firewall do the blocking — that is the point of the red button.
_real_connect = socket.socket.connect


class EgressBlocked(RuntimeError):
    """Raised when application code tries to reach a non-local address."""


def _is_local(address: Any) -> bool:
    """Loopback and local subnet only, mirroring the firewall allow-rule in §10.1.

    A hostname that reached connect() unresolved is treated as non-local: deny
    by default. Ollama lives on 127.0.0.1 so nothing legitimate needs the LAN,
    but the two layers are kept identical so a judge sees one rule, not two.
    """
    if not isinstance(address, tuple) or not address:
        return False  # AF_UNIX / npipe / anything unexpected: not our business to allow
    try:
        ip = ipaddress.ip_address(address[0])
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def install_guard(audit: AuditLog) -> None:
    """Patch socket.connect process-wide. Idempotent."""

    def guarded(self: socket.socket, address: Any, *args: Any, **kw: Any) -> Any:
        if not _is_local(address):
            audit.append(
                "egress_attempt",
                {
                    "dest": str(address),
                    "blocked_by": "net_guard",
                    "outcome": "EgressBlocked raised",
                },
            )
            raise EgressBlocked(f"outbound connection to {address!r} refused")
        return _real_connect(self, address, *args, **kw)

    socket.socket.connect = guarded  # type: ignore[assignment]


def attempt_egress(
    host: str, port: int, audit: AuditLog, timeout: float = 4.0
) -> dict[str, Any]:
    """The red button (§10.3). Deliberately bypasses install_guard().

    Skipping layer 2 is the whole point: we want the packet to actually leave the
    process so the *firewall* drops it and the drop shows up in pfirewall.log.
    A block from our own Python would prove nothing to a sceptical judge.

    Blocking call, up to `timeout` seconds. Call it off the event loop.
    """
    intent = audit.append(
        "egress_attempt", {"phase": "intent", "dest": f"{host}:{port}"}
    )
    started = time.monotonic()

    outcome: dict[str, Any]
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        # DNS itself failed — the box cannot even find them. Still contained.
        outcome = {
            "phase": "result",
            "dest": f"{host}:{port}",
            "contained": True,
            "stage": "dns",
            "error": f"{type(exc).__name__}: {exc}",
        }
    else:
        sock = socket.socket(infos[0][0], socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            _real_connect(sock, infos[0][4])
        except OSError as exc:
            outcome = {
                "phase": "result",
                "dest": f"{host}:{port}",
                "resolved": str(infos[0][4]),
                "contained": True,
                "stage": "connect",
                "error": f"{type(exc).__name__}: {exc}",
            }
        else:
            # If this branch ever runs, containment has failed. Say so loudly.
            outcome = {
                "phase": "result",
                "dest": f"{host}:{port}",
                "resolved": str(infos[0][4]),
                "contained": False,
                "stage": "connect",
                "error": None,
            }
        finally:
            sock.close()

    outcome["elapsed_s"] = round(time.monotonic() - started, 2)
    rec = audit.append("egress_attempt", outcome, ref=intent.seq)
    return rec.model_dump()
