"""Read-only aiohttp WebSocket + REST server for the DOPE Fleet & Fidelity KPI Dashboard."""
from __future__ import annotations

import argparse
import asyncio
from collections import deque
import json
from pathlib import Path
import socket
import sys
import time
from urllib.parse import urlsplit

from aiohttp import web

from .metrics import fleet_summary, host_view

HERE = Path(__file__).resolve().parent
HTML_PATH = HERE / "static" / "dope-dashboard.html"
PROBE_PATH = HERE / "probe.py"
DEFAULT_HOSTS = tuple(f"xbabe{i}" for i in range(10))
SSH_CMD = (
    "ssh",
    "-T",
    "-o",
    "BatchMode=yes",
    "-o",
    "ConnectTimeout=5",
    "-o",
    "ServerAliveInterval=5",
    "-o",
    "ServerAliveCountMax=2",
)
FRAME_LIMIT = 2 << 20


def _wire(msg: dict) -> str:
    return json.dumps(msg, separators=(",", ":"), allow_nan=False)


class DopeFleetState:
    def __init__(self, hosts: tuple[str, ...]) -> None:
        self.readings: dict[str, tuple[dict | None, float | None, str]] = {h: (None, None, "") for h in hosts}
        self.hosts: dict[str, dict] = {}
        self.revisions: dict[str, int] = dict.fromkeys(hosts, 0)
        self.history: deque[dict] = deque(maxlen=240)
        self.waiters: set[asyncio.Event] = set()
        self.at = time.time()
        self.summary: dict = {}
        self.pulse()

    def feed(self, host: str, payload: dict) -> None:
        self.readings[host] = (payload, time.time(), "")

    def lost(self, host: str, reason: str) -> None:
        payload, seen, _ = self.readings[host]
        self.readings[host] = (payload, seen, reason)

    def pulse(self) -> None:
        self.at = now = time.time()
        for host, (reading, seen, detail) in self.readings.items():
            view = host_view(host, reading, seen, detail, now)
            if view != self.hosts.get(host):
                self.revisions[host] += 1
                self.hosts[host] = view
        self.summary = fleet_summary(list(self.hosts.values()), self.history, now)
        for ev in list(self.waiters):
            ev.set()

    def snapshot_dict(self) -> dict:
        return {"type": "snapshot", "at": self.at, "fleet": self.summary, "hosts": list(self.hosts.values())}


async def _reap(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        pass
    try:
        await asyncio.wait_for(proc.wait(), 2.0)
    except asyncio.TimeoutError:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()


async def collect_host(host: str, state: DopeFleetState, local_host: str) -> None:
    local = host == local_host
    cmd = [sys.executable, "-u", str(PROBE_PATH)] if local else [*SSH_CMD, host, "python3", "-u", "-"]
    source = b"" if local else PROBE_PATH.read_bytes()
    attempt = 0
    while True:
        proc = None
        last_seen = state.readings[host][1]
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE if source else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
                limit=FRAME_LIMIT,
            )
            if source and proc.stdin is not None:
                proc.stdin.write(source)
                await asyncio.wait_for(proc.stdin.drain(), 5.0)
                proc.stdin.close()
            assert proc.stdout is not None
            while True:
                line = await asyncio.wait_for(proc.stdout.readline(), timeout=20.0)
                if not line:
                    break
                try:
                    state.feed(host, json.loads(line))
                except (ValueError, TypeError):
                    continue
            code = await asyncio.wait_for(proc.wait(), 2.0)
            state.lost(host, f"probe exited ({code})")
        except asyncio.TimeoutError:
            state.lost(host, "probe timeout")
        except (OSError, ValueError):
            state.lost(host, "ssh unavailable")
        finally:
            if proc is not None:
                await _reap(proc)
        attempt = 0 if state.readings[host][1] != last_seen else min(5, attempt + 1)
        await asyncio.sleep(2 ** attempt)


def _allowed_origin(request: web.Request) -> bool:
    origin = request.headers.get("Origin")
    if not origin:
        return True
    parsed = urlsplit(origin)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def build_app(state: DopeFleetState) -> web.Application:
    app = web.Application()

    async def index(_request: web.Request) -> web.Response:
        return web.Response(text=HTML_PATH.read_text(encoding="utf-8"), content_type="text/html")

    async def api_snapshot(_request: web.Request) -> web.Response:
        return web.Response(text=_wire(state.snapshot_dict()), content_type="application/json")

    async def ws_handler(request: web.Request) -> web.StreamResponse:
        if not _allowed_origin(request):
            raise web.HTTPForbidden(text="disallowed origin")
        ws = web.WebSocketResponse(heartbeat=20.0)
        await ws.prepare(request)
        ev = asyncio.Event()
        state.waiters.add(ev)
        try:
            revs = state.revisions.copy()
            await ws.send_str(_wire(state.snapshot_dict()))
            while not ws.closed:
                await ev.wait()
                ev.clear()
                changed = [h for h in revs if state.revisions[h] != revs[h]]
                if not changed:
                    await ws.send_str(_wire({"type": "tick", "at": state.at, "fleet": state.summary}))
                for idx, h in enumerate(changed):
                    revs[h] = state.revisions[h]
                    summary = state.summary if idx == len(changed) - 1 else {
                        k: v for k, v in state.summary.items() if k != "history"
                    }
                    await ws.send_str(_wire({"type": "update", "at": state.at, "host": state.hosts[h], "fleet": summary}))
        except (ConnectionResetError, asyncio.CancelledError, web.HTTPException):
            pass
        finally:
            state.waiters.discard(ev)
        return ws

    app.router.add_get("/", index)
    app.router.add_get("/api/snapshot", api_snapshot)
    app.router.add_get("/ws", ws_handler)
    return app


async def _serve(args: argparse.Namespace) -> None:
    hosts = tuple(args.hosts.split(",")) if args.hosts else DEFAULT_HOSTS
    local_host = args.local_host or socket.gethostname().split(".")[0]
    state = DopeFleetState(hosts)
    app = build_app(state)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, args.bind, args.port)
    await site.start()
    print(f"[DOPE Dashboard] Listening on http://{args.bind}:{args.port}/ (hosts={','.join(hosts)})", flush=True)

    async def clock() -> None:
        while True:
            state.pulse()
            await asyncio.sleep(1.0)

    tasks = [asyncio.create_task(clock())]
    tasks.extend(asyncio.create_task(collect_host(h, state, local_host)) for h in hosts)
    try:
        await asyncio.gather(*tasks)
    finally:
        for t in tasks:
            t.cancel()
        await runner.cleanup()


def main() -> None:
    ap = argparse.ArgumentParser(description="Live DOPE Fleet & Fidelity KPI Dashboard")
    ap.add_argument("--bind", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8768)
    ap.add_argument("--hosts", default=",".join(DEFAULT_HOSTS))
    ap.add_argument("--local-host", default="")
    asyncio.run(_serve(ap.parse_args()))


if __name__ == "__main__":
    main()
