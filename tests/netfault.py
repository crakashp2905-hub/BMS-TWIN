"""Network fault-injection helpers shared by the gateway tests.

``ChaosProxy`` sits between a client and a real server and can cut a connection after N bytes,
swallow traffic (a link that stays up but goes silent), sever everything, or retarget to a new
upstream (a server restart).  ``collect`` runs a client in a thread so a hang fails one test
instead of the whole suite; ``wait_until`` polls an observable condition instead of sleeping a
guessed time.
"""

from __future__ import annotations

import socket
import threading
import time


class ChaosProxy:
    """A TCP proxy that can cut a connection after N bytes, swallow traffic, or retarget the upstream."""

    def __init__(self, upstream_port: int) -> None:
        self.upstream_port = upstream_port
        self.cut_after_bytes: int | None = None
        self.blackhole = False
        self.connections = 0
        self._socks: list[socket.socket] = []
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._srv = socket.socket()
        self._srv.bind(("127.0.0.1", 0))
        self._srv.listen(32)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._srv.accept()
            except (socket.timeout, OSError):
                continue
            try:
                upstream = socket.create_connection(("127.0.0.1", self.upstream_port), timeout=5)
            except OSError:
                client.close()
                continue
            with self._lock:
                self.connections += 1
                self._socks += [client, upstream]
            limit = self.cut_after_bytes
            threading.Thread(target=self._pump, args=(client, upstream, None, False), daemon=True).start()
            threading.Thread(target=self._pump, args=(upstream, client, limit, True), daemon=True).start()

    def _pump(self, src, dst, limit, server_to_client: bool) -> None:
        sent = 0
        try:
            while not self._stop.is_set():
                data = src.recv(4096)
                if not data:
                    break
                if server_to_client and self.blackhole:
                    continue                                          # swallow: the link looks alive but silent
                if limit is not None and sent + len(data) > limit:
                    dst.sendall(data[: max(0, limit - sent)])
                    break
                dst.sendall(data)
                sent += len(data)
        except OSError:
            pass
        finally:
            for s in (src, dst):
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()

    def sever_all(self) -> None:
        with self._lock:
            for s in self._socks:
                try:
                    s.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                s.close()
            self._socks.clear()

    def stop(self) -> None:
        self._stop.set()
        self.sever_all()
        self._srv.close()


def wait_until(predicate, timeout_s: float = 20.0) -> None:
    """Poll for an observable condition instead of sleeping a guessed time."""
    deadline = time.monotonic() + timeout_s
    while not predicate():
        assert time.monotonic() < deadline, "condition not reached in time"
        time.sleep(0.01)


def collect(client, limit: int | None = None, timeout_s: float = 40.0, **kw) -> list:
    """Run ``client.stream`` in a thread so a hang fails the test instead of the suite."""
    out: list = []
    err: list = []

    def run():
        try:
            for rec in client.stream(**kw):
                out.append(rec)
                if limit is not None and len(out) >= limit:
                    return
        except Exception as exc:                                              # surfaced to the test thread
            err.append(exc)
    th = threading.Thread(target=run, daemon=True)
    th.start()
    th.join(timeout_s)
    assert not th.is_alive(), f"client hung with {len(out)} records"
    if err:
        raise err[0]
    return out
