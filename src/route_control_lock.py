"""Cross-process exclusive ownership for bot control and route recording."""

from __future__ import annotations

import os
from pathlib import Path

from src.app_paths import user_data_root


class RouteControlBusyError(RuntimeError):
    pass


class RouteControlLock:
    def __init__(self, owner):
        self.owner = str(owner)
        self.handle = None

    @property
    def acquired(self):
        return self.handle is not None

    def acquire(self):
        if self.acquired:
            return
        path = user_data_root() / "locks" / "route_control.lock"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(path, "a+b")
        try:
            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (OSError, BlockingIOError) as exc:
            handle.close()
            raise RouteControlBusyError(
                "自动控制与路线录制器不能同时运行；请先关闭另一个功能。"
            ) from exc
        self.handle = handle

    def release(self):
        handle, self.handle = self.handle, None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.release()
