"""
dunetrace/buffer.py

Thread-safe ring buffer for the drain queue.
Drops the oldest event when full — the agent thread is never blocked.
"""
from __future__ import annotations

from collections import deque
from threading import Lock
from typing import Generic, List, Optional, TypeVar

T = TypeVar("T")


class RingBuffer(Generic[T]):
    """
    Fixed-capacity FIFO backed by a deque with maxlen.

    CPython's deque.append() and deque.popleft() are GIL-protected,
    making single-producer / single-consumer usage safe without an
    explicit lock. The Lock here guards multi-consumer drain calls.
    """

    def __init__(self, maxsize: int = 10_000) -> None:
        self._buf: deque[T] = deque(maxlen=maxsize)
        self._lock = Lock()

    def push(self, item: T) -> None:
        """Append item. If full, the oldest item is silently dropped."""
        self._buf.append(item)

    def drain(self, n: int = 100) -> List[T]:
        """Return up to *n* items from the front, removing them from the buffer."""
        with self._lock:
            batch: List[T] = []
            while self._buf and len(batch) < n:
                batch.append(self._buf.popleft())
            return batch

    def drain_all(self) -> List[T]:
        """Drain every item currently in the buffer."""
        with self._lock:
            items = list(self._buf)
            self._buf.clear()
            return items

    def __len__(self) -> int:
        return len(self._buf)

    def __bool__(self) -> bool:
        return bool(self._buf)
