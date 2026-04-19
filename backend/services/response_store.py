from __future__ import annotations

import asyncio
import copy
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StoredResponse:
    response_id: str
    payload: dict[str, Any]
    history_messages: list[dict[str, Any]]


class InMemoryResponseStore:
    def __init__(self) -> None:
        self._items: dict[str, StoredResponse] = {}
        self._lock = asyncio.Lock()

    async def save(self, response_id: str, payload: dict[str, Any], history_messages: list[dict[str, Any]]) -> None:
        async with self._lock:
            self._items[response_id] = StoredResponse(
                response_id=response_id,
                payload=copy.deepcopy(payload),
                history_messages=copy.deepcopy(history_messages),
            )

    async def get(self, response_id: str) -> StoredResponse | None:
        async with self._lock:
            item = self._items.get(response_id)
            if item is None:
                return None
            return StoredResponse(
                response_id=item.response_id,
                payload=copy.deepcopy(item.payload),
                history_messages=copy.deepcopy(item.history_messages),
            )
