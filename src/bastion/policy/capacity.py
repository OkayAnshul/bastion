"""The daily review budget, shared by every scoring worker through Redis.

There is one counter per event-time UTC day. A reservation increments it; if that takes the count
past the budget, the increment is undone and the reservation refused. Only over-budget increments
are undone, so concurrent requests can never push a day past its budget. The offline equivalent is
``bastion.policy.decide.within_daily_budget``, and a parity test replays one stream through both.
"""

from __future__ import annotations

from datetime import datetime

import redis.asyncio

SECONDS_PER_DAY = 86_400
KEY_TTL_S = 3 * SECONDS_PER_DAY  # counters for older event days expire


def event_day(event_ts: datetime) -> int:
    """Days since 1970-01-01 UTC: the same day key offline evaluation uses."""
    return int(event_ts.timestamp() // SECONDS_PER_DAY)


class ReviewCapacity:
    def __init__(self, client: redis.asyncio.Redis, prefix: str, reviews_per_day: int) -> None:
        if reviews_per_day < 0:
            raise ValueError("reviews_per_day must be non-negative")
        self._client = client
        self._prefix = prefix
        self.reviews_per_day = reviews_per_day

    def key(self, day: int) -> str:
        return f"{self._prefix}:policy:reviews:{day}"

    async def try_reserve(self, day: int) -> bool:
        """Take one of ``day``'s reviews if any is left."""
        if self.reviews_per_day == 0:
            return False
        key = self.key(day)
        pipe = self._client.pipeline(transaction=False)
        pipe.incr(key)
        pipe.expire(key, KEY_TTL_S)
        count, _ = await pipe.execute()
        if int(count) <= self.reviews_per_day:
            return True
        await self._client.decr(key)
        return False

    async def used(self, day: int) -> int:
        value = await self._client.get(self.key(day))
        return 0 if value is None else int(value)
