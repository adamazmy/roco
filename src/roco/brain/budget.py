"""Cloud spending caps for the Claude brain (``cloud.budget`` in the config).

Spend is read from workstream D's usage log (``~/.roco/cloud_usage.jsonl``), which holds
the real token counts of every Claude call (cache reads and writes included) and its
USD cost from ``cloud.pricing``. Calls from the body's own fallback count too: it is the
same key and the same bill. Days and months are local (``assistant.timezone``), so the
daily cap resets at Adam's midnight. Prices are in USD; ``usd_to_gbp`` converts.

The router asks :meth:`CloudBudget.status` before each cloud turn:

- ``ok``: under ``warn_at`` (80 %) of the daily cap;
- ``warn``: past 80 % of the daily cap (ROCO mentions it once a day);
- ``daily`` / ``monthly``: the cap is reached; the brain stays local until local
  midnight / the first of next month.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from roco.config import Config
from roco.fallback.usage import Usage, estimate_cost, usage_file

log = logging.getLogger(__name__)

DEFAULT_DAILY_GBP = 0.20
DEFAULT_MONTHLY_GBP = 5.00
DEFAULT_USD_TO_GBP = 0.79
DEFAULT_WARN_AT = 0.8


@dataclass(frozen=True)
class Spend:
    today_gbp: float
    month_gbp: float
    daily_gbp: float
    monthly_gbp: float
    warn_at: float

    @property
    def status(self) -> str:
        if self.monthly_gbp > 0 and self.month_gbp >= self.monthly_gbp:
            return "monthly"
        if self.daily_gbp > 0 and self.today_gbp >= self.daily_gbp:
            return "daily"
        if self.daily_gbp > 0 and self.today_gbp >= self.warn_at * self.daily_gbp:
            return "warn"
        return "ok"


class CloudBudget:
    """Sums the usage log by local day and month. Reads only what was appended since."""

    def __init__(
        self,
        path: Path,
        *,
        daily_gbp: float = DEFAULT_DAILY_GBP,
        monthly_gbp: float = DEFAULT_MONTHLY_GBP,
        usd_to_gbp: float = DEFAULT_USD_TO_GBP,
        warn_at: float = DEFAULT_WARN_AT,
        pricing: Mapping[str, Any] | None = None,
        tz: ZoneInfo | None = None,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        self.path = path
        self.daily_gbp = float(daily_gbp)
        self.monthly_gbp = float(monthly_gbp)
        self.usd_to_gbp = float(usd_to_gbp)
        self.warn_at = float(warn_at)
        self.pricing = pricing or {}
        self.tz = tz
        self._now = now or (lambda: datetime.now(self.tz))
        self._lock = threading.Lock()
        self._offset = 0
        self._usd_by_day: dict[date, float] = defaultdict(float)

    @classmethod
    def from_config(cls, cfg: Config, now: Callable[[], datetime] | None = None) -> CloudBudget:
        from roco.tools.clock import local_timezone

        b = cfg.get_path("cloud.budget", {}) or {}
        pricing = cfg.get_path("cloud.pricing", {}) or {}
        return cls(
            usage_file(cfg),
            daily_gbp=float(b.get("daily_gbp", DEFAULT_DAILY_GBP)),
            monthly_gbp=float(b.get("monthly_gbp", DEFAULT_MONTHLY_GBP)),
            usd_to_gbp=float(b.get("usd_to_gbp", DEFAULT_USD_TO_GBP)),
            warn_at=float(b.get("warn_at", DEFAULT_WARN_AT)),
            pricing=pricing if isinstance(pricing, Mapping) else {},
            tz=local_timezone(cfg),
            now=now,
        )

    def spend(self) -> Spend:
        with self._lock:
            self._read_new()
            today = self._now().date()
            month_usd = sum(v for d, v in self._usd_by_day.items()
                            if (d.year, d.month) == (today.year, today.month))
            return Spend(
                today_gbp=self._usd_by_day.get(today, 0.0) * self.usd_to_gbp,
                month_gbp=month_usd * self.usd_to_gbp,
                daily_gbp=self.daily_gbp,
                monthly_gbp=self.monthly_gbp,
                warn_at=self.warn_at,
            )

    def status(self) -> str:
        return self.spend().status

    def today(self) -> date:
        return self._now().date()

    def resets_at(self, status: str) -> datetime:
        """When a ``daily`` / ``monthly`` cap stops applying (local midnight / the 1st)."""
        now = self._now()
        if status == "monthly":
            first = (now.replace(day=28) + timedelta(days=4)).replace(day=1)
            return first.replace(hour=0, minute=0, second=0, microsecond=0)
        return (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

    # ------------------------------------------------------------------ internals
    def _read_new(self) -> None:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            self._offset = 0
            self._usd_by_day.clear()
            return
        except OSError as exc:
            log.warning("Cannot read the cloud usage log %s: %s", self.path, exc)
            return
        if size < self._offset:  # the log was rotated or deleted: start again
            self._offset = 0
            self._usd_by_day.clear()
        if size == self._offset:
            return
        try:
            with self.path.open("rb") as fh:
                fh.seek(self._offset)
                data = fh.read()
        except OSError as exc:
            log.warning("Cannot read the cloud usage log %s: %s", self.path, exc)
            return
        end = data.rfind(b"\n") + 1  # a line still being written is read next time
        self._offset += end
        for raw in data[:end].splitlines():
            self._add(raw)

    def _add(self, raw: bytes) -> None:
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return
        if not isinstance(rec, dict):
            return
        try:
            stamp = datetime.fromisoformat(str(rec.get("ts")))
        except ValueError:
            return
        if self.tz is not None and stamp.tzinfo is not None:
            stamp = stamp.astimezone(self.tz)
        cost = rec.get("cost_usd")
        if cost is None:  # model missing from the price table when it was logged
            cost = estimate_cost(str(rec.get("model", "")), Usage.from_api(rec), self.pricing)
        try:
            self._usd_by_day[stamp.date()] += float(cost or 0.0)
        except (TypeError, ValueError):
            return


def format_budget(spend: Spend) -> str:
    """Two lines for ``roco cloud-usage``: today and this month against the caps."""

    def row(label: str, used: float, cap: float) -> str:
        pct = f"{used / cap:.0%}" if cap > 0 else "-"
        cap_text = f"£{cap:.2f}" if cap > 0 else "no cap"
        return f"{label:<12}£{used:.4f} of {cap_text} ({pct})"

    state = {"ok": "under budget", "warn": f"past {spend.warn_at:.0%} of today's cap",
             "daily": "daily cap reached: local brain until midnight",
             "monthly": "monthly cap reached: local brain until the 1st"}[spend.status]
    return "\n".join([
        row("Today", spend.today_gbp, spend.daily_gbp),
        row("This month", spend.month_gbp, spend.monthly_gbp),
        f"Budget: {state}",
    ])
