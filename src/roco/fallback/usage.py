"""Cloud spend log (spec §10: "logs cloud spend").

Every Claude call appends one JSON line to ``~/.roco/cloud_usage.jsonl``::

    {"ts": "2026-11-21T20:15:03+00:00", "model": "claude-haiku-5-5", "ok": true,
     "input_tokens": 512, "output_tokens": 48, "cache_read_input_tokens": 0,
     "cache_creation_input_tokens": 0, "cost_usd": 0.0000752, "first_text_ms": 640,
     "total_ms": 1210, "purpose": "fallback"}

Failed calls are logged too (``ok: false`` plus ``error``), with whatever tokens the
API reported. The cost is an estimate from the ``cloud.pricing`` table in the config
(USD per million tokens); a model missing from the table is logged with ``cost_usd: null``.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roco.config import Config
from roco.paths import resolve

log = logging.getLogger(__name__)

DEFAULT_USAGE_FILE = "cloud_usage.jsonl"
_LOCK = threading.Lock()


@dataclass(frozen=True)
class Usage:
    """Token counts of one call, as reported by the API."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @classmethod
    def from_api(cls, usage: Any) -> Usage:
        """From an SDK ``Usage`` object or a dict; missing/None fields count as 0."""
        if usage is None:
            return cls()

        def get(name: str) -> int:
            value = usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, 0)
            try:
                return int(value or 0)
            except (TypeError, ValueError):
                return 0

        return cls(
            get("input_tokens"), get("output_tokens"),
            get("cache_read_input_tokens"), get("cache_creation_input_tokens"),
        )


def usage_file(cfg: Config) -> Path:
    return resolve(str(cfg.get_path("cloud.usage_file", DEFAULT_USAGE_FILE) or DEFAULT_USAGE_FILE))


def estimate_cost(model: str, usage: Usage, pricing: Mapping[str, Any]) -> float | None:
    """USD estimate, or None when the model has no price in the table."""
    price = pricing.get(model)
    if not isinstance(price, Mapping):
        return None
    try:
        p_in = float(price.get("input", 0))
        p_out = float(price.get("output", 0))
        p_read = float(price.get("cache_read", p_in * 0.1))
        p_write = float(price.get("cache_write", p_in * 1.25))
    except (TypeError, ValueError):
        return None
    total = (
        usage.input_tokens * p_in
        + usage.output_tokens * p_out
        + usage.cache_read_input_tokens * p_read
        + usage.cache_creation_input_tokens * p_write
    )
    return round(total / 1_000_000, 8)


class UsageLog:
    """Appends records to the JSONL file. Logging failures never break a reply."""

    def __init__(self, path: Path, pricing: Mapping[str, Any] | None = None) -> None:
        self.path = path
        self.pricing: Mapping[str, Any] = pricing or {}

    @classmethod
    def from_config(cls, cfg: Config) -> UsageLog:
        pricing = cfg.get_path("cloud.pricing", {}) or {}
        return cls(usage_file(cfg), pricing if isinstance(pricing, Mapping) else {})

    def record(
        self,
        model: str,
        usage: Usage,
        *,
        ok: bool,
        error: str | None = None,
        first_text_ms: int | None = None,
        total_ms: int | None = None,
        purpose: str = "fallback",
        now: datetime | None = None,
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "ts": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
            "model": model,
            "purpose": purpose,
            "ok": ok,
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_input_tokens": usage.cache_read_input_tokens,
            "cache_creation_input_tokens": usage.cache_creation_input_tokens,
            "cost_usd": estimate_cost(model, usage, self.pricing),
            "first_text_ms": first_text_ms,
            "total_ms": total_ms,
        }
        if error:
            entry["error"] = error[:300]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps(entry, ensure_ascii=False)
            with _LOCK, self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError as exc:
            log.warning("Could not write the cloud usage log %s: %s", self.path, exc)
        return entry


# --------------------------------------------------------------------------- summary


@dataclass
class Bucket:
    calls: int = 0
    failed: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    unpriced: int = 0
    models: set[str] = field(default_factory=set)

    def add(self, rec: Mapping[str, Any]) -> None:
        self.calls += 1
        if not rec.get("ok", True):
            self.failed += 1
        self.input_tokens += int(rec.get("input_tokens") or 0) + int(
            rec.get("cache_read_input_tokens") or 0
        ) + int(rec.get("cache_creation_input_tokens") or 0)
        self.output_tokens += int(rec.get("output_tokens") or 0)
        cost = rec.get("cost_usd")
        if cost is None:
            self.unpriced += 1
        else:
            self.cost_usd += float(cost)
        if rec.get("model"):
            self.models.add(str(rec["model"]))


def read_records(path: Path) -> list[dict[str, Any]]:
    """Every valid line of the log; corrupt lines are skipped."""
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return records
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            records.append(rec)
    return records


def summarize(records: Iterable[Mapping[str, Any]], by: str = "month") -> dict[str, Bucket]:
    """Group by ``day`` (YYYY-MM-DD) or ``month`` (YYYY-MM), plus a ``total`` bucket."""
    width = 10 if by == "day" else 7
    buckets: dict[str, Bucket] = defaultdict(Bucket)
    total = Bucket()
    for rec in records:
        key = str(rec.get("ts", "unknown"))[:width] or "unknown"
        buckets[key].add(rec)
        total.add(rec)
    result = dict(sorted(buckets.items()))
    result["total"] = total
    return result


def format_summary(summary: Mapping[str, Bucket], path: Path) -> str:
    if summary["total"].calls == 0:
        return f"No cloud calls logged yet ({path})."
    rows = [f"Cloud usage from {path}", ""]
    rows.append(f"{'period':<12}{'calls':>7}{'failed':>8}{'tokens in':>12}{'tokens out':>12}"
                f"{'est. USD':>11}")
    for key, b in summary.items():
        if key == "total":
            rows.append("-" * 62)
        cost = f"{b.cost_usd:.4f}" + ("*" if b.unpriced else "")
        rows.append(f"{key:<12}{b.calls:>7}{b.failed:>8}{b.input_tokens:>12}"
                    f"{b.output_tokens:>12}{cost:>11}")
    total = summary["total"]
    if total.unpriced:
        rows.append(f"* {total.unpriced} call(s) used a model missing from cloud.pricing")
    rows.append(f"Models: {', '.join(sorted(total.models)) or '-'}")
    return "\n".join(rows)
