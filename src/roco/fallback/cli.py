"""``roco cloud-usage`` and ``roco cloud-key``."""

from __future__ import annotations

import getpass
import sys

from roco.config import Config
from roco.fallback.keys import ENV_VAR, key_source, load_api_key, mask, save_api_key
from roco.fallback.usage import format_summary, read_records, summarize, usage_file


def run_cloud_usage(cfg: Config, by: str = "month") -> int:
    path = usage_file(cfg)
    print(format_summary(summarize(read_records(path), by=by), path))
    try:  # today's and this month's spend against the caps (cloud.budget)
        from roco.brain.budget import CloudBudget, format_budget
    except ImportError:  # a body-only install without the brain's dependencies
        return 0
    print()
    print(format_budget(CloudBudget.from_config(cfg).spend()))
    return 0


def run_cloud_key(cfg: Config, from_stdin: bool = False, show: bool = False) -> int:
    """Store the Anthropic API key in ~/.roco (hidden prompt; never echoed or logged)."""
    if show:
        key = load_api_key(cfg)
        if not key:
            print(f"No Anthropic key (set {ENV_VAR} or run `roco cloud-key`).")
            return 1
        print(f"Anthropic key: {mask(key)} (from {key_source(cfg)})")
        return 0
    if from_stdin:
        key = sys.stdin.readline()
    else:
        key = getpass.getpass("Paste your Anthropic API key (input hidden), then Enter: ")
    try:
        path = save_api_key(cfg, key)
    except ValueError as exc:
        print(f"Not saved: {exc}", file=sys.stderr)
        return 1
    print(f"Saved {mask(key.strip())} to {path} (only your user can read it).")
    return 0
