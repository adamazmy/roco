"""Command-line entry point: ``roco <command>``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Callable, Sequence

from roco import __version__
from roco.config import Config, ConfigError, load_config

EXIT_OK = 0
EXIT_FAIL = 1
EXIT_NOT_IMPLEMENTED = 3


def _not_implemented(feature: str, workstream: str) -> int:
    print(f"'{feature}' is not implemented yet (workstream {workstream}).", file=sys.stderr)
    return EXIT_NOT_IMPLEMENTED


# --------------------------------------------------------------------------- commands
# Each command takes (args, config) and returns an exit code. Workstreams replace
# the stubs below by importing their implementation lazily inside the function,
# so `roco --help` and text mode never import optional dependencies.


def cmd_run(args: argparse.Namespace, cfg: Config) -> int:
    """Full voice loop: wake word -> STT -> brain -> TTS, with face and servos."""
    from roco.voice.app import run_voice

    return EXIT_OK if run_voice(cfg) == 0 else EXIT_FAIL


def cmd_chat(args: argparse.Namespace, cfg: Config) -> int:
    """Text mode: keyboard in, printed sentences out."""
    from roco.brain.chat import run_chat

    return run_chat(args, cfg)


def cmd_doctor(args: argparse.Namespace, cfg: Config) -> int:
    from roco.doctor import print_report, run_checks

    return EXIT_OK if print_report(run_checks(cfg)) else EXIT_FAIL


def cmd_auth_usage(args: argparse.Namespace, cfg: Config) -> int:
    print("usage: roco auth {google,outlook}", file=sys.stderr)
    return EXIT_FAIL


def cmd_auth_google(args: argparse.Namespace, cfg: Config) -> int:
    """Browser sign-in for Google Calendar + Gmail (read-only); saves the token in ~/.roco."""
    from roco.integrations.google_auth import run_auth_flow

    return EXIT_OK if run_auth_flow(cfg) == 0 else EXIT_FAIL


def cmd_auth_outlook(args: argparse.Namespace, cfg: Config) -> int:
    """Device-code sign-in for Outlook.com / Hotmail mail (read-only)."""
    from roco.integrations.outlook_auth import run_auth

    return EXIT_OK if run_auth(cfg, browser=getattr(args, "browser", False)) == 0 else EXIT_FAIL


def cmd_download_models(args: argparse.Namespace, cfg: Config) -> int:
    """Download the wake-word models, the Piper voice and the whisper model (idempotent)."""
    from roco.voice.models import main_download

    return EXIT_OK if main_download(cfg, force=args.force) else EXIT_FAIL


def cmd_cloud_usage(args: argparse.Namespace, cfg: Config) -> int:
    """Summarise the fallback brain's Claude calls and estimated spend."""
    from roco.fallback.cli import run_cloud_usage

    return run_cloud_usage(cfg, by=args.by)


def cmd_cloud_key(args: argparse.Namespace, cfg: Config) -> int:
    """Store the Anthropic API key for the fallback brain in ~/.roco."""
    from roco.fallback.cli import run_cloud_key

    return run_cloud_key(cfg, from_stdin=args.stdin, show=args.show)


def cmd_demo_face(args: argparse.Namespace, cfg: Config) -> int:
    """Cycle through every emotion and state on the face."""
    try:
        import pygame  # noqa: F401
    except ImportError:
        print('demo-face needs pygame: pip install -e ".[face]"', file=sys.stderr)
        return EXIT_FAIL
    from roco.body.demo import run_demo

    print("Space = pause, Left/Right = step, Esc = quit.")
    run_demo(cfg)
    return EXIT_OK


# --------------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="roco", description="ROCO, a local-first desk robot assistant."
    )
    parser.add_argument("--version", action="version", version=f"roco {__version__}")
    parser.add_argument("-c", "--config", help="path to config.yaml")
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def add(name: str, func: Callable[..., int], help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.set_defaults(func=func)
        return p

    add("run", cmd_run, "run the full voice assistant (wake word, speech, face)")

    chat = add("chat", cmd_chat, "chat with ROCO by keyboard")
    chat.add_argument("--speak", action="store_true", help="also speak replies aloud")
    chat.add_argument("--face", action="store_true", help="also show the animated face")
    chat.add_argument("--reset", action="store_true", help="start with an empty history")

    add("doctor", cmd_doctor, "check Ollama, models, audio devices and optional extras")

    auth = add("auth", cmd_auth_usage, "connect accounts")
    auth_sub = auth.add_subparsers(dest="provider", metavar="<provider>")
    google = auth_sub.add_parser("google", help="connect Google Calendar and Gmail (read-only)")
    google.set_defaults(func=cmd_auth_google)
    outlook = auth_sub.add_parser("outlook", help="connect Outlook.com / Hotmail mail (read-only)")
    outlook.set_defaults(func=cmd_auth_outlook)
    outlook.add_argument(
        "--browser", action="store_true",
        help="sign in through the browser instead of a device code",
    )

    dl = add("download-models", cmd_download_models, "download wake-word and voice models")
    dl.add_argument("--force", action="store_true", help="re-download even if present")

    usage = add("cloud-usage", cmd_cloud_usage, "show Claude fallback calls and estimated cost")
    usage.add_argument("--by", choices=("day", "month"), default="month", help="group by")
    key = add("cloud-key", cmd_cloud_key, "store the Anthropic API key in ~/.roco (hidden prompt)")
    key.add_argument("--stdin", action="store_true", help="read the key from stdin")
    key.add_argument("--show", action="store_true", help="show which key is in use (masked)")

    add("demo-face", cmd_demo_face, "cycle the face through every emotion")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_FAIL

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return EXIT_FAIL

    level = "DEBUG" if args.verbose else str(cfg.get_path("logging.level", "INFO")).upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return int(args.func(args, cfg) or EXIT_OK)
    except KeyboardInterrupt:
        return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
