# Architecture

```
          ┌──────────── Listener ────────────┐
 mic ───▶ │ wake word → record → faster-whisper │ ──text──┐
          └──────────────────────────────────┘          ▼
                                                    ┌────────┐  tools  ┌──────────────┐
                                                    │ Brain  │ ◀─────▶ │ ToolRegistry │
                                                    └────────┘         └──────────────┘
                                         sentences │   │ emotion/state
          ┌──────────── Speaker ────────────┐      ▼   ▼
 speaker ◀│ queue → Piper → sounddevice     │◀── say  EventBus ──▶ Body ──▶ Face (pygame)
          └─────────────────────────────────┘  mouth ──▶             └──▶ ServoBackend
```

Everything talks through an in-process **event bus** (`roco.bus.EventBus`), so the brain
never knows whether a face or servos exist, and the body never knows where text came from.

## Bus topics

| Topic | Payload | Published by | Consumed by |
|---|---|---|---|
| `state` | `State`: idle, listening, thinking, speaking, sleeping | Listener, Brain, Speaker | Body/Face |
| `emotion` | `Emotion`: neutral, happy, excited, curious, confused, sad, sleepy, surprised | Brain (from the reply's leading tag) | Body/Face, gestures |
| `mouth` | `float` 0..1 | Speaker, per audio block | Face |
| `announce` | `str` | Reminder scheduler | Voice loop (speaks it even while idle), `roco chat` (prints it) |

Handlers run synchronously in the publisher's thread and must be quick. Use the typed
helpers (`bus.set_state`, `set_emotion`, `set_mouth`, `announce`) rather than raw strings.

## Interfaces (the contract between workstreams)

| Interface | Module | Key methods |
|---|---|---|
| `Brain` | `roco.brain.base` | `respond(text, on_sentence) -> str`, `reset()` |
| `Speaker` | `roco.voice.base` | `say(text)`, `wait_done(timeout)`, `is_speaking`, `stop()`, `close()` |
| `Listener` | `roco.voice.base` | `listen() -> str \| None`, `listen_followup(timeout)`, `set_muted(bool)`, `close()` |
| `Face` | `roco.body.base` | `run()` (main thread), `stop()`, `set_state`, `set_emotion`, `set_mouth` |
| `ServoBackend` | `roco.body.base` | `set_angle(channel, deg)`, `get_angle(channel)`, `release`, `close` |
| `Body` | `roco.body.base` | `attach(bus)`, `detach()`, `on_state/on_emotion/on_mouth` hooks |

The docstrings in those modules are the detailed contract (who publishes which state, what
happens on errors). Null implementations (`NullSpeaker`, `NullFace`) exist for text mode and tests.

## Tools

Tool modules expose `register(registry, ctx)` and are listed in `roco.tools.TOOL_MODULES`.
A module decides itself whether to register anything (Google registers nothing unless it is
enabled and authenticated); a module whose optional dependency is missing is skipped.
`ToolRegistry.call()` never raises: every failure comes back to the model as `"Error: ..."` text.
`roco.tools.clock` is the reference example.

## Threading

pygame must own the main thread. `roco.runtime.run_with_face(face, worker)` runs the
assistant loop on a background thread and the face on the main thread; closing the window
sets the worker's stop event, and a finished worker closes the face.

## Configuration and state

`roco.config.load_config()` deep-merges `config.yaml` over `DEFAULTS`. Relative file paths
resolve into `~/.roco` (`roco.paths.resolve`), which holds history, reminders, Google
credentials/token and models. None of it is committed.

## Workstreams

| | Branch | Owns |
|---|---|---|
| A | `feat/brain` | `roco/brain/*` (except `base.py`), `roco/tools/clock.py`, `roco/tools/reminders.py`, `cmd_chat`, mock Ollama tests |
| B | `feat/voice` | `roco/voice/*` (except `base.py`), `cmd_run`, `cmd_download_models`, voice checks in `roco/doctor.py` |
| C | `feat/body` | `roco/body/*` (except `base.py`), `cmd_demo_face` |
| D | `feat/google` | `roco/tools/google.py`, `roco/integrations/`, `cmd_auth_google`, Google checks in `roco/doctor.py`, README Google section |

Each workstream edits only its own `cmd_*` function in `cli.py` (importing its implementation
lazily inside the function) and adds dependencies only to its own extra in `pyproject.toml`.
Changing a `base.py` interface needs a separate small PR so the others can rebase.

## Phase 3: brain server + body client

The product spec is [`docs/ROCO_SPEC.md`](ROCO_SPEC.md) (§9 architecture, §11 targets).
Heavy work runs on the PC; the Pi 5 (2 GB) is the body only.

```
 Pi / PC: roco body                                   PC: roco brain
 ┌──────────────────────────────┐   WebSocket    ┌──────────────────────────────────┐
 │ mic → AEC → wake word → VAD  │ ── audio up ──▶ │ streaming STT (faster-whisper)    │
 │ playback (stops on barge-in) │ ◀─ TTS audio ── │ Brain (Ollama + tools + persona)  │
 │ face, servos (from messages) │ ◀─ text/state ─ │ sentence TTS · filler lines       │
 │ fallback: Claude + tiny.en   │ ── control ───▶ │ per-turn latency log              │
 └──────────────────────────────┘                └──────────────────────────────────┘
```

`roco run` runs both halves in one process, joined by `memory_pipe()`. Protocol:
[`docs/PROTOCOL.md`](PROTOCOL.md).

| Package | Owner (Phase 3) | Contents |
|---|---|---|
| `roco.protocol` | planning commit | Messages, audio frames, transports, link token |
| `roco.server` | A | WebSocket server, sessions, streaming STT → brain → TTS, interrupts, latency log, `roco brain` |
| `roco.client` | B | Connection + reconnect, mic upload, wake word, VAD, playback, barge-in, face/servos, `roco body`, `roco run` |
| `roco.voice.aec` | B | Software echo cancellation |
| `roco.personality` | C | Settings, presets, prompt rendering, voice commands, per-person overrides |
| `roco.voice` TTS backends | C | Tuned Piper + post-processing, Kokoro |
| `roco.fallback` | D | Brain chain (PC → Claude → canned), Claude client, cost log, local tiny.en STT |

The brain stays the `Brain` interface (`respond(text, on_sentence)`); the server wraps it.
The body client decides the end of speech, cuts playback on barge-in locally, then tells the
brain. When the brain link is down, the body asks the fallback chain instead.
