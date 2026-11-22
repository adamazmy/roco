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
 │ playback (stops on barge-in) │ ◀─ TTS audio ── │ Brain (Claude → Ollama, tools)    │
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
| `roco.brain` | A | Claude + Ollama brains, router, budgets, local-only mode |
| `roco.client` | B | Connection + reconnect, mic upload, wake word, VAD, playback, barge-in, face/servos, `roco body`, `roco run` |
| `roco.voice.aec` | B | Software echo cancellation |
| `roco.personality` | C | Settings, presets, prompt rendering, voice commands, per-person overrides |
| `roco.voice` TTS backends | C | Tuned Piper + post-processing, Kokoro |
| `roco.fallback` | D | Brain chain (PC → Claude → canned), Claude client, cost log, local tiny.en STT |

The brain stays the `Brain` interface (`respond(text, on_sentence)`); the server wraps it.
The body client decides the end of speech, cuts playback on barge-in locally, then tells the
brain. When the brain link is down, the body asks the fallback chain instead.

### Brain selection: Claude by default, Ollama as the fallback (decided 2026-11-22)

Adam's decision: the PC brain answers on the **Claude API** by default and on **local
Ollama** (qwen2.5:7b) when the cloud is not an option. Why:

- **Quality and tool reliability.** Claude Haiku calls the right tool every time in the
  15-turn regression test and keeps her character; qwen2.5:7b needs safety nets (reminder
  repair, time prefetch, JSON tool-call recovery) and still misses some turns. Hard
  questions can escalate to a larger Claude model.
- **Privacy and offline.** Local stays one sentence away: "local only" sends nothing to the
  cloud, and no key / no internet / API errors fall back to Ollama without a gap.
- **Cost is capped.** A Haiku turn costs a fraction of a penny with prompt caching; daily
  (£0.20) and monthly (£5) caps switch to local when spent.

| Module | Role |
|---|---|
| `brain.core.ToolBrain` | Model-independent layer both brains share: persona (C's renderer), history, voice commands, time prefetch, reminder-claim repair, example-line filter, emotion tags, sentence streaming, barge-in. |
| `brain.claude_brain.ClaudeBrain` | Tool loop on the Messages API via D's `fallback.claude.ClaudeClient` (streaming, deadlines, usage log). Cache marks on the last tool, the system prompt and the last message. |
| `brain.ollama_brain.OllamaBrain` | Tool loop on Ollama (unchanged behaviour). |
| `brain.router.BrainRouter` | The `Brain` the server uses: picks Claude / local / offline line per turn, privacy mode, escalation, budget, switch lines. `create_brain` builds it. |
| `brain.budget.CloudBudget` | Today's and this month's spend from `~/.roco/cloud_usage.jsonl`, by local day. |

Speech-to-text and TTS stay local whatever the brain. The body's own Claude fallback
(`roco.fallback`, used when the PC is unreachable) is unchanged and shares the key, the
usage log and therefore the budget.

### Body client (`roco.client`)

| Module | Role |
|---|---|
| `link.LinkClient` | One thread: connect (injected), `hello` → `welcome`, receive loop, ping every 5 s (RTT, liveness), reconnect with jittered backoff 0.5 s → 10 s. |
| `body.BodyClient` | Mic state machine (idle → recording → reply → follow-up), turn messages, barge-in, late-frame dropping, fallback hook, per-turn metrics. |
| `playback.Player` / `CallbackOutput` | TTS frames per turn, pulled by a PortAudio callback: instant local cut, exact AEC reference, "heard" text, mouth level. |
| `vad` | Silero VAD v5 on onnxruntime (no torch), energy fallback, `VadEndpointer` (450 ms trailing silence). |
| `roco.voice.aec` | WebRTC AEC3 / Speex / none behind one protocol; see [`docs/AEC.md`](AEC.md). |
| `factory` / `app` | Real components from config; `roco body` (WebSocket) and `roco run` (`memory_pipe` + `roco.server.serve_connection`, a fresh pipe per connection). |

Threads: mic (state machine), link (incoming frames), player monitor (playback
events, mouth), PortAudio callbacks, and a fallback worker while the brain is down.
All body state is behind one lock; the audio callback only copies bytes.
