# ROCO

ROCO is a local-first voice assistant for a small desktop robot: a screen face and a
pan/tilt head. You say "hey Jarvis", ask a question, and ROCO answers out loud in a
British voice while its face reacts.

All the AI runs on your own machine:

| Part | Runs on | Tech |
|---|---|---|
| Language model | your PC | [Ollama](https://ollama.com), `qwen2.5:7b` with native tool calling |
| Speech-to-text | PC now, Pi later | faster-whisper `base.en` (int8, CPU) |
| Text-to-speech | PC now, Pi later | Piper, `en_GB-alan-medium` |
| Wake word | PC now, Pi later | openWakeWord `hey_jarvis` (ONNX) |
| Face | PC window now, Pi display later | pygame |
| Head | simulated now, PCA9685 later | 2 servos, pan/tilt |

Calendar and email use the Google APIs (read-only for email), so that data is cloud by
nature; "local" means that all the AI processing stays on your machine.

> **Status:** Phase 0 scaffold. Package layout, config, event bus, tool registry, CLI
> and interfaces are in place; the brain, voice, face and Google integration land in
> follow-up PRs. See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quickstart (Windows)

1. Install Python 3.11+ and [Ollama](https://ollama.com/download), then pull the model:
   ```powershell
   ollama pull qwen2.5:7b
   ```
2. Clone and install everything:
   ```powershell
   git clone https://github.com/adamazmy/roco.git
   cd roco
   py -3.11 -m venv .venv
   .venv\Scripts\Activate.ps1
   pip install -e ".[all]"
   ```
   Text mode only needs `pip install -e .`
3. Download the wake-word and voice models, then check your setup:
   ```powershell
   roco download-models
   roco doctor
   ```
4. Talk to it:
   ```powershell
   roco chat            # keyboard in, text out
   roco chat --speak    # ...and spoken replies
   roco run             # full voice mode: say "hey Jarvis"
   ```

## Commands

| Command | What it does |
|---|---|
| `roco run` | Full voice loop with face and (simulated) servos |
| `roco chat [--speak] [--face] [--reset]` | Text chat |
| `roco doctor` | Checks Ollama, the model, audio devices, model files and extras |
| `roco auth google` | Connects Google Calendar + Gmail (read-only) |
| `roco download-models` | Downloads openWakeWord models and the Piper voice |
| `roco demo-face` | Cycles the face through every emotion |

Global options: `-c/--config PATH`, `-v/--verbose`.

## Configuration

Copy [`config.example.yaml`](config.example.yaml) to `config.yaml` (in the folder you run
`roco` from) or to `~/.roco/config.yaml`, and keep only the keys you change: it is
deep-merged over the defaults in [`src/roco/config.py`](src/roco/config.py).
`ROCO_CONFIG` points at a specific file; `ROCO_HOME` moves the state directory.

ROCO's state (conversation history, reminders, Google token, models) lives in
`~/.roco/` and never in the repository.

## Google setup

_Coming with the calendar/email workstream._

## Development

```powershell
pip install -e ".[dev]"
ruff check .
pytest
```

CI runs the test suite in text mode on Windows and Linux. Commits follow
[Conventional Commits](https://www.conventionalcommits.org/).

## More

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): event bus, interfaces, threading, workstreams
- [docs/HARDWARE.md](docs/HARDWARE.md): the recommended v1 robot body
- [docs/ROADMAP.md](docs/ROADMAP.md): what comes next

## License

MIT
