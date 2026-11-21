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
| `roco auth outlook` | Connects Outlook.com / Hotmail email (read-only) |
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

ROCO can read your Google Calendar, add events (always after asking you to confirm) and
read your Gmail. It can **never** send, delete or change email. You create your own
Google Cloud "app" once, so your data only goes between your PC and Google. It is free.

1. **Create a project.** Open <https://console.cloud.google.com/>, sign in with the
   Google account ROCO should use, and create a new project (e.g. `roco`).
2. **Enable the APIs.** In *APIs & Services > Library*, search for and enable
   **Google Calendar API** and **Gmail API**.
3. **Configure the OAuth consent screen** (*APIs & Services > OAuth consent screen*, now
   also called *Google Auth Platform*): user type **External**, app name `ROCO`, your
   email as support and developer contact. Leave the app in **Testing** and, under
   *Audience > Test users*, add your own Google address. You don't need to add scopes here.
4. **Create the OAuth client.** *APIs & Services > Credentials > Create credentials >
   OAuth client ID* (or *Clients > Create client*), application type **Desktop app**,
   name `ROCO desktop`. Click **Download JSON**.
5. **Save the JSON** as `~/.roco/google_credentials.json`
   (on Windows: `C:\Users\<you>\.roco\google_credentials.json`). Keep it private; it is
   never committed.
6. **Sign in:**
   ```powershell
   pip install -e ".[google]"
   roco auth google
   ```
   Your browser opens. Because the app is in Testing, Google warns "Google hasn't
   verified this app": click **Continue**, then allow both permissions (calendar events,
   and read-only Gmail). The token is saved to `~/.roco/google_token.json`.
7. **Turn it on** in `config.yaml`:
   ```yaml
   google:
     enabled: true
     # calendar_id: primary   # or another calendar's id
   ```
   If your email is Hotmail/Outlook rather than Gmail, keep Google for the calendar
   and see [Outlook setup](#outlook-setup-hotmail--outlookcom-email) for email.
8. **Check:** `roco doctor` should show the Google checks as OK. Then try `roco chat`:
   "what's on tomorrow?", "any unread emails?".

ROCO's Google tools are `get_events`, `create_event`, `get_unread_emails`,
`search_emails` (Gmail search syntax, e.g. `from:alice newer_than:7d`) and `read_email`.
Times are shown in `assistant.timezone`.

> **Token expiry in Testing mode:** while the consent screen is in *Testing*, Google
> expires the refresh token after **7 days**. When ROCO says the Google sign-in has
> expired (or `roco doctor` flags the token), run `roco auth google` again.
> Publishing the app *In production* removes the limit, but Google then shows an
> "unverified app" screen for the Gmail scope; for a personal assistant, signing in
> again weekly is the simpler option.

To disconnect: delete `~/.roco/google_token.json` and remove ROCO's access at
<https://myaccount.google.com/permissions>.

## Outlook setup (Hotmail / Outlook.com email)

If your email is a Microsoft account (`@hotmail.com`, `@outlook.com`, `@live.com`), ROCO
can read it instead of Gmail, through Microsoft Graph. It is read-only (`Mail.Read`):
ROCO never sends, deletes or changes email. Google Calendar keeps working alongside it.

1. **Register an app.** Open <https://entra.microsoft.com> (or
   <https://portal.azure.com>), sign in with your Microsoft account, then go to
   *Applications > App registrations > New registration*:
   - Name: `ROCO`
   - Supported account types: **Personal Microsoft accounts only**
   - Redirect URI: leave empty

   If the portal says you need a directory or an Azure subscription first, sign up for
   the free Azure account it offers, then come back to this step.
2. **Copy the Application (client) ID** from the app's *Overview* page. There is no
   client secret: ROCO is a public client.
3. **Allow the device code sign-in.** *Authentication* (or *Authentication > Settings*):
   set **Allow public client flows** to **Yes** and save.
4. **Add the permission.** *API permissions > Add a permission > Microsoft Graph >
   Delegated permissions*, tick **Mail.Read**, then *Add permissions*.
5. **Configure ROCO** in `config.yaml`:
   ```yaml
   email:
     provider: outlook
   outlook:
     client_id: "paste-the-application-client-id-here"
   ```
6. **Sign in:**
   ```powershell
   pip install -e ".[outlook]"
   roco auth outlook
   ```
   ROCO prints a code. Open <https://microsoft.com/devicelogin> in any browser, enter it,
   sign in with your Hotmail address and accept "Read your mail". The token is saved to
   `~/.roco/outlook_token.json`.

   If Microsoft's page refuses the code (for example "The application is a first party
   application... users are not permitted to consent"), use the browser sign-in instead:
   in the app registration open *Authentication > Add a platform > Mobile and desktop
   applications*, add the redirect URI `http://localhost`, save, then run
   `roco auth outlook --browser`.
7. **Check:** `roco doctor` should show `Outlook sign-in` as OK. Then ask `roco chat`:
   "any unread emails?".

The email tools have the same names as with Gmail (`get_unread_emails`, `search_emails`,
`read_email`); `email.provider` picks the account. Email ids look like `o1`, `o2`
(short aliases ROCO maps back to Outlook's long ids). Searches take plain words or
`from:alice`, `subject:invoice`.

If ROCO says the Outlook sign-in has expired (for example after a long time unused, or
after you change your Microsoft password), run `roco auth outlook` again. To disconnect:
delete `~/.roco/outlook_token.json` and remove ROCO at
<https://account.live.com/consent/Manage>.

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
