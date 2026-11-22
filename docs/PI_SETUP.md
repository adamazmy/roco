# Pi setup (headless) and Tailscale

Goal: Adam does **only the physical steps** (card, power, network, two sign-ins in a
browser); an agent does everything else over SSH. Every command Adam types is one line of
PowerShell on the PC. Parts and wiring: [`HARDWARE.md`](HARDWARE.md).

```
 PC (Windows)                    Tailscale / home LAN             Pi 5 (roco.local)
 roco brain  :8765  ◀──────── ws://<pc-tailnet-name>:8765 ──────── roco body (systemd)
 ~/.roco/link_token  ── copied once (scp) ─────────────────────▶   ~/.roco/link_token
                                                                   ~/.roco/anthropic_key
```

## Adam's part (about 15 minutes)

### 1. Make the card headless (Raspberry Pi Imager)

The kit's card has Raspberry Pi OS preinstalled, but its first boot asks for a screen and
keyboard, and SSH is off. The simplest way round it is to **re-flash it with presets**:
Imager writes Wi-Fi, SSH and the user name onto the card, so the Pi joins the network by
itself on first boot. You need a microSD reader on the PC (a cheap USB one is fine).

1. Install Imager (PowerShell):
   `winget install --id RaspberryPiFoundation.RaspberryPiImager -e`
2. Open **Raspberry Pi Imager**, card in the reader:
   - Device: **Raspberry Pi 5**
   - OS: **Raspberry Pi OS (other) → Raspberry Pi OS Lite (64-bit)** (no desktop: saves
     ~300 MB of the 2 GB; the face still works full-screen later)
   - Storage: the microSD card
3. **Next → Edit settings** (the OS customisation screen):
   - General: hostname **`roco`**, username **`roco`** + a password you'll remember,
     Wi-Fi name + password + country **GB**, locale Europe/London
   - Services: **Enable SSH**, "Use password authentication"
4. **Save → Yes** and let it write and verify (this erases the card).

No card reader? Plan B: plug the Pi into a TV/monitor (micro-HDMI cable from the kit) and
a USB keyboard, finish the first-boot wizard (user `roco`, your Wi-Fi), then run
`sudo raspi-config nonint do_ssh 0` and unplug the screen.

### 2. Boot it

Card into the Pi (slot underneath, contacts up), case closed, plug in the official 27 W
PSU. Wait two minutes (the first boot resizes the card and reboots once). An Ethernet cable
to the router also works and needs no Wi-Fi settings.

### 3. Check it's on the network

`ping roco.local`

If that fails, find `roco` in your router's device list and use its IP instead of
`roco.local` below.

### 4. Hand it to the agent

Tell the agent the Pi is up and the password you set. The agent saves the Pi as an SSH
connection in AgentsRoom (so the password stays in the app, not in a chat), switches it to
key login, and does everything below. Two moments need you in a browser: the Tailscale
sign-in link (step 7) and, if you want the cloud fallback, the Anthropic key (step 8).

## The agent's part (over SSH)

### 5. Code + setup script

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/adamazmy/roco.git ~/roco
~/roco/scripts/pi_setup.sh --brain-url ws://<pc-tailnet-name>:8765
```

`pi_setup.sh` is idempotent (safe to re-run after every `git pull`; `--dry-run` shows what
it would do). It installs the apt packages (PortAudio, ALSA tools, i2c-tools, SDL2),
enables I2C, creates `~/roco/.venv`, runs `pip install -e ".[body,pi,cloud]"`, writes
`~/.roco/config.yaml` once (brain URL, `face.enabled: false` until the screen arrives,
`voice.stt.model: tiny.en`), downloads the wake word, VAD, Piper voice and whisper
`tiny.en`, pre-renders the "my brain's offline" line to a WAV (`roco fallback-prepare`;
the body plays it if its own TTS ever fails), and installs the **`roco-body` systemd user service** (starts at boot via
`loginctl enable-linger`, restarts on any failure).

### 6. Link token

The PC brain creates `~/.roco/link_token` on its first `roco brain`. Copy it to the Pi
from the PC (one line, PowerShell):

`scp $env:USERPROFILE\.roco\link_token roco@roco.local:.roco/link_token`

then on the Pi: `chmod 600 ~/.roco/link_token && systemctl --user restart roco-body`.
The token is a secret: it is never committed, never pasted into chats.

### 7. Tailscale on the Pi

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname=roco
```

`tailscale up` prints a login URL: Adam opens it and approves the device in the same
tailnet as the PC. Check with `tailscale status` (the PC must be listed).

### 8. Anthropic key (cloud fallback)

Optional: without it the fallback only says "my brain's offline". On the Pi:

```bash
~/roco/.venv/bin/roco cloud-key      # hidden prompt; Adam pastes the key himself
```

It is stored in `~/.roco/anthropic_key` with `chmod 600`, never in the repo or config
(`.gitignore` also blocks `anthropic_key`). `ANTHROPIC_API_KEY` in the environment works
too. Every cloud call is logged with its tokens and estimated cost:
`~/roco/.venv/bin/roco cloud-usage` (or `--by day`).

Two things stop the Pi from calling Claude even with a key:

- **Private mode.** "ROCO, local only" / "go private" on the PC brain also covers the
  Pi. The brain sends the mode to the body, and the Pi saves the last value in
  `~/.roco/fallback_state.json`, so it still applies if the Pi boots while the PC is off.
  With private mode on and the PC down, ROCO says once that she won't use the cloud, then
  only the offline line. `brain.local_only: true` in the Pi's config forces it on.
- **Budget caps** (`cloud.budget`: £0.20 a day, £5 a month by default). **The caps are
  per machine.** The Pi counts only its own `~/.roco/cloud_usage.jsonl` (its fallback
  calls), and the PC counts its own, so with both at the defaults the worst case is twice
  the caps. Lower `cloud.budget` in the Pi's config if you want a tighter combined limit.
  Past a cap, ROCO says so once, then only the offline line until local midnight (daily)
  or the 1st (monthly).

### 9. Check

```bash
systemctl --user status roco-body
journalctl --user -u roco-body -f       # live log: link up, wake word, turns
python -m roco.voice.diag devices        # (inside the venv) USB mic/speaker names
```

Updating later: `cd ~/roco && git pull && scripts/pi_setup.sh`.

## Tailscale: brain on the PC, body anywhere

Adam already has Tailscale on the PC. With the Pi in the same tailnet, **always use the PC's
tailnet name** in `link.brain_url`: at home Tailscale takes the direct LAN path (no extra
latency worth noticing), and away from home it still works with no config change.

1. The PC's tailnet name and IP (PowerShell on the PC):
   `tailscale status`
   The first line is the PC itself: its IP (`100.x.y.z`), then its name (e.g. `adam-pc`).
2. On the Pi, in `~/.roco/config.yaml`:
   ```yaml
   link:
     brain_url: ws://adam-pc:8765        # MagicDNS name, or ws://100.x.y.z:8765
   ```
   then `systemctl --user restart roco-body`.
3. The brain listens on `0.0.0.0:8765` by default (`link.host`), which includes the
   Tailscale interface. Allow the port through Windows Firewall, for the home LAN and the
   tailnet only (PowerShell **as Administrator**, one line):

   `New-NetFirewallRule -DisplayName "ROCO brain (8765)" -Direction Inbound -Protocol TCP -LocalPort 8765 -RemoteAddress LocalSubnet,100.64.0.0/10 -Action Allow`

4. Test from the Pi while `roco brain` runs:
   `timeout 3 bash -c '</dev/tcp/adam-pc/8765' && echo open`

Security: the link is plain `ws://`, which is fine because Tailscale encrypts everything
end to end (WireGuard), and on the home LAN every body must still present the link token.
Don't forward port 8765 on the router.

## Troubleshooting

| Symptom | Check |
|---|---|
| `ping roco.local` fails | Router device list for the IP; Wi-Fi country/password in Imager; try Ethernet |
| Service restarts in a loop | `journalctl --user -u roco-body -n 50`; usually a missing USB audio device or `link_token` |
| "auth" error in the log | `link_token` on the Pi differs from the PC's: copy it again (step 6) |
| ROCO says it's on the cloud brain at home | The PC brain is down or the firewall rule is missing (step 3 above) |
| "my brain's offline" | No internet and no PC brain, or no Anthropic key: `roco cloud-key --show` |
| `i2cdetect` shows nothing | Reboot after the first setup (I2C), check SDA/SCL and the common ground |
