#!/usr/bin/env bash
# ROCO body setup for a Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
#
# Idempotent: run it again after a `git pull` and it only does what is missing.
# Run it as the normal user (e.g. `pi`), from the repository checkout; it uses sudo for
# apt and I2C. See docs/PI_SETUP.md.
#
#   scripts/pi_setup.sh --brain-url ws://adam-pc:8765      # first time
#   scripts/pi_setup.sh                                    # later: update deps + restart
#   scripts/pi_setup.sh --dry-run                          # print what it would do
#
# Options:
#   --brain-url URL   where the PC brain listens (written to ~/.roco/config.yaml once)
#   --dry-run         print the commands instead of running them
#   --no-apt          skip apt packages        --no-models   skip model downloads
#   --no-service      skip the systemd unit    --no-i2c      skip enabling I2C
#   -h, --help        this help

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$REPO/.venv"
ROCO_HOME="${ROCO_HOME:-$HOME/.roco}"
UNIT_NAME="roco-body.service"
UNIT_DIR="$HOME/.config/systemd/user"
EXTRAS="body,pi,cloud"
USER="${USER:-$(id -un)}"

DRY_RUN=0
DO_APT=1
DO_MODELS=1
DO_SERVICE=1
DO_I2C=1
BRAIN_URL=""

APT_PACKAGES=(
  python3-venv python3-dev python3-pip git
  libportaudio2 portaudio19-dev alsa-utils    # sounddevice (mic + speaker), arecord/aplay
  i2c-tools                                   # i2cdetect for the PCA9685 (Round 2)
  libsdl2-2.0-0 libsdl2-ttf-2.0-0 libsdl2-image-2.0-0   # pygame face (Round 2 screen)
  libopenblas0                                # numpy / onnxruntime on aarch64
)

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --brain-url) BRAIN_URL="${2:?--brain-url needs a value}"; shift 2 ;;
    --brain-url=*) BRAIN_URL="${1#*=}"; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-apt) DO_APT=0; shift ;;
    --no-models) DO_MODELS=0; shift ;;
    --no-service) DO_SERVICE=0; shift ;;
    --no-i2c) DO_I2C=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

step() { printf '\n==> %s\n' "$*"; }
note() { printf '    %s\n' "$*"; }
run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf '    [dry-run] %s\n' "$*"
  else
    "$@"
  fi
}

# ------------------------------------------------------------------------ checks
step "Checks"
if [[ $EUID -eq 0 ]]; then
  echo "Run this as your normal user (it calls sudo itself), not as root." >&2
  exit 1
fi
if [[ "$(uname -s)" != "Linux" && $DRY_RUN -eq 0 ]]; then
  echo "This script sets up a Raspberry Pi (Linux). Use --dry-run to preview elsewhere." >&2
  exit 1
fi
MODEL="$(tr -d '\0' 2>/dev/null </proc/device-tree/model || echo unknown)"
note "Board: $MODEL"
note "Repo:  $REPO"
note "Venv:  $VENV"
if [[ -n "$BRAIN_URL" && ! "$BRAIN_URL" =~ ^wss?:// ]]; then
  echo "--brain-url must start with ws:// (e.g. ws://adam-pc:8765)" >&2
  exit 2
fi

# ------------------------------------------------------------------------ apt
if [[ $DO_APT -eq 1 ]]; then
  step "System packages"
  missing=()
  for pkg in "${APT_PACKAGES[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [[ ${#missing[@]} -eq 0 ]]; then
    note "all present"
  else
    note "installing: ${missing[*]}"
    run sudo apt-get update -q
    run sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q "${missing[@]}"
  fi
fi

# ------------------------------------------------------------------------ I2C + groups
if [[ $DO_I2C -eq 1 ]]; then
  step "I2C (PCA9685 servo board) and device groups"
  if command -v raspi-config >/dev/null 2>&1; then
    if [[ "$(sudo raspi-config nonint get_i2c 2>/dev/null || echo 1)" == "0" ]]; then
      note "I2C already enabled"
    else
      run sudo raspi-config nonint do_i2c 0
      note "I2C enabled (takes effect after a reboot if /dev/i2c-1 is missing)"
    fi
  else
    note "raspi-config not found: not a Raspberry Pi OS image? skipping I2C"
  fi
  for group in audio i2c gpio video; do
    if getent group "$group" >/dev/null 2>&1 && ! id -nG "$USER" | tr ' ' '\n' | grep -qx "$group"; then
      run sudo usermod -aG "$group" "$USER"
      note "added $USER to $group (applies at next login / reboot)"
    fi
  done
fi

# ------------------------------------------------------------------------ Python
step "Python venv + ROCO ($EXTRAS)"
if [[ ! -x "$VENV/bin/python" ]]; then
  run python3 -m venv "$VENV"
fi
run "$VENV/bin/python" -m pip install -q --upgrade pip wheel
run "$VENV/bin/python" -m pip install -q -e "${REPO}[${EXTRAS}]"
ROCO="$VENV/bin/roco"

# ------------------------------------------------------------------------ config
step "Config in $ROCO_HOME"
run mkdir -p "$ROCO_HOME"
run chmod 700 "$ROCO_HOME"
CONFIG="$ROCO_HOME/config.yaml"
if [[ -f "$CONFIG" ]]; then
  note "$CONFIG exists: left unchanged"
  if [[ -n "$BRAIN_URL" ]]; then
    note "to change the brain address, edit link.brain_url in it (wanted: $BRAIN_URL)"
  fi
else
  url="${BRAIN_URL:-ws://CHANGE-ME:8765}"
  note "writing $CONFIG (brain_url: $url)"
  if [[ $DRY_RUN -eq 0 ]]; then
    cat >"$CONFIG" <<EOF
# ROCO body config on the Pi (written by scripts/pi_setup.sh; edit freely).
# Only keys that differ from src/roco/config.py are needed.
link:
  # The PC brain: its LAN name/IP at home, or its Tailscale name/IP anywhere
  # (docs/PI_SETUP.md, "Tailscale").
  brain_url: $url
voice:
  stt:
    model: tiny.en        # the Pi only transcribes for the fallback brain
face:
  enabled: false          # true once the HDMI screen arrives (Round 2)
  fullscreen: true
servos:
  backend: simulated      # pca9685 once the servo board is wired (Round 2)
EOF
  fi
fi
for secret in link_token anthropic_key; do
  if [[ -f "$ROCO_HOME/$secret" ]]; then
    run chmod 600 "$ROCO_HOME/$secret"
  fi
done

# ------------------------------------------------------------------------ models
if [[ $DO_MODELS -eq 1 ]]; then
  step "Models (wake word, VAD, Piper voice, whisper tiny.en)"
  # Wake word + Piper voice (the body's own voice for fallback replies) + VAD + the
  # whisper model named in voice.stt.model (tiny.en above). Idempotent.
  run "$ROCO" download-models
  # The fallback's STT model (cloud.stt_model), in case voice.stt.model differs.
  run "$VENV/bin/python" -c \
    "from roco.config import load_config; from roco.fallback.stt import download_stt_model as d; raise SystemExit(0 if d(load_config()) else 1)"
fi

# ------------------------------------------------------------------------ service
if [[ $DO_SERVICE -eq 1 ]]; then
  step "systemd user service ($UNIT_NAME)"
  run mkdir -p "$UNIT_DIR"
  if [[ $DRY_RUN -eq 1 ]]; then
    note "[dry-run] write $UNIT_DIR/$UNIT_NAME from scripts/roco-body.service"
  else
    sed -e "s|@REPO@|$REPO|g" -e "s|@VENV@|$VENV|g" \
      "$REPO/scripts/roco-body.service" >"$UNIT_DIR/$UNIT_NAME.new"
    if cmp -s "$UNIT_DIR/$UNIT_NAME.new" "$UNIT_DIR/$UNIT_NAME" 2>/dev/null; then
      rm -f "$UNIT_DIR/$UNIT_NAME.new"
      note "unit unchanged"
    else
      mv "$UNIT_DIR/$UNIT_NAME.new" "$UNIT_DIR/$UNIT_NAME"
      note "unit installed"
    fi
  fi
  # Start at boot without anyone logging in.
  if [[ "$(loginctl show-user "$USER" -p Linger --value 2>/dev/null || echo no)" != "yes" ]]; then
    run sudo loginctl enable-linger "$USER"
  fi
  run systemctl --user daemon-reload
  run systemctl --user enable "$UNIT_NAME"
  if [[ -f "$ROCO_HOME/link_token" ]]; then
    run systemctl --user restart "$UNIT_NAME"
    note "started; logs: journalctl --user -u roco-body -f"
  else
    note "not started yet: copy link_token from the PC first (docs/PI_SETUP.md, step 6),"
    note "then: systemctl --user restart roco-body"
  fi
fi

# ------------------------------------------------------------------------ summary
step "Summary"
if [[ -f "$ROCO_HOME/link_token" ]]; then
  note "link token:     present"
else
  note "link token:     MISSING (copy ~/.roco/link_token from the PC)"
fi
if [[ -f "$ROCO_HOME/anthropic_key" || -n "${ANTHROPIC_API_KEY:-}" ]]; then
  note "Anthropic key:  present (cloud fallback on)"
else
  note "Anthropic key:  missing (fallback says the offline line only; run: roco cloud-key)"
fi
if grep -q "CHANGE-ME" "$CONFIG" 2>/dev/null; then
  note "brain_url:      NOT SET (edit $CONFIG or re-run with --brain-url)"
fi
note "Done. Re-run any time; it is safe."
