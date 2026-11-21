# Hardware: the v1 body

The product spec ([`ROCO_SPEC.md`](ROCO_SPEC.md) §4) is the source of truth; this page is
the shopping list, the wiring and the board notes. Setting the Pi up is in
[`PI_SETUP.md`](PI_SETUP.md).

## Split: brain on the PC, body on the Pi

The Pi 5 (2 GB) is the **body only** (spec §9). A 7B model on a Pi CPU manages a couple
of tokens per second and does not fit next to everything else in 2 GB, so:

- **PC (brain):** `roco brain`: speech-to-text, Ollama `qwen2.5:7b`, TTS, tools, memory.
- **Pi 5 (body):** `roco body` as a systemd service: mic, wake word, VAD, echo
  cancellation, playback, face, servos. Talks to the brain over one WebSocket on port 8765
  ([`PROTOCOL.md`](PROTOCOL.md)), on the home LAN or over Tailscale when away.
- **Fallback (on the Pi):** when the PC brain is unreachable the body transcribes with
  whisper `tiny.en` and asks Claude directly (`claude-haiku-5-5`); with no internet it says a
  canned "my brain's offline" line in its own local voice (`roco.fallback`).

## Parts

### Ordered (Round 1): The Pi Hut "Raspberry Pi 5 Starter Kit", 2 GB

| Part | In the kit | Notes |
|---|---|---|
| Computer | Raspberry Pi 5, 2 GB | Body only, see above |
| Power | Official Raspberry Pi 27 W USB-C PSU | Full current for the USB mic/speaker/webcam |
| Case + cooling | Official Pi 5 case with built-in fan | **The lid lifts off** to reach the 40-pin GPIO header (needed for the PCA9685 in Round 2); the fan stays on its connector |
| Storage | Official 64 GB microSD, Raspberry Pi OS preinstalled | See [`PI_SETUP.md`](PI_SETUP.md) step 1 to make it headless |
| Display cable | micro-HDMI to HDMI | For the Round 2 screen (use the port next to the USB-C power, HDMI0) |

That is enough for Phase 3: the Pi runs the body client headless (no screen, no servos
yet) and can be tested with any USB headset or USB mic + speaker you already have.

### Round 2 shopping list (buy before Phase 4)

| Part | Suggestion | Approx. | Why |
|---|---|---|---|
| Screen | 4–5" HDMI display, 800×480, USB-powered | £20–35 | Upward-facing, reflected by the acrylic = hologram head |
| Camera | USB webcam, 720p+ | £10–20 | Face/hand tracking on the Pi |
| Microphone | USB mic (small omni/boundary mic) | £8–15 | Wake word + speech; AEC is software (`roco.voice.aec`) |
| Speaker | Small USB-powered speaker | £8–15 | ROCO's voice |
| Servo driver | PCA9685 16-channel PWM board (I2C, 0x40) | £5–8 | Drives the pan/tilt servos |
| Servos | 2 × MG90S (metal gear) | £8 | Pan + tilt |
| Bracket | Mini pan/tilt bracket for SG90/MG90S | £4 | The neck |
| Servo power | Separate 5 V, 2–3 A supply (barrel/USB) + female DC jack or screw terminals | £6–10 | **Never** power servos from the Pi |
| Capacitor | 470–1000 µF, ≥10 V electrolytic | £1 | Across the servo supply, absorbs current spikes |
| Wires | Female–female jumper wires (4 for I2C), a few male–male | £3 | Pi ↔ PCA9685 |
| Hologram | Clear acrylic sheet (cheap photo-frame glass replacement) | £3–5 | Pepper's ghost at 45° |
| Enclosure | Black foam board + glue/tape | £5 | Dark box with an open front |

**Later upgrades** (spec §4; config changes only): ReSpeaker USB Mic Array v2.0 (hardware
AEC + voice direction, first upgrade), Pi AI Camera, a bigger Pi (8 GB: on-Pi backup LLM),
a DSI display, touch (MPR121 + copper tape), knock sensor (piezo), poke detection
(VL53L1X), mute switch + LED.

## Wiring (Round 2)

Power everything **off** while wiring. Lift the case lid to reach the GPIO header; pin 1 is
the corner pin nearest the microSD-card end of the board, marked by the square pad.

### PCA9685 → Pi 5 (I2C bus 1)

| PCA9685 pin | Pi 5 header pin |
|---|---|
| VCC (logic power) | Pin 1 (3.3 V) |
| GND | Pin 6 (GND) |
| SDA | Pin 3 (GPIO2 / SDA1) |
| SCL | Pin 5 (GPIO3 / SCL1) |
| OE | not connected |

### Servo power (separate supply, common ground)

```
 5 V 2-3 A supply (+) ──────┬──────────────── PCA9685 V+ terminal
                            │
                          [470-1000 µF]   long leg / + side to +5 V
                            │
 5 V 2-3 A supply (−) ──────┴──────┬───────── PCA9685 GND terminal
                                   │
                                   └───────── Pi GND (e.g. pin 9)   ← common ground
 MG90S pan  → PCA9685 channel 0  (brown = GND, red = V+, orange = PWM)
 MG90S tilt → PCA9685 channel 1
```

- The servos take their current from the **V+ terminal** (separate supply), never from
  the Pi's 5 V pins: a stalled MG90S draws ~0.7 A and would brown out the Pi.
- **Common ground**: the servo supply's − must be joined to a Pi GND, or the PWM signal has
  no reference and the servos jitter.
- The capacitor goes across V+ and GND at the PCA9685 terminal; mind the polarity.
- Channels are configurable under `servos.pan/tilt.channel`.

### Checks once wired

```bash
i2cdetect -y 1          # the board shows up as 40
```

Then in `~/.roco/config.yaml` set `servos.backend: pca9685` and restart the service. Start
with the conservative `min`/`max` limits from the default config and widen them only once
you've seen the bracket's real range: MG90S servos will happily strip a bracket that hits
an end stop. `pi_setup.sh` already enabled I2C and installed the `pi` extra.

### Screen, camera, mic, speaker

- HDMI screen: micro-HDMI port HDMI0, power from its own USB lead. In the config:
  `face.enabled: true`, `face.fullscreen: true`. Disable screen blanking
  (`sudo raspi-config nonint do_blanking 1`).
- USB webcam, mic and speaker: any USB port (the blue USB 3 ports for the webcam). Pick the
  devices with `voice.input_device` / `voice.output_device` (a name substring) after
  checking them with `python -m roco.voice.diag devices` on the Pi.

## Software on the Pi

Raspberry Pi OS Bookworm 64-bit, Python 3.11, everything installed by
[`scripts/pi_setup.sh`](../scripts/pi_setup.sh): `pip install -e ".[body,pi,cloud]"`, the
wake-word/VAD/Piper/whisper `tiny.en` models, I2C, and the `roco-body` systemd service.
On Linux openWakeWord can also use tflite, but ONNX works everywhere and keeps one code path.
