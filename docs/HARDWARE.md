# Hardware: the v1 body

## Split: brain on the PC, body on the Pi

A Raspberry Pi 5 can run wake word, speech-to-text (`base.en`), Piper and the face
comfortably, but a 7B model on its CPU manages only a couple of tokens per second, far too
slow for conversation. So in v1:

- **Pi 5 (body):** runs `roco run`: wake word, recording, faster-whisper, Piper, pygame face
  in fullscreen, servos via PCA9685.
- **PC (brain):** runs Ollama only. Set `OLLAMA_HOST=0.0.0.0` on the PC so it listens on the
  LAN, and point the Pi's `llm.host` at it (e.g. `http://192.168.1.20:11434`). Give the PC a
  DHCP reservation so the address doesn't change.

## Parts list

| Part | Suggestion | Notes |
|---|---|---|
| Computer | Raspberry Pi 5, 4 GB or 8 GB | 8 GB leaves room for `small.en` STT later |
| Power | Official Raspberry Pi 27 W USB-C PSU | Needed so USB devices get full current |
| Cooling | Official Active Cooler | Whisper keeps the CPU busy |
| Storage | 32 GB+ A2 microSD (or NVMe HAT) | |
| Microphone | ReSpeaker USB Mic Array v2.0 (or Lite) | Far-field, on-board AEC/beamforming helps barge-in later |
| Speaker | Small USB speaker, or MAX98357A I2S amp + 3 W 4 Ω speaker | I2S is tidier inside an enclosure |
| Display | 5" DSI or HDMI display, 800×480 | Matches the default face size; set `face.fullscreen: true` |
| Servo driver | PCA9685 16-channel PWM board (I2C) | |
| Servos | 2 × MG90S (metal gear) | Pan + tilt |
| Bracket | Mini pan/tilt bracket for SG90/MG90S | |
| Servo power | Separate 5 V 2–3 A supply (or buck converter) | Never power servos from the Pi's 5 V pin |

## Wiring

### PCA9685 → Pi 5 (I2C)

| PCA9685 | Pi 5 header |
|---|---|
| VCC (logic) | Pin 1 (3.3 V) |
| GND | Pin 6 (GND) |
| SDA | Pin 3 (GPIO2 / SDA1) |
| SCL | Pin 5 (GPIO3 / SCL1) |
| V+ (servo power terminal) | External 5 V supply + |
| GND (terminal) | External 5 V supply − **and** common with Pi GND |

Servos: pan on channel 0, tilt on channel 1 (configurable under `servos.pan/tilt.channel`).
Enable I2C with `sudo raspi-config` → Interface Options → I2C, then check the board shows up
at `0x40` with `i2cdetect -y 1`. Install the `pi` extra: `pip install -e ".[pi]"` and set
`servos.backend: pca9685`.

Start with conservative limits (`min`/`max` in config) and widen them once you've seen the
bracket's real range: MG90S servos will happily strip a bracket that hits an end stop.
Add a 470–1000 µF capacitor across the servo supply to absorb current spikes.

### MAX98357A I2S amp (optional, instead of USB speaker)

| MAX98357A | Pi 5 header |
|---|---|
| VIN | Pin 2 (5 V) |
| GND | Pin 9 (GND) |
| BCLK | Pin 12 (GPIO18) |
| LRC | Pin 35 (GPIO19) |
| DIN | Pin 40 (GPIO21) |

Add `dtoverlay=hifiberry-dac` (or `max98357a`) to `/boot/firmware/config.txt` and reboot.

### Display

DSI: ribbon cable to a DSI port, it powers from the Pi. HDMI: micro-HDMI port 0, power the
display from its own USB. Disable screen blanking (`raspi-config` → Display Options) so the
face stays on.

## Software on the Pi

Raspberry Pi OS (64-bit, Bookworm), Python 3.11, then
`pip install -e ".[voice,face,pi]"`, `roco download-models`, `roco doctor`, `roco run`.
On Linux openWakeWord can also use tflite, but ONNX works everywhere and keeps one code path.
