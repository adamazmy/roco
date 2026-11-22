# Echo cancellation (AEC) and barge-in

ROCO keeps listening while it talks, so Adam can interrupt it just by speaking
(spec §5 "stops instantly when interrupted", §11 barge-in stop time < 200 ms). The mic
then also hears ROCO's own voice from the speaker. The body's echo canceller removes
it using a copy of what was played, so only Adam's voice reaches the VAD.

**Decision: WebRTC AudioProcessingModule (AEC3) through the `livekit` wheel.**
SpeexDSP (`pyaec`) stays available as `voice.aec.engine: speex`; `none` disables AEC.
`auto` (the default) tries WebRTC, then Speex, then none.

## How it is wired

```
TTS frames ─▶ Player ─pull()─▶ PortAudio callback ─▶ speaker
                                    │ every block actually played
                                    ▼
                              EchoReference (FIFO, resampled to 16 kHz)
                                    │ same number of samples as captured
mic ─▶ MicStream ─▶ AecFrameSource ─┴─▶ AEC3 (10 ms chunks) ─▶ wake word + Silero VAD
                                    └─▶ EchoMonitor (ERLE while ROCO speaks)
```

- The output stream runs for the whole session (silence when idle), so the reference
  is a continuous stream that stays sample-aligned with the mic up to a constant
  delay, which AEC3 estimates itself.
- The engine is injected (`roco.voice.aec.EchoCanceller` protocol); CI runs with fakes
  and `NullAec`, and no AEC library is needed to import anything.
- Safety net: `EchoMonitor` measures ERLE over each reply. Below `voice.aec.min_erle_db`
  (10 dB), or after two barge-ins in a row with no words in them (empty brain
  transcript), the body switches to **wake-word-only barge-in** and prints so. With no
  AEC installed it starts in that mode. The current mode is in `hello.caps.barge_in`.

## Candidates

| | WebRTC AEC3 (`livekit`) | SpeexDSP (`pyaec` → aec-rs) | others |
|---|---|---|---|
| Windows x64 wheel | yes (livekit 1.3.0) | yes (pyaec 1.0.1) | `aec-audio-processing`: Windows only |
| Linux aarch64 (Pi 5) wheel | yes, `manylinux_2_28_aarch64` | yes, `manylinux_2_28_aarch64` | `speexdsp`, `webrtc-audio-processing`: no wheels |
| Delay handling | estimates the delay itself (up to ~0.5 s) | filter must cover delay + tail | |
| Non-linear residual | suppressor on top of the linear filter | preprocessor, weaker | |
| Install size | ~11 MB wheel (24 MB FFI library) + protobuf | 82 KB | |

## Offline evaluation (synthetic laptop echo path)

Script: far end = Piper (alba) monologue, near end = Piper (alan) "Wait, stop, what time
is the dentist?" overlapping the far end. Echo path = buffering delay + 150 ms room
impulse response (RT60 ≈ 0.25 s) + speaker soft clipping + mic noise; echo louder than
the near-end voice (laptop speaker next to the mic). 80 ms mic frames, as the body uses.

| Engine | Delay | ERLE (single talk) | Frames with residual > speech level | False barge-ins on echo alone (Silero) | CPU (ms per s of audio, PC) |
|---|---|---|---|---|---|
| none | any | 0 dB | 76 % | constant | 0 |
| Speex, 300 ms filter | 40 / 120 / 250 ms | 26.1 / 27.4 / 18.9 dB | 11–25 % | 15–16 in 10 s | 27 |
| Speex, 500 ms filter | 40 / 120 / 250 ms | 19.5 / 18.4 / 14.9 dB | 29–32 % | | 36 |
| **WebRTC AEC3** | 40 / 120 / 250 ms | **38.5 / 35.7 / 38.0 dB** | **0–1.7 %** | **0** | **6** |

Double talk: AEC3 attenuates the near-end voice by 4–11 dB while the echo dominates.
Adam's first interruption was detected 0.3 s after its onset (including Piper's
leading silence), but late in a long reply with a very loud echo, detection took up to
~1.3 s in this synthetic test. The hardware test below did not show that.

CPU on the Pi 5: AEC3 costs ~0.6 % of one PC core; the Pi's Cortex-A76 is roughly 3–4x
slower per core, so expect ~2–3 % of one core (still to be measured on the Pi).

## Hardware test (Adam's laptop, Windows MME, built-in mic + speakers, 2026-11-21)

`roco body` against a stub brain (Piper cori replies of ~25 s), normal desk distance
and volume.

| Metric | Result |
|---|---|
| ERLE while ROCO speaks (median per reply) | 20.8–30.3 dB (6 replies) |
| Self-interruption while Adam stayed silent | none (4 full replies) |
| Playback cut, decision → audio stopped (`Player.cut` + stream abort) | 0.0 ms (below timer resolution) |
| Speech onset → silence, first run (160 ms confirmation, ~100 ms output buffer) | 260–356 ms |
| Speech onset → silence, defaults now (96 ms confirmation, 50 ms output buffer) | **156–220 ms** (4 barge-ins) |
| Body RSS at start / peak (face, wake word, VAD, AEC; fallback not loaded) | 202 / 251 MB |

"Speech onset → silence" = speech the VAD needed before deciding + cut time + the output
device latency PortAudio reports. It does not include the mic capture buffer. The 200 ms
target is met at the low end and missed by up to 20 ms at the high end, set by the
80 ms mic frame granularity. Knobs: `voice.barge_in.start_ms` (speech needed) and
`voice.playback.latency` (output buffer; raise it if the voice crackles).

## Spoken test on the integrated build (2026-11-22): echo self-interrupts

`roco run` on Adam's laptop (built-in mic + speakers, fans on) was unusable: 4 false
barge-ins in 20 s. ROCO's own voice leaked through the AEC (ERLE 19.7-31.6 dB), the VAD
called it speech, and the brain answered the garbled echo as if Adam had said it:
"on Sunday" came back as "on the song", "just gone" as "Just go", "half past three" as
"House three", and also "I've passed". Real interruptions also lost their first words
("Stop, do I have any unread emails?" → "drop any hundred emails").

**Correction to the table above:** "speech onset → silence" there was the speech run the
VAD counted before deciding, plus the cut and the device latency. When AEC3 suppresses
Adam's onset, the VAD starts counting late, so the true time from his first syllable is
longer. The bench below measures from the true onset.

### What changed (branch `phase3/b-echo-guard`)

| Layer | Behaviour | Config |
|---|---|---|
| Double-talk gate | While ROCO plays (+320 ms hangover), a frame only counts toward barge-in when the AEC output is ≥ 3 dB above the residual echo level learnt over the last ~3 s (`EchoMonitor.near_end`). | `voice.barge_in.echo_margin_db` |
| Confirmation | 128 ms of speech instead of 96 ms. | `voice.barge_in.start_ms` |
| Echo guard | The brain sends the barge-in turn's transcript before it answers. The body compares it with what ROCO was saying (the 2 sentences heard before the cut, the one playing, the next). On a phonetic match it sends `interrupt{turn, reason:"echo"}`: the brain drops the turn without answering. Interrupt, question and command words ("stop", "wait", "what", "turn"…) are always treated as Adam. | `voice.barge_in.echo_guard` |
| Auto-fallback | 2 echo matches within 30 s, 2 barge-ins in a row with empty transcripts, ERLE < 10 dB, or no AEC installed → `name` mode (only the wake word interrupts). Printed once. | `voice.aec.min_erle_db` |
| Pre-roll | Barge-in turns send 800 ms of audio from before the decision, follow-up turns too. Wake turns send 1.2 s, so whisper hears the whole "Hey Jarvis" (which the brain strips) instead of a fragment that garbles the question ("…vis go into" → "of this"). | `voice.barge_in.preroll_ms`, `voice.wake_word.preroll_ms` |
| Modes | `speech` (default) · `name` · `off`; `wake_word` is still accepted for `name`. | `voice.barge_in.mode` |

ROCO does not resume an utterance after an echo barge-in: by then the brain has cancelled
it, so it stops and listens (follow-up window). The echo turn's user text stays in the
brain's history unless the server drops `reason: "echo"` turns (optional follow-up for A).

### Bench: synthetic laptop echo with real cuts

Scripts: playback that stops on barge-in and resumes after Adam's phrase. Echo path: hard
speaker distortion + rattle, small-room IR, ±2.6 dB gain drift, three delay jumps
(MME buffer glitches), fan noise. Adam (Piper alan) interrupts 5 times per minute;
whisper base.en transcribes the barge-in turn as the brain would.

- **Echo quieter than Adam** (speaker turned away or further from the mic, echo gain
  0.1): every real interruption caught (5/5). ~1-2 false triggers per minute remain, all
  short. 800 ms pre-roll keeps the first words ("Stop. Do I have any unread emails?";
  with 480 ms / 160 ms: "Do I have any unread emails?" / "e-mail.").
- **Echo as loud as Adam or louder** (laptop speakers next to the mic): AEC3 suppresses
  Adam during double talk, so only 2-3 of 5 interruptions are detected, 0.5-2 s late.
  Requiring 160 ms of speech made that worse (later decisions, clipped words) without
  fewer false triggers, hence 128 ms. A raw-mic energy double-talk detector was unusable
  (29-144 false triggers/min). The wake word over that echo was not detected either
  (0-2/5, on the raw mic as well as the AEC output).
- Overall (6 runs, 6.5 min of ROCO talking, gains 0.25/0.5/1.0): false barge-ins
  0.6 → 0.3 per minute, answered-as-Adam 0.3 → 0.2 per minute, hits 10 → 11 of 30.
  The synthetic path leaks less than Adam's laptop did (4 in 20 s), so these absolute
  rates are optimistic. The echo guard is tested on the exact transcripts from the
  spoken test instead (`tests/test_client_echo_guard.py`): all 4 are caught, and 30
  real-interruption phrases are kept.
- Time from Adam's first syllable to the decision: ~0.2-0.3 s with a quiet echo, plus
  ~60 ms cut + output buffer. Under a loud echo it is whenever AEC3 lets him through.

### Recommendation for the Pi (USB speaker + USB mic)

1. **Separate them physically**: speaker pointing away from the mic, as far as the
   enclosure allows (≥ 20 cm, ideally the mic in front and the speaker behind or below).
   "Any speech" barge-in works when ROCO's echo at the mic is clearly quieter than Adam's
   voice. That is a geometry problem, not a software one.
2. Keep the playback volume moderate. The echo level grows with it and Adam's doesn't.
3. If the echo stays loud, set `voice.barge_in.mode: name`. The body also switches by
   itself after 2 echo barge-ins in 30 s, and says so on the console.
4. The real fix is the ReSpeaker USB Mic Array (spec §4: hardware AEC + beamforming
   towards Adam). Then set `voice.aec.engine: none` and keep `mode: speech`.
5. Measure on the Pi first: run `roco body` with ROCO reading a long story, stay silent
   for a minute, count "Barge-in on turn" lines (target 0), then interrupt 10 times.

## Open points

- Measure AEC3 CPU and ERLE on the Pi 5 with the USB speaker + USB mic (different
  echo path, and a separate playback/capture clock: AEC3 tolerates drift, Speex does not).
- The ReSpeaker upgrade (spec §4) has hardware AEC: set `voice.aec.engine: none` and
  `voice.barge_in.mode: speech` then.
- The 0.45 s end of speech splits "wait… stop" into two turns; that is the trade-off of
  the fast `voice.vad.trailing_ms`.
