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

## Open points

- Measure AEC3 CPU and ERLE on the Pi 5 with the USB speaker + USB mic (different
  echo path, and a separate playback/capture clock: AEC3 tolerates drift, Speex does not).
- The ReSpeaker upgrade (spec §4) has hardware AEC: set `voice.aec.engine: none` and
  `voice.barge_in.mode: speech` then.
- The 0.45 s end of speech splits "wait… stop" into two turns; that is the trade-off of
  the fast `voice.vad.trailing_ms`.
