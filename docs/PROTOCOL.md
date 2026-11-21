# Link protocol v1 (brain server ↔ body client)

The body (Pi, or the PC during development) and the brain (PC) talk over one WebSocket.
Code: `roco.protocol` (`messages`, `audio`, `transport`). Tests: `tests/test_protocol.py`.

- **Text frames** are JSON control messages: `{"v": 1, "type": "...", ...}`.
- **Binary frames** are audio: an 18-byte header (`RA`, version, kind, flags, sentence,
  turn, seq, rate) then PCM16 little-endian mono samples.
- `roco brain` + `roco body` use `WebSocketConnection`; `roco run` connects the same two
  halves with `memory_pipe()`, which still encodes every frame, so both paths share
  one codec.

## Handshake

```
body  → hello{protocol:1, body_id, token, mic_rate:16000, caps}
brain → welcome{protocol:1, brain:"ollama:qwen2.5:7b", tts_rate, session, settings}
        or error{code:"version"|"auth"|"busy", fatal:true} and close
```

`token` is the shared secret in `~/.roco/link_token` (`load_or_create_token`). The brain
creates it on first start. Copy the same file to the Pi; it is never committed.
One body at a time: a second one gets `error{code:"busy"}`.

## A normal turn

```
body  → turn_start{turn:1, trigger:"wake"}
body  → [audio MIC turn 1 seq 0..n]        streamed while Adam talks (80 ms frames)
body  → turn_end{turn:1, reason:"silence", trailing_ms:450}
brain → state{thinking} · transcript{turn:1, text}
brain → emotion{happy}
brain → sentence{turn:1, index:0, text}    text always before its audio
brain → [audio TTS turn 1 sentence 0 ... END]
brain → sentence{turn:1, index:1, ...} ...
brain → turn_done{turn:1, text, brain, stages:{stt_done, first_token, first_sentence, first_audio_sent}}
body  → playback_done{turn:1, interrupted:false, heard}
body  → turn_metrics{turn:1, stages:{vad_trailing, first_audio_played}}
```

Slow tools: the brain may send `sentence{filler:true}` (with audio) straight away, e.g.
"Let me check.", then the real answer as later sentences of the same turn.

## Barge-in

```
(ROCO is playing turn 1; AEC-cleaned mic shows Adam speaking)
body  : cuts playback locally (< 200 ms, no round trip needed)
body  → interrupt{turn:1, reason:"barge_in"}
body  → playback_done{turn:1, interrupted:true, heard:"Tomorrow you have the dentist,"}
body  → turn_start{turn:2, trigger:"barge_in"} → audio → turn_end
brain : cancels turn 1 generation/TTS, stores only what was heard in history
```

Anything that later arrives for a cancelled turn is dropped by the receiver.

## Announcements

The brain speaks on its own with `announce{turn: brain_turn_id(n), text}`, followed by
`sentence` and TTS audio for that turn, then `turn_done`. Brain turn ids have the top bit
set, so they never collide with the body's turn numbers.

## Other messages

| Type | Direction | Purpose |
|---|---|---|
| `ping` / `pong` | both | Liveness and round-trip time (`ts` is the sender's clock, echoed) |
| `settings` | both | Partial settings change; the brain applies it and broadcasts the result |
| `text_input` | body → brain | A typed turn instead of audio |
| `error` | both | `version`, `auth`, `busy`, `bad_message`, `internal`; `fatal` closes |

## Timing and latency

Clocks are not synchronised. Each side reports durations in ms on its own monotonic clock,
**measured from end of speech**: the brain from the moment `turn_end` arrives minus
`trailing_ms`; the body from the last speech frame. The brain logs one line per turn that
merges its `stages` with the body's `turn_metrics`.

## Compatibility rules

- New optional fields and new message types are allowed within v1. Receivers ignore unknown
  fields and log/skip `Unknown` messages.
- Removing or renaming a field, or changing the audio header, needs `PROTOCOL_VERSION = 2`.
- A compressed audio codec is a new `AudioKind`, not a format change.
