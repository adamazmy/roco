# Roadmap

Rough order, not a promise.

## Barge-in
Let Adam interrupt ROCO mid-sentence. Needs acoustic echo cancellation (or a mic array
with built-in AEC such as the ReSpeaker) so the wake-word detector can keep running while
ROCO speaks. `Speaker.stop()` already exists for this.

## Custom "hey roco" wake word
Train an openWakeWord model with its synthetic-data notebook (Piper generates thousands of
"hey roco" clips, mixed with background noise), export to ONNX, drop it in
`~/.roco/models/` and set `voice.wake_word.model` to its path. Collect a few minutes of real
false-trigger audio from the desk to tune `threshold`.

## Camera and face tracking
Pi Camera Module 3 + a small face detector (MediaPipe / YuNet) driving the pan/tilt servos
so ROCO looks at whoever is talking. Publish a `look_at` bus event the Body consumes.

## Smart-home control
Tools for lights and plugs, starting with the IKEA KAJPLATS bulb project (Matter over
Thread via a DIRIGERA hub or Home Assistant). Every state-changing action keeps the
"confirm first" rule for anything non-trivial.

## Speaker identification
Voice embeddings (e.g. resemblyzer / ECAPA) so ROCO knows Adam from guests and only reads
email to Adam.

## Long-term memory
A small local store of facts Adam tells ROCO ("my sister's birthday is in May"), retrieved
by embedding search and injected into the system prompt, with "forget that" support.
