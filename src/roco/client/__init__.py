"""The body client: mic, wake word, VAD, echo cancellation, playback, barge-in, face.

``roco body`` connects it to a brain server over a WebSocket; ``roco run`` joins it to
an in-process brain with :func:`roco.protocol.memory_pipe`. See docs/PROTOCOL.md.
Nothing here imports numpy/sounddevice/onnxruntime at module level.
"""
