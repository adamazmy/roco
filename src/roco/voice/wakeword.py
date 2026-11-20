"""openWakeWord detector (ONNX runtime: tflite is not available on Windows)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class OpenWakeWordDetector:
    def __init__(self, model_path: Path, feature_dir: Path) -> None:
        from openwakeword.model import Model

        missing = [
            p for p in (model_path, feature_dir / "melspectrogram.onnx",
                        feature_dir / "embedding_model.onnx")
            if not Path(p).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                "Wake word model files missing: "
                + ", ".join(str(p) for p in missing) + ". Run: roco download-models"
            )
        self.model = Model(
            wakeword_models=[str(model_path)],
            inference_framework="onnx",
            melspec_model_path=str(feature_dir / "melspectrogram.onnx"),
            embedding_model_path=str(feature_dir / "embedding_model.onnx"),
        )
        self.name = Path(model_path).stem
        log.info("Wake word model loaded: %s", self.name)

    def score(self, frame: Any) -> float:
        scores = self.model.predict(frame)
        return float(max(scores.values())) if scores else 0.0

    def reset(self) -> None:
        self.model.reset()
