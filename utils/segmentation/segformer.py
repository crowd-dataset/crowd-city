"""SegFormer Cityscapes inference wrapper used to locate road and footpath.

The model is loaded lazily and only once per process. Frames arrive already
scaled to the network input size by ffmpeg, so no resizing happens here and the
returned label map keeps a fixed, known geometry: normalised bounding-box
coordinates map linearly onto it.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np

from custom_logger import CustomLogger


logger = CustomLogger(__name__)

# Any nvidia/segformer-b*-finetuned-cityscapes checkpoint is a drop-in
# replacement; b4 and b5 are more accurate and correspondingly slower. The
# smaller b0 variant exists as an escape hatch when segmentation throughput,
# rather than accuracy, is the binding constraint.
DEFAULT_MODEL_NAME = "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
DEFAULT_INPUT_WIDTH = 1024
DEFAULT_INPUT_HEIGHT = 512
DEFAULT_BATCH_SIZE = 8


def resolve_device(requested: Optional[str] = None) -> str:
    """Return the torch device to run segmentation on."""
    import torch

    choice = str(requested or "auto").strip().lower()
    if choice not in {"", "auto"}:
        return choice
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class SurfaceSegmenter:
    """Batch SegFormer inference returning Cityscapes trainId label maps."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        input_width: int = DEFAULT_INPUT_WIDTH,
        input_height: int = DEFAULT_INPUT_HEIGHT,
    ) -> None:
        self.model_name = str(model_name)
        self.batch_size = max(1, int(batch_size))
        self.input_width = int(input_width)
        self.input_height = int(input_height)
        self.device = resolve_device(device)
        self._model = None
        self._processor = None

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        logger.info(
            f"Loading segmentation model {self.model_name} on device {self.device}."
        )
        self._processor = SegformerImageProcessor.from_pretrained(self.model_name)
        model = SegformerForSemanticSegmentation.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)
        self._model = model

        # A warning here is far cheaper than discovering a silent class-order
        # mismatch after a full run.
        labels = getattr(model.config, "id2label", {}) or {}
        if len(labels) != 19:
            logger.warning(
                f"Segmentation model {self.model_name} reports {len(labels)} classes; "
                "CROWD surface mapping assumes the 19 Cityscapes evaluation classes."
            )
        del torch

    @property
    def model_identifier(self) -> str:
        """Return the identifier recorded in the segmentation manifest."""
        return f"{self.model_name}@{self.input_width}x{self.input_height}"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def segment(self, frames: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return the trainId map and per-pixel confidence for RGB ``frames``.

        ``frames`` is an ``(n, height, width, 3)`` uint8 array already scaled to
        the network input size. The returned maps are the raw SegFormer output
        grid, which is a quarter of the input resolution in each axis; callers
        address it with normalised coordinates so the exact size never matters.
        """
        import torch

        if frames.size == 0:
            return (
                np.zeros((0, 0, 0), dtype=np.int16),
                np.zeros((0, 0, 0), dtype=np.float32),
            )

        self._ensure_loaded()
        assert self._model is not None and self._processor is not None

        label_batches = []
        confidence_batches = []

        with torch.inference_mode():
            for start in range(0, len(frames), self.batch_size):
                chunk = frames[start:start + self.batch_size]
                inputs = self._processor(
                    images=list(chunk),
                    do_resize=False,
                    return_tensors="pt",
                )
                pixel_values = inputs["pixel_values"].to(self.device)
                logits = self._model(pixel_values=pixel_values).logits
                probabilities = torch.softmax(logits.float(), dim=1)
                confidence, labels = probabilities.max(dim=1)
                label_batches.append(labels.to("cpu").numpy().astype(np.int16))
                confidence_batches.append(
                    confidence.to("cpu").numpy().astype(np.float32)
                )

        return (
            np.concatenate(label_batches, axis=0),
            np.concatenate(confidence_batches, axis=0),
        )


def segmentation_is_available() -> bool:
    """Return whether the optional segmentation dependencies are importable."""
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as error:
        logger.warning(f"Segmentation dependencies are unavailable: {error}")
        return False
    return True


def offline_mode_enabled() -> bool:
    """Return whether Hugging Face downloads are disabled for this process."""
    for name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        value = str(os.environ.get(name, "")).strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
    return False
