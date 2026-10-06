"""How far the background slides sideways while a pedestrian is tracked.

When the car turns, the whole picture sweeps sideways, and a pedestrian
standing on the pavement is carried across the image as if crossing. The
box-only camera-motion check (independent_motion_rejection) catches this
only when a static object is tracked in the same frames; this measures the
sweep from the picture itself.

Consecutive frames, CAMERA_SHIFT_HZ apart and scaled to FRAME_WIDTH, are
compared by phase correlation over their upper half (buildings and the far
scene rather than the road and nearby traffic), and the sideways shifts are
summed into the background's displacement over the window, in image widths
(positive to the right).
"""

from __future__ import annotations

import cv2
import numpy as np

CAMERA_SHIFT_HZ = 5.0
FRAME_WIDTH = 320
FRAME_HEIGHT = 180


def background_shift(frames: np.ndarray) -> float:
    """Return the background's sideways displacement over ``frames``, in image widths.

    ``frames`` is ``(n, height, width, 3)`` RGB, in time order.
    """
    previous = None
    window = None
    total = 0.0
    for frame in frames:
        grey = cv2.cvtColor(np.ascontiguousarray(frame), cv2.COLOR_RGB2GRAY).astype(np.float32)
        grey = grey[: grey.shape[0] // 2]
        if window is None:
            window = cv2.createHanningWindow((grey.shape[1], grey.shape[0]), cv2.CV_32F)
        grey = grey * window
        if previous is not None:
            (dx, _), _ = cv2.phaseCorrelate(previous, grey)
            total += float(dx) / float(grey.shape[1])
        previous = grey
    return total
