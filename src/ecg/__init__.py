"""Patient-specific ECG beat classification on the MIT-BIH Arrhythmia Database."""

# PyTorch's dynamo module must be initialised before TensorFlow is loaded.
# Importing torch._dynamo *after* TF is in the process segfaults the
# interpreter (exit 139, no traceback) -- which is reachable here because the
# cGAN augmentation is PyTorch while the classifier is Keras. Importing it
# eagerly fixes the order regardless of which submodule a caller touches first.
# A plain ``import torch`` is not sufficient; the dynamo import is the trigger.
try:  # pragma: no cover - depends on the optional torch extra
    import torch as _torch
    import torch._dynamo as _torch_dynamo  # noqa: F401
except Exception:  # torch is optional; the Keras path does not need it
    _torch = None

from .config import CLASSES, DEFAULT, DS1, DS2, Config, seed_everything

__all__ = ["CLASSES", "DEFAULT", "DS1", "DS2", "Config", "seed_everything"]
