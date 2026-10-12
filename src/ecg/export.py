"""ONNX export and INT8 quantisation for deployment.

Requires the ``deploy`` extra (onnx, onnxruntime, tf2onnx). Export goes through
Keras's own ``model.export(format="onnx")``; quantisation through ONNX Runtime.

Two INT8 schemes are provided:

- **static** -- weights *and* activations are INT8. Activation ranges are fixed
  ahead of time from calibration data. In the patient-specific protocol that
  data is the patient's own fine-tuning window, which the deployed system
  already holds, so calibration needs nothing it would not have.
- **dynamic** -- weights are INT8; activation ranges are computed per call. No
  calibration data, but extra work at inference time.

Only ``MatMul`` is quantised. The softmax output stays in float, so the class
probabilities are not rounded to 256 levels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

QUANTISED_OPS = ("MatMul",)


def to_onnx(model, path: str | Path) -> Path:
    """Export a built Keras model to ONNX with a dynamic batch dimension."""
    path = Path(path)
    model.export(str(path), format="onnx", verbose=False)
    return path


def session(path: str | Path, threads: int | None = None):
    """Open an ONNX Runtime CPU session, optionally pinned to ``threads``."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if threads is not None:
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = threads
    return ort.InferenceSession(
        str(path), opts, providers=["CPUExecutionProvider"]
    )


def predict(sess, X: np.ndarray) -> np.ndarray:
    """Class probabilities from an ONNX Runtime session."""
    name = sess.get_inputs()[0].name
    return sess.run(None, {name: np.asarray(X, dtype=np.float32)})[0]


def quantize(
    src: str | Path,
    dst: str | Path,
    scheme: str = "static",
    calibration: np.ndarray | None = None,
) -> Path:
    """Quantise an FP32 ONNX model to INT8.

    ``scheme="static"`` needs ``calibration``: beats in model input layout.
    """
    from onnxruntime.quantization import (
        CalibrationDataReader,
        QuantFormat,
        QuantType,
        quantize_dynamic,
        quantize_static,
    )

    src, dst = Path(src), Path(dst)

    if scheme == "dynamic":
        quantize_dynamic(
            src, dst, weight_type=QuantType.QInt8,
            op_types_to_quantize=list(QUANTISED_OPS),
        )
        return dst

    if scheme != "static":
        raise ValueError(f"unknown scheme {scheme!r}; expected 'static' or 'dynamic'")
    if calibration is None or not len(calibration):
        raise ValueError("static quantisation needs calibration beats")

    input_name = session(src).get_inputs()[0].name
    beats = np.asarray(calibration, dtype=np.float32)

    class _Reader(CalibrationDataReader):
        def __init__(self):
            self._batches = iter(np.array_split(beats, max(1, len(beats) // 64)))

        def get_next(self):
            batch = next(self._batches, None)
            return None if batch is None else {input_name: batch}

    quantize_static(
        src, dst, _Reader(),
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        op_types_to_quantize=list(QUANTISED_OPS),
    )
    return dst
