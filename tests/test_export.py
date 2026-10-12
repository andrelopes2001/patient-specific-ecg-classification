"""ONNX export and quantisation. No data, no training: a seeded, untrained MLP.

Skipped unless the deploy extra is installed.
"""

from __future__ import annotations

from importlib.util import find_spec

import numpy as np
import pytest

# find_spec rather than importorskip: importing tf2onnx loads TensorFlow, and TF
# must not load before the ecg package has imported torch._dynamo (see
# src/ecg/__init__.py), or the interpreter segfaults.
if find_spec("onnxruntime") is None or find_spec("tf2onnx") is None:
    pytest.skip("needs the deploy extra", allow_module_level=True)

from ecg import export  # noqa: E402
from ecg.config import Config, seed_everything  # noqa: E402
from ecg.models import mlp  # noqa: E402


@pytest.fixture(scope="module")
def model_and_beats():
    seed_everything(12)
    cfg = Config()
    model = mlp(cfg=cfg)
    rng = np.random.default_rng(12)
    # Beat-like scale: MLII samples sit roughly within +-2 mV.
    beats = rng.normal(0.0, 0.5, size=(512, cfg.window_length)).astype(np.float32)
    model.predict(beats[:1], verbose=0)
    return model, beats


def _reference_forward(model, beats):
    """The MLP in float64 NumPy. Keras itself is not the reference: on a GPU it
    runs TF32 matmuls, which differ from FP32 by ~1e-4."""
    weights = model.get_weights()
    h = beats.astype(np.float64)
    for i in range(0, len(weights) - 2, 2):
        h = np.maximum(h @ weights[i] + weights[i + 1], 0.0)
    logits = h @ weights[-2] + weights[-1]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def test_onnx_matches_keras_weights(model_and_beats, tmp_path):
    model, beats = model_and_beats
    sess = export.session(export.to_onnx(model, tmp_path / "m.onnx"))
    np.testing.assert_allclose(
        export.predict(sess, beats), _reference_forward(model, beats), atol=1e-5)


def test_onnx_batch_dimension_is_dynamic(model_and_beats, tmp_path):
    model, beats = model_and_beats
    sess = export.session(export.to_onnx(model, tmp_path / "m.onnx"))
    for n in (1, 7, 300):
        assert export.predict(sess, beats[:n]).shape == (n, 4)


@pytest.mark.parametrize("scheme", ["static", "dynamic"])
def test_int8_is_smaller_and_close(model_and_beats, tmp_path, scheme):
    model, beats = model_and_beats
    fp32 = export.to_onnx(model, tmp_path / "fp32.onnx")
    int8 = export.quantize(fp32, tmp_path / f"{scheme}.onnx", scheme,
                           calibration=beats[:256])

    assert int8.stat().st_size < 0.5 * fp32.stat().st_size

    probs = export.predict(export.session(int8), beats[256:])
    ref = export.predict(export.session(fp32), beats[256:])
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.abs(probs - ref).max() < 0.05


def test_static_quantisation_requires_calibration(model_and_beats, tmp_path):
    model, _ = model_and_beats
    fp32 = export.to_onnx(model, tmp_path / "fp32.onnx")
    with pytest.raises(ValueError, match="calibration"):
        export.quantize(fp32, tmp_path / "q.onnx", "static")
    with pytest.raises(ValueError, match="scheme"):
        export.quantize(fp32, tmp_path / "q.onnx", "float16")
