"""Smoke test: runs the live workflow on one sample frame.

Skipped unless ROBOFLOW_API_KEY is set and SAMPLE_IMAGE points to a JPEG/PNG file.
"""
import base64
import os

import pytest

pytestmark = pytest.mark.skipif(
    not (os.environ.get("ROBOFLOW_API_KEY") and os.environ.get("SAMPLE_IMAGE")),
    reason="set ROBOFLOW_API_KEY and SAMPLE_IMAGE to run the live smoke test",
)


def test_workflow_returns_expected_keys():
    import server

    with open(os.environ["SAMPLE_IMAGE"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    entry = server.run_workflow_on_frame(b64)
    assert server.OUTPUT_IMAGE_KEY in entry, f"missing {server.OUTPUT_IMAGE_KEY}: {list(entry)}"
    assert server.PREDICTIONS_KEY in entry, f"missing {server.PREDICTIONS_KEY}: {list(entry)}"

    image = server.extract_output_image(entry)
    assert image and image.startswith("data:image/")
    preds = server.extract_predictions(entry)
    assert "predictions" in preds and "top" in preds
