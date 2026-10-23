import base64
import binascii
import os
import time

from flask import Flask, jsonify, render_template, request
from inference_sdk import InferenceHTTPClient

WORKSPACE = "diva-s-workspace"
WORKFLOW_ID = "happy-or-sad-face-preview-1792607881848"
OUTPUT_IMAGE_KEY = "output_image"
PREDICTIONS_KEY = "predictions"
MAX_UPLOAD_BYTES = 2 * 1024 * 1024  # a 400x300 JPEG is far smaller than this
MAX_ATTEMPTS = 3

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES * 2

API_KEY = os.environ.get("ROBOFLOW_API_KEY")
if not API_KEY:
    raise RuntimeError("ROBOFLOW_API_KEY is not set. Export it before starting the server.")

CLIENT = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=API_KEY)


class WorkflowError(Exception):
    """Raised when the Roboflow workflow call fails or returns an unexpected shape."""


def strip_data_url(data_url):
    """Return the raw base64 payload from a 'data:image/...;base64,' string."""
    if not isinstance(data_url, str) or not data_url:
        raise ValueError("Missing image data.")
    payload = data_url.partition(",")[2] if data_url.startswith("data:") else data_url
    try:
        base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("Image is not valid base64.") from exc
    return payload


def run_workflow_on_frame(b64_payload):
    """Call the workflow with retries and backoff. Returns the first result entry."""
    last_error = None
    for attempt in range(MAX_ATTEMPTS):
        try:
            results = CLIENT.run_workflow(
                workspace_name=WORKSPACE,
                workflow_id=WORKFLOW_ID,
                images={"image": b64_payload},
            )
            if not isinstance(results, list) or not results or not isinstance(results[0], dict):
                raise WorkflowError(f"Unexpected response shape: {type(results).__name__}")
            return results[0]
        except WorkflowError:
            raise
        except Exception as exc:  # network or HTTP error from the SDK
            last_error = exc
            time.sleep(0.5 * (2 ** attempt))
    raise WorkflowError(f"Workflow call failed after {MAX_ATTEMPTS} attempts: {last_error}")


def extract_output_image(entry):
    """Return the annotated image as a data URL. Handles a raw base64 string or {type, value}."""
    raw = entry.get(OUTPUT_IMAGE_KEY)
    if isinstance(raw, dict):
        raw = raw.get("value")
    if not isinstance(raw, str) or not raw:
        return None
    if raw.startswith("data:"):
        return raw
    return f"data:image/jpeg;base64,{raw}"


def extract_predictions(entry):
    """Keep only the fields the page uses. Drops polygon/point data."""
    preds = entry.get(PREDICTIONS_KEY)
    if not isinstance(preds, dict):
        return {"top": None, "predictions": []}
    items = preds.get("predictions") or []
    slim = [
        {"class": p.get("class"), "confidence": p.get("confidence")}
        for p in items
        if isinstance(p, dict)
    ]
    return {"top": preds.get("top"), "predictions": slim}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True) or {}
    try:
        payload = strip_data_url(body.get("image"))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        entry = run_workflow_on_frame(payload)
    except WorkflowError as exc:
        return jsonify({"error": str(exc)}), 502

    return jsonify({
        "image": extract_output_image(entry),
        "predictions": extract_predictions(entry),
    })


if __name__ == "__main__":
    # Debug reloader is off by default; set FLASK_DEBUG=1 locally to enable it.
    app.run(debug=os.environ.get("FLASK_DEBUG") == "1", port=5000)
