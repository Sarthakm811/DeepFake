from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.image_detector import ImageDetector
from src.text_detector import TextDetector
from src.video_detector import VideoDetector


app = FastAPI(title="Deepfake Detection API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_models: dict[str, Any] = {}


def _get_threshold(env_name: str, default: float) -> float:
    raw = os.getenv(env_name)
    if raw is None:
        return default
    try:
        value = float(raw)
        if 0.0 <= value <= 1.0:
            return value
    except ValueError:
        pass
    return default


def _resolve_label(score: float, threshold: float, margin: float) -> str:
    return "FAKE" if score > (threshold + margin) else "REAL"


def _get_model(key: str):
    if key not in _models:
        if key == "text":
            _models[key] = TextDetector()
        elif key == "image":
            _models[key] = ImageDetector()
        elif key == "video":
            _models[key] = VideoDetector()
    return _models[key]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze")
async def analyze(
    text: str | None = Form(default=None),
    image: UploadFile | None = File(default=None),
    video: UploadFile | None = File(default=None),
) -> dict[str, Any]:
    thresholds = {
        "text": _get_threshold("DEEPFAKE_THRESHOLD_TEXT", 0.954),
        "image": _get_threshold("DEEPFAKE_THRESHOLD_IMAGE", 0.568),
        "video": _get_threshold("DEEPFAKE_THRESHOLD_VIDEO", 0.3055),
    }

    margins = {
        "text": _get_threshold("DEEPFAKE_MARGIN_TEXT", 0.0275),
        "image": _get_threshold("DEEPFAKE_MARGIN_IMAGE", 0.0),
        "video": _get_threshold("DEEPFAKE_MARGIN_VIDEO", 0.2),
    }

    std_limits = {
        "text": _get_threshold("DEEPFAKE_TEXT_STD_LIMIT", 0.2),
        "image": _get_threshold("DEEPFAKE_IMAGE_STD_LIMIT", 0.12),
        "video": _get_threshold("DEEPFAKE_VIDEO_STD_LIMIT", 0.15),
    }

    selected_count = int(bool(text and text.strip())) + int(image is not None) + int(video is not None)
    if selected_count != 1:
        raise HTTPException(status_code=400, detail="Please provide exactly one modality: text OR image OR video.")

    scores: dict[str, float | None] = {"text": None, "image": None, "video": None}
    states: dict[str, str] = {"text": "Not selected", "image": "Not selected", "video": "Not selected"}
    errors: dict[str, str | None] = {"text": None, "image": None, "video": None}
    frames_analyzed: int | None = None
    image_fake_class_index: int | None = None
    text_verification: dict[str, float | list[float]] | None = None
    image_verification: dict[str, float | list[float]] | None = None
    video_verification: dict[str, float | list[float]] | None = None
    selected_modality = "text" if (text and text.strip()) else ("image" if image is not None else "video")

    if text and text.strip():
        try:
            text_model = _get_model("text")
            consistency = text_model.predict_with_consistency(text)
            text_verification = consistency

            raw_score = float(consistency["score"])
            mean_score = float(consistency["mean_score"])
            std_score = float(consistency["std_score"])

            text_threshold = float(thresholds["text"])
            text_std_limit = float(std_limits["text"])
            if raw_score >= text_threshold and (mean_score < text_threshold or std_score > text_std_limit):
                score = mean_score
            else:
                score = raw_score

            scores["text"] = score
            states["text"] = "Processed"
        except Exception as exc:
            errors["text"] = str(exc)
            states["text"] = "Failed"

    if image is not None:
        temp_image_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(image.filename or "image.jpg").suffix or ".jpg") as temp_file:
                temp_image_path = Path(temp_file.name)
                temp_file.write(await image.read())

            image_model = _get_model("image")
            image_fake_class_index = int(getattr(image_model, "fake_class_index", 0))
            consistency = image_model.predict_with_consistency(str(temp_image_path))
            image_verification = consistency

            raw_score = float(consistency["score"])
            mean_score = float(consistency["mean_score"])
            std_score = float(consistency["std_score"])

            image_threshold = float(thresholds["image"])
            image_std_limit = float(std_limits["image"])
            if raw_score >= image_threshold and (mean_score < image_threshold or std_score > image_std_limit):
                score = mean_score
            else:
                score = raw_score

            scores["image"] = score
            states["image"] = "Processed"
        except Exception as exc:
            errors["image"] = str(exc)
            states["image"] = "Failed"
        finally:
            if temp_image_path and temp_image_path.exists():
                temp_image_path.unlink(missing_ok=True)

    if video is not None:
        temp_video_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(video.filename or "video.mp4").suffix or ".mp4") as temp_file:
                temp_video_path = Path(temp_file.name)
                temp_file.write(await video.read())

            video_model = _get_model("video")
            score, frames = video_model.predict(str(temp_video_path))
            video_score = float(score)
            scores["video"] = video_score
            frames_analyzed = int(frames)
            video_verification = {
                "score": video_score,
                "mean_score": video_score,
                "std_score": 0.0,
                "scores": [video_score],
            }
            states["video"] = "Fallback score (model mismatch)" if getattr(video_model, "model_loaded", True) is False else "Processed"
        except Exception as exc:
            errors["video"] = str(exc)
            states["video"] = "Failed"
        finally:
            if temp_video_path and temp_video_path.exists():
                temp_video_path.unlink(missing_ok=True)

    selected_score = scores[selected_modality]
    final_score = float(selected_score if selected_score is not None else 0.5)
    selected_threshold = float(thresholds[selected_modality])
    selected_margin = float(margins[selected_modality])

    return {
        "scores": scores,
        "states": states,
        "errors": errors,
        "frames": frames_analyzed,
        "selected_modality": selected_modality,
        "final_score": final_score,
        "label": _resolve_label(final_score, selected_threshold, selected_margin),
        "threshold": selected_threshold,
        "thresholds": thresholds,
        "margin": selected_margin,
        "margins": margins,
        "std_limits": std_limits,
        "image_fake_class_index": image_fake_class_index,
        "text_verification": text_verification,
        "image_verification": image_verification,
        "video_verification": video_verification,
    }
