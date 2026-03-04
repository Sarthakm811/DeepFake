from pathlib import Path
import re

import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


def _resolve_text_model_path(model_path: str | None = None) -> Path:
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "models" / "text_distilbert",
        project_root / "notebooks" / "models" / "text_distilbert",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Text model not found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )

class TextDetector:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        resolved_path = _resolve_text_model_path(model_path)
        self.tokenizer = DistilBertTokenizer.from_pretrained(resolved_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(resolved_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts):
        """Predict fake probability for texts"""
        if texts is None:
            return np.array([])
        if isinstance(texts, str):
            texts = [texts]
        if len(texts) == 0:
            return np.array([])

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        return probs

    def _text_variants(self, text: str) -> list[str]:
        collapsed = re.sub(r"\s+", " ", text).strip()
        no_emoji = re.sub(r"[^\w\s.,!?;:'\"()\-]", " ", collapsed)
        no_emoji = re.sub(r"\s+", " ", no_emoji).strip()
        return [
            text,
            collapsed,
            collapsed.lower(),
            no_emoji,
        ]

    def predict_with_consistency(self, text: str) -> dict:
        variants = self._text_variants(text)
        scores = self.predict(variants)
        return {
            "score": float(scores[0]),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": [float(score) for score in scores],
        }
    
    def explain_shap(self, background_texts, test_texts, n_samples=100):
        """SHAP explanations"""
        try:
            import shap
        except ImportError as error:
            raise ImportError(
                "SHAP is not installed. Install with: pip install shap"
            ) from error

        def predict_fn(texts):
            return self.predict(texts)

        explainer = shap.KernelExplainer(predict_fn, background_texts[:50])
        shap_values = explainer.shap_values(test_texts, nsamples=n_samples)
        return shap_values
