from pathlib import Path
import os

import numpy as np
import timm
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image
from PIL import ImageOps
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torchvision import transforms


def _resolve_image_model_path(model_path: str | None = None) -> Path:
    if model_path:
        candidate = Path(model_path)
        if candidate.exists():
            return candidate

    project_root = Path(__file__).resolve().parents[1]
    candidates = [
        project_root / "models" / "image_efficientnet_b4.pth",
        project_root / "notebooks" / "models" / "best_model.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Image model not found. Expected one of: "
        f"{', '.join(str(path) for path in candidates)}"
    )

class ImageDetector:
    def __init__(self, model_path: str | None = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(42)
        resolved_path = _resolve_image_model_path(model_path)
        checkpoint = torch.load(resolved_path, map_location=self.device)
        self.input_size = 380

        self.fake_class_index = self._resolve_fake_class_index(checkpoint)

        self.model = self._build_compatible_model(checkpoint)
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((self.input_size, self.input_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # GradCAM setup
        target_layer = getattr(self.model, "_conv_head", None)
        if target_layer is None:
            target_layer = getattr(self.model, "conv_head", None)
        backbone = getattr(self.model, "backbone", None)
        if target_layer is None and backbone is not None:
            target_layer = getattr(backbone, "conv_head", None)
        if target_layer is None:
            raise RuntimeError("Could not determine GradCAM target layer for image model")

        self.target_layers = [target_layer]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)

    def _build_compatible_model(self, checkpoint):
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        if isinstance(state_dict, dict) and any(str(key).startswith("backbone.") for key in state_dict.keys()):
            model = _SimpleDeepfakeModel()
            model.load_state_dict(state_dict, strict=True)
            self.input_size = 224
            return model

        try:
            model = EfficientNet.from_pretrained("efficientnet-b4")
            model._fc = nn.Linear(model._fc.in_features, 2)
            model.load_state_dict(state_dict)
            return model
        except Exception:
            if any(str(key).startswith("backbone.") for key in state_dict.keys()):
                state_dict = {
                    str(key).replace("backbone.", "", 1): value
                    for key, value in state_dict.items()
                    if str(key).startswith("backbone.")
                }

            model_name = self._select_timm_efficientnet_variant(state_dict)
            model = timm.create_model(model_name, pretrained=True, num_classes=2)

            model_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and model_state[key].shape == value.shape
            }
            model.load_state_dict(compatible_state, strict=False)
            return model

    def _resolve_fake_class_index(self, checkpoint) -> int:
        env_value = os.getenv("DEEPFAKE_IMAGE_FAKE_INDEX")
        if env_value in {"0", "1"}:
            return int(env_value)

        class_to_idx = None
        if isinstance(checkpoint, dict):
            maybe_mapping = checkpoint.get("class_to_idx")
            if isinstance(maybe_mapping, dict):
                class_to_idx = maybe_mapping

        if class_to_idx:
            normalized = {str(k).strip().lower(): int(v) for k, v in class_to_idx.items()}
            if "fake" in normalized:
                return normalized["fake"]
            if "real" in normalized and normalized["real"] in {0, 1}:
                return 1 - normalized["real"]

        return 1

    def _select_timm_efficientnet_variant(self, state_dict):
        stem = state_dict.get("conv_stem.weight")
        if stem is None:
            return "efficientnet_b0"

        target_shape = stem.shape
        candidates = [
            "efficientnet_b0",
            "efficientnet_b1",
            "efficientnet_b2",
            "efficientnet_b3",
            "efficientnet_b4",
            "efficientnet_b5",
            "efficientnet_b6",
            "efficientnet_b7",
        ]
        for model_name in candidates:
            try:
                probe = timm.create_model(model_name, pretrained=False, num_classes=2)
                probe_stem = probe.state_dict().get("conv_stem.weight")
                if probe_stem is not None and probe_stem.shape == target_shape:
                    return model_name
            except Exception:
                continue

        return "efficientnet_b0"

    def predict(self, image_paths):
        """Predict on image paths"""
        if image_paths is None:
            return np.array([])

        probs = []
        for path in image_paths:
            img = Image.open(Path(path)).convert("RGB")
            probs.append(self._predict_single_image(img))
        return np.array(probs)

    def _predict_single_image(self, img: Image.Image) -> float:
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
            prob = torch.softmax(output, dim=1)[0, self.fake_class_index].item()
        return float(prob)

    def predict_with_consistency(self, image_path: str) -> dict:
        img = Image.open(Path(image_path)).convert("RGB")
        variants = [img, ImageOps.mirror(img)]
        scores = [self._predict_single_image(variant) for variant in variants]
        return {
            "score": float(scores[0]),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "scores": [float(score) for score in scores],
        }

    def explain_gradcam(self, image_path):
        """Generate GradCAM heatmap"""
        img = Image.open(Path(image_path)).convert("RGB")
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)

        grayscale_cam = self.cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(self.fake_class_index)]
        )
        visualization = show_cam_on_image(
            np.float32(img) / 255.,
            grayscale_cam[0],
            use_rgb=True
        )
        return Image.fromarray(visualization)


class _SimpleDeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=True)
        in_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 2),
        )

    def forward(self, x):
        return self.backbone(x)
