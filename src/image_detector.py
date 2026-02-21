import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision import transforms
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

class ImageDetector:
    def __init__(self, model_path="models/image_efficientnet_b4.pth"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = EfficientNet.from_pretrained('efficientnet-b4')
        self.model._fc = nn.Linear(self.model._fc.in_features, 2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # GradCAM setup
        self.target_layers = [self.model._conv_head]
        self.cam = GradCAM(model=self.model, target_layers=self.target_layers)
    
    def predict(self, image_paths):
        """Predict on image paths"""
        probs = []
        for path in image_paths:
            img = Image.open(path).convert('RGB')
            input_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
            probs.append(prob)
        return np.array(probs)
    
    def explain_gradcam(self, image_path):
        """Generate GradCAM heatmap"""
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        grayscale_cam = self.cam(
            input_tensor=input_tensor, 
            targets=[ClassifierOutputTarget(1)]
        )
        visualization = show_cam_on_image(
            np.float32(img) / 255., 
            grayscale_cam[0], 
            use_rgb=True
        )
        return Image.fromarray(visualization)
