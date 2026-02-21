import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import shap
import numpy as np

class TextDetector:
    def __init__(self, model_path="models/text_distilbert"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, texts):
        """Predict fake probability for texts"""
        inputs = self.tokenizer(texts, return_tensors='pt', 
                               padding=True, truncation=True, max_length=128)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
        return probs
    
    def explain_shap(self, background_texts, test_texts, n_samples=100):
        """SHAP explanations"""
        def predict_fn(texts):
            return self.predict(texts)
        
        explainer = shap.KernelExplainer(predict_fn, background_texts[:50])
        shap_values = explainer.shap_values(test_texts, nsamples=n_samples)
        return shap_values
