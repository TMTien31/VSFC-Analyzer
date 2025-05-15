from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

class BasePredictor:
    def __init__(self, model_repo, device=None):
        """
        model_repo: str, tên repo model trên Hugging Face Hub, ví dụ "username/model-name"
        device: "cuda" hoặc "cpu" hoặc None (auto chọn)
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load tokenizer & model trực tiếp từ Hugging Face Hub
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_repo)
        
        self.model.to(self.device)
        self.model.eval()

    def predict(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)
        
        return preds.cpu().numpy().tolist()

class SentimentPredictor(BasePredictor):
    label_map = ['Negative', 'Neutral', 'Positive']

    def predict(self, sentences):
        preds = super().predict(sentences)
        return [self.label_map[p] for p in preds]

class TopicPredictor(BasePredictor):
    label_map = ['Lecturer', 'Training program', 'Facility', 'Others']

    def predict(self, sentences):
        preds = super().predict(sentences)
        return [self.label_map[p] for p in preds]
