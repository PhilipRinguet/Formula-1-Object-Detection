from typing import Dict, List, Optional, Tuple
import torch
import yaml
from ultralytics import YOLO

class BaseDetector:
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize the base detector.
        
        Args:
            model_path (str, optional): Path to pretrained weights
            config_path (str, optional): Path to model configuration
        """
        self.model = None
        self.config = None
        
        if model_path:
            self.load_model(model_path)
        if config_path:
            self.load_config(config_path)
            
    def load_model(self, model_path: str) -> None:
        """Load a pretrained YOLO model."""
        self.model = YOLO(model_path)
        
    def load_config(self, config_path: str) -> None:
        """Load model configuration from YAML."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
    def predict(self, image, **kwargs):
        """Run inference on an image."""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.predict(image, **kwargs)
    
    def train(self, **kwargs):
        """Train the model with given parameters."""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.train(**kwargs)
    
    def validate(self, **kwargs):
        """Validate the model."""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.val(**kwargs)
    
    def export(self, format: str = "onnx", **kwargs):
        """Export the model to different formats."""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.export(format=format, **kwargs)
    
    def process_video(self, video_path: str, output_path: str, **kwargs):
        """Process a video file."""
        if self.model is None:
            raise ValueError("No model loaded")
        return self.model.predict(source=video_path, save=True, project=output_path, **kwargs)