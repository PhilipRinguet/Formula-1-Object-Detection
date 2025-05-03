import cv2
import numpy as np
from typing import Dict, List, Tuple, Union
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import torch

class DetectionVisualizer:
    def __init__(self):
        """Initialize visualizer with team colors and fonts."""
        self.team_colors = {
            'mercedes': (0, 210, 190),
            'red_bull': (30, 91, 198),
            'ferrari': (220, 0, 0),
            'mclaren': (255, 135, 0),
            'alpine': (0, 144, 255),
            'aston_martin': (0, 111, 98),
            'williams': (0, 90, 255),
            'alpha_tauri': (43, 69, 98),
            'alfa_romeo': (144, 0, 0),
            'haas': (255, 255, 255)
        }
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.5
        self.thickness = 2
        
    def draw_detection(self, 
                      image: np.ndarray,
                      box: List[float],
                      label: str,
                      team: str = None,
                      confidence: float = None,
                      color: Tuple[int, int, int] = None) -> np.ndarray:
        """
        Draw a single detection on the image.
        
        Args:
            image: Input image
            box: Bounding box coordinates [x1, y1, x2, y2]
            label: Text label
            team: Team name for color selection
            confidence: Detection confidence
            color: Custom color tuple (B,G,R)
        
        Returns:
            Image with detection visualization
        """
        x1, y1, x2, y2 = map(int, box)
        
        # Use team color or custom color or default white
        if team and team in self.team_colors:
            color = self.team_colors[team]
        elif color is None:
            color = (255, 255, 255)
            
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, self.thickness)
        
        # Prepare label text
        if confidence is not None:
            label = f"{label} {confidence:.2f}"
            
        # Draw label background
        label_size, baseline = cv2.getTextSize(label, self.font, self.font_scale, self.thickness)
        cv2.rectangle(image, (x1, y1 - label_size[1] - baseline),
                     (x1 + label_size[0], y1), color, -1)
        
        # Draw label text
        cv2.putText(image, label, (x1, y1 - baseline), self.font,
                   self.font_scale, (0, 0, 0), self.thickness)
        
        return image
        
    def visualize_detections(self,
                           image: np.ndarray,
                           detections: List[Dict],
                           show_confidence: bool = True) -> np.ndarray:
        """
        Visualize multiple detections on an image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            show_confidence: Whether to show confidence scores
        
        Returns:
            Image with all detections visualized
        """
        img_copy = image.copy()
        
        for det in detections:
            label = det.get('driver', det.get('team', 'car'))
            confidence = det['confidence'] if show_confidence else None
            team = det.get('team')
            
            img_copy = self.draw_detection(
                img_copy,
                det['box'],
                label,
                team=team,
                confidence=confidence
            )
            
        return img_copy
        
    def plot_training_metrics(self,
                            metrics: Dict[str, List[float]],
                            save_path: str = None):
        """
        Plot training metrics.
        
        Args:
            metrics: Dictionary of metric lists
            save_path: Optional path to save the plot
        """
        plt.figure(figsize=(12, 8))
        
        for metric_name, values in metrics.items():
            plt.plot(values, label=metric_name)
            
        plt.title('Training Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path)
        plt.close()
        
    def create_comparison_grid(self,
                             images: List[np.ndarray],
                             titles: List[str],
                             cols: int = 2,
                             figsize: Tuple[int, int] = (15, 10)) -> None:
        """
        Create a grid of images for comparison.
        
        Args:
            images: List of images to display
            titles: List of titles for each image
            cols: Number of columns in the grid
            figsize: Figure size
        """
        rows = (len(images) + cols - 1) // cols
        plt.figure(figsize=figsize)
        
        for idx, (img, title) in enumerate(zip(images, titles)):
            plt.subplot(rows, cols, idx + 1)
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            plt.title(title)
            plt.axis('off')
            
        plt.tight_layout()
        
    def visualize_attention_maps(self,
                               image: np.ndarray,
                               attention_maps: torch.Tensor,
                               save_path: str = None):
        """
        Visualize attention maps from the model.
        
        Args:
            image: Original image
            attention_maps: Attention maps tensor
            save_path: Optional path to save visualization
        """
        # Convert attention maps to numpy and normalize
        att_maps = attention_maps.cpu().numpy()
        att_maps = (att_maps - att_maps.min()) / (att_maps.max() - att_maps.min())
        
        plt.figure(figsize=(15, 5))
        
        # Original image
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')
        
        # Attention heatmap
        plt.subplot(1, 3, 2)
        plt.imshow(att_maps[0], cmap='jet')
        plt.title('Attention Heatmap')
        plt.colorbar()
        plt.axis('off')
        
        # Overlay
        plt.subplot(1, 3, 3)
        img_float = image.astype(float) / 255
        att_map_resized = cv2.resize(att_maps[0], (image.shape[1], image.shape[0]))
        overlay = (img_float * 0.7 + np.stack([att_map_resized]*3, axis=-1) * 0.3)
        overlay = np.clip(overlay * 255, 0, 255).astype(np.uint8)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title('Attention Overlay')
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path)
        plt.close()