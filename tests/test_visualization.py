import pytest
import numpy as np
import torch
from pathlib import Path
from src.visualization.visualize import DetectionVisualizer

def test_visualizer_initialization(visualizer):
    """Test visualizer initialization and team colors."""
    assert len(visualizer.team_colors) == 10
    assert 'red_bull' in visualizer.team_colors
    assert 'mercedes' in visualizer.team_colors

def test_draw_detection(visualizer, sample_image, sample_detection):
    """Test drawing a single detection."""
    result = visualizer.draw_detection(
        sample_image.copy(),
        sample_detection['box'],
        sample_detection['driver'],
        team=sample_detection['team'],
        confidence=sample_detection['confidence']
    )
    
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape

def test_visualize_detections(visualizer, sample_image):
    """Test visualizing multiple detections."""
    detections = [
        {
            'box': [100, 100, 200, 200],
            'team': 'red_bull',
            'driver': 'verstappen',
            'confidence': 0.95
        },
        {
            'box': [300, 300, 400, 400],
            'team': 'mercedes',
            'driver': 'hamilton',
            'confidence': 0.92
        }
    ]
    
    result = visualizer.visualize_detections(sample_image.copy(), detections)
    assert isinstance(result, np.ndarray)
    assert result.shape == sample_image.shape

def test_plot_training_metrics(visualizer, tmp_path):
    """Test plotting training metrics."""
    metrics = {
        'loss': [0.5, 0.4, 0.3],
        'map50': [0.6, 0.7, 0.8]
    }
    
    save_path = tmp_path / "metrics.png"
    visualizer.plot_training_metrics(metrics, str(save_path))
    assert save_path.exists()

def test_create_comparison_grid(visualizer):
    """Test creating comparison grid."""
    images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(4)]
    titles = ['Image 1', 'Image 2', 'Image 3', 'Image 4']
    
    visualizer.create_comparison_grid(images, titles)
    # No assertion needed as this just creates a matplotlib figure

def test_visualize_attention_maps(visualizer, sample_image, tmp_path):
    """Test attention map visualization."""
    attention_maps = torch.rand((1, sample_image.shape[0], sample_image.shape[1]))
    save_path = tmp_path / "attention.png"
    
    visualizer.visualize_attention_maps(
        sample_image,
        attention_maps,
        str(save_path)
    )
    assert save_path.exists()

def test_team_color_consistency(visualizer, sample_image):
    """Test team color consistency across visualizations."""
    team = 'ferrari'
    detection1 = {'box': [100, 100, 200, 200], 'team': team, 'driver': 'leclerc', 'confidence': 0.9}
    detection2 = {'box': [300, 300, 400, 400], 'team': team, 'driver': 'sainz', 'confidence': 0.85}
    
    img1 = visualizer.draw_detection(
        sample_image.copy(),
        detection1['box'],
        detection1['driver'],
        team=team
    )
    img2 = visualizer.draw_detection(
        sample_image.copy(),
        detection2['box'],
        detection2['driver'],
        team=team
    )
    
    # Both visualizations should use the same team color
    assert np.array_equal(
        img1[detection1['box'][1]:detection1['box'][1]+1, detection1['box'][0]:detection1['box'][0]+1],
        img2[detection2['box'][1]:detection2['box'][1]+1, detection2['box'][0]:detection2['box'][0]+1]
    )