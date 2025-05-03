"""Shared fixtures for F1 object detection tests."""
import pytest
import numpy as np
import torch
from pathlib import Path
from src.models.car_detector import CarDetector
from src.models.team_detector import TeamDetector
from src.models.driver_detector import DriverDetector
from src.visualization.visualize import DetectionVisualizer

@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

@pytest.fixture
def mock_config_path(tmp_path):
    """Create a mock config file."""
    config = {
        'model_type': 'yolov11',
        'backbone': {'type': 'darknet'},
        'training': {'imgsz': 640}
    }
    config_path = tmp_path / "mock_config.yaml"
    config_path.write_text("""
model_type: yolov11
backbone:
    type: darknet
training:
    imgsz: 640
    """)
    return str(config_path)

@pytest.fixture
def car_detector():
    """Create a car detector instance for testing."""
    return CarDetector()

@pytest.fixture
def team_detector():
    """Create a team detector instance for testing."""
    return TeamDetector()

@pytest.fixture
def driver_detector():
    """Create a driver detector instance for testing."""
    return DriverDetector()

@pytest.fixture
def visualizer():
    """Create a visualizer instance for testing."""
    return DetectionVisualizer()

@pytest.fixture
def sample_detection():
    """Create a sample detection result."""
    return {
        'box': [100, 100, 200, 200],
        'team': 'red_bull',
        'driver': 'verstappen',
        'confidence': 0.95
    }