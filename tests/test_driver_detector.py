import pytest
import torch
import numpy as np
from pathlib import Path
from src.models.driver_detector import DriverDetector

def test_driver_detector_initialization(driver_detector):
    """Test driver detector initialization."""
    # Test all drivers are included
    assert len(driver_detector.class_names) == 20  # 10 teams × 2 drivers
    
    # Test team-driver mapping
    assert len(driver_detector.team_drivers) == 10
    assert all(len(drivers) == 2 for drivers in driver_detector.team_drivers.values())
    
    # Test specific driver-team relationships
    assert 'verstappen' in driver_detector.team_drivers['red_bull']
    assert 'hamilton' in driver_detector.team_drivers['mercedes']

def test_driver_detector_predict_hierarchy(driver_detector, sample_image):
    """Test hierarchical prediction functionality."""
    with pytest.raises(ValueError):  # Should fail without model
        driver_detector.predict_with_hierarchy(sample_image)

def test_driver_detector_tracking(driver_detector, tmp_path):
    """Test video tracking functionality."""
    video_path = str(tmp_path / "test.mp4")
    output_path = str(tmp_path / "output")
    
    with pytest.raises(ValueError):  # Should fail without model
        driver_detector.process_video_with_tracking(video_path, output_path)

def test_driver_detector_team_consistency(driver_detector):
    """Test team-driver consistency in predictions."""
    assert all(
        driver in [d for drivers in driver_detector.team_drivers.values() for d in drivers]
        for driver in driver_detector.class_names
    )

def test_driver_detector_mlflow_metrics(driver_detector, tmp_path):
    """Test MLflow metrics logging."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("""
        train: path/to/train
        val: path/to/val
        test: path/to/test
        nc: 20
        names: ['hamilton', 'russell', 'verstappen', 'perez', 'leclerc', 
                'sainz', 'norris', 'piastri', 'gasly', 'ocon', 'alonso',
                'stroll', 'albon', 'sargeant', 'tsunoda', 'ricciardo',
                'bottas', 'zhou', 'magnussen', 'hulkenberg']
    """)
    
    with pytest.raises(ValueError):  # Should fail without model
        driver_detector.train_with_mlflow(
            str(data_yaml),
            epochs=1,
            experiment_name="test_driver_detection"
        )

def test_driver_detector_confidence_thresholds(driver_detector, sample_image):
    """Test different confidence threshold levels."""
    with pytest.raises(ValueError):  # Should fail without model
        driver_detector.predict_with_hierarchy(
            sample_image,
            team_conf=0.5,
            driver_conf=0.3
        )