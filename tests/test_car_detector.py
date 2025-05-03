import pytest
import torch
import numpy as np
from pathlib import Path
from src.models.car_detector import CarDetector

def test_car_detector_initialization(car_detector):
    """Test car detector initialization."""
    assert car_detector.class_names == ['f1_car']
    assert car_detector.model is None

def test_car_detector_predict_without_model(car_detector, sample_image):
    """Test prediction without model loaded raises error."""
    with pytest.raises(ValueError):
        car_detector.predict(sample_image)

def test_car_detector_hyperparameter_optimization(car_detector, tmp_path):
    """Test hyperparameter optimization functionality."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("""
        train: path/to/train
        val: path/to/val
        test: path/to/test
        nc: 1
        names: ['f1_car']
    """)
    
    with pytest.raises(Exception):  # Should fail without a model
        car_detector.optimize_hyperparameters(str(data_yaml), n_trials=1)

def test_car_detector_mlflow_training(car_detector, tmp_path):
    """Test MLflow training integration."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("""
        train: path/to/train
        val: path/to/val
        test: path/to/test
        nc: 1
        names: ['f1_car']
    """)
    
    with pytest.raises(ValueError):  # Should fail without a model
        car_detector.train_with_mlflow(str(data_yaml), epochs=1)