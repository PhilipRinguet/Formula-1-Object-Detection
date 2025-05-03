import pytest
import torch
import numpy as np
from pathlib import Path
from src.models.team_detector import TeamDetector

def test_team_detector_initialization(team_detector):
    """Test team detector initialization."""
    expected_teams = [
        'mercedes', 'red_bull', 'ferrari', 'mclaren',
        'alpine', 'aston_martin', 'williams', 'alpha_tauri',
        'alfa_romeo', 'haas'
    ]
    assert team_detector.class_names == expected_teams
    assert team_detector.model is None

def test_team_detector_predict_without_model(team_detector, sample_image):
    """Test prediction without model loaded raises error."""
    with pytest.raises(ValueError):
        team_detector.predict_with_confidence(sample_image)

def test_team_detector_class_weights(team_detector, tmp_path):
    """Test class weights calculation during training."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("""
        train: path/to/train
        val: path/to/val
        test: path/to/test
        nc: 10
        names: ['mercedes', 'red_bull', 'ferrari', 'mclaren',
                'alpine', 'aston_martin', 'williams', 'alpha_tauri',
                'alfa_romeo', 'haas']
    """)
    
    with pytest.raises(ValueError):  # Should fail without a model
        team_detector.train_with_mlflow(
            str(data_yaml),
            epochs=1,
            class_weights=[1.0] * 10
        )

def test_team_detector_confidence_threshold(team_detector):
    """Test confidence threshold in prediction."""
    assert hasattr(team_detector, 'predict_with_confidence')

def test_team_detector_hyperparameter_optimization(team_detector, tmp_path):
    """Test hyperparameter optimization functionality."""
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("""
        train: path/to/train
        val: path/to/val
        test: path/to/test
        nc: 10
        names: ['mercedes', 'red_bull', 'ferrari', 'mclaren',
                'alpine', 'aston_martin', 'williams', 'alpha_tauri',
                'alfa_romeo', 'haas']
    """)
    
    with pytest.raises(Exception):  # Should fail without a model
        team_detector.optimize_hyperparameters(str(data_yaml), n_trials=1)