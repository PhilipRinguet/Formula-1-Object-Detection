import os
import yaml
import mlflow
import torch
import random
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from src.data.dataset import prepare_datasets
from src.training.train import optimize_hyperparameters, train_model
from src.visualization.visualize import visualize_results


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def setup_mlflow(config: dict) -> None:
    """Setup MLflow tracking."""
    mlflow.set_experiment(config['mlflow_experiment_name'])
    
def train_pipeline(config: dict) -> tuple:
    """Run the complete training pipeline."""
    # Prepare datasets
    prepare_datasets()
    
    # Train car detector
    print("Training car detector...")
    if config['car_detector']['optimize_hyperparameters']:
        car_params = optimize_hyperparameters(
            base_model=config['car_detector']['model_path'],
            data_yaml=config['car_detector']['data_yaml'],
            n_trials=config['car_detector']['n_trials']
        )
    else:
        car_params = config['car_detector']['training_params']
    
    car_model = train_model(
        model_path=config['car_detector']['model_path'],
        data_yaml=config['car_detector']['data_yaml'],
        params=car_params,
        epochs=config['car_detector']['epochs']
    )
    
    # Train team detector
    print("Training team detector...")
    if config['team_detector']['optimize_hyperparameters']:
        team_params = optimize_hyperparameters(
            base_model=car_model.best,  # Use best car detector as base
            data_yaml=config['team_detector']['data_yaml'],
            n_trials=config['team_detector']['n_trials']
        )
    else:
        team_params = config['team_detector']['training_params']
    
    team_model = train_model(
        model_path=car_model.best,
        data_yaml=config['team_detector']['data_yaml'],
        params=team_params,
        epochs=config['team_detector']['epochs']
    )
    
    # Train driver detector
    print("Training driver detector...")
    if config['driver_detector']['optimize_hyperparameters']:
        driver_params = optimize_hyperparameters(
            base_model=team_model.best,  # Use best team detector as base
            data_yaml=config['driver_detector']['data_yaml'],
            n_trials=config['driver_detector']['n_trials']
        )
    else:
        driver_params = config['driver_detector']['training_params']
    
    driver_model = train_model(
        model_path=team_model.best,
        data_yaml=config['driver_detector']['data_yaml'],
        params=driver_params,
        epochs=config['driver_detector']['epochs']
    )
    
    return car_model, team_model, driver_model

def export_models(car_model, team_model, driver_model, format='onnx'):
    """Export trained models to specified format."""
    print(f"Exporting models to {format}...")
    car_model.export(format=format)
    team_model.export(format=format)
    driver_model.export(format=format)

def main():
    # Load configuration
    config_path = 'configs/training_configs/training_pipeline.yaml'
    config = load_config(config_path)
    
    # Set random seed
    set_seed(config['seed'])
    
    # Setup MLflow
    setup_mlflow(config)
    
    # Run training pipeline
    with mlflow.start_run(run_name="complete_pipeline"):
        car_model, team_model, driver_model = train_pipeline(config)
        
        # Export models
        export_models(car_model, team_model, driver_model)
        
        # Visualize results
        visualize_results(car_model, team_model, driver_model)

if __name__ == "__main__":
    main()