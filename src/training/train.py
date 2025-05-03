import os
import yaml
import mlflow
from typing import Dict, Optional, List
import torch
from torch.utils.tensorboard import SummaryWriter
from ..models import CarDetector, TeamDetector, DriverDetector

class TrainingPipeline:
    def __init__(self, config_path: str):
        """
        Initialize training pipeline.
        
        Args:
            config_path (str): Path to training configuration YAML
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            
        # Initialize models
        self.car_detector = CarDetector(
            model_path=self.config.get('car_detector', {}).get('model_path'),
            config_path=self.config.get('car_detector', {}).get('config_path')
        )
        self.team_detector = TeamDetector(
            model_path=self.config.get('team_detector', {}).get('model_path'),
            config_path=self.config.get('team_detector', {}).get('config_path')
        )
        self.driver_detector = DriverDetector(
            model_path=self.config.get('driver_detector', {}).get('model_path'),
            config_path=self.config.get('driver_detector', {}).get('config_path')
        )
        
        # Initialize TensorBoard writer
        self.writer = SummaryWriter(self.config.get('tensorboard_dir', 'runs/f1_detection'))
        
    def train_car_detector(self, optimize: bool = True) -> Dict:
        """Train the car detection model."""
        config = self.config['car_detector']
        
        if optimize and config.get('optimize_hyperparameters', False):
            best_params, _ = self.car_detector.optimize_hyperparameters(
                config['data_yaml'],
                n_trials=config.get('n_trials', 100)
            )
            config.update(best_params)
            
        results = self.car_detector.train_with_mlflow(
            data_yaml=config['data_yaml'],
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 16),
            experiment_name='car_detection',
            **config.get('training_params', {})
        )
        
        return results.results_dict
        
    def train_team_detector(self, optimize: bool = True) -> Dict:
        """Train the team detection model."""
        config = self.config['team_detector']
        
        if optimize and config.get('optimize_hyperparameters', False):
            best_params, _ = self.team_detector.optimize_hyperparameters(
                config['data_yaml'],
                n_trials=config.get('n_trials', 100)
            )
            config.update(best_params)
            
        # Calculate class weights if specified
        if config.get('use_class_weights', False):
            class_weights = self._calculate_class_weights(
                config['data_yaml'],
                len(self.team_detector.class_names)
            )
            config['training_params']['class_weights'] = class_weights
            
        results = self.team_detector.train_with_mlflow(
            data_yaml=config['data_yaml'],
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 16),
            experiment_name='team_detection',
            **config.get('training_params', {})
        )
        
        return results.results_dict
        
    def train_driver_detector(self, optimize: bool = True) -> Dict:
        """Train the driver detection model."""
        config = self.config['driver_detector']
        
        if optimize and config.get('optimize_hyperparameters', False):
            best_params, _ = self.driver_detector.optimize_hyperparameters(
                config['data_yaml'],
                n_trials=config.get('n_trials', 100)
            )
            config.update(best_params)
            
        # Calculate class weights if specified
        if config.get('use_class_weights', False):
            class_weights = self._calculate_class_weights(
                config['data_yaml'],
                len(self.driver_detector.class_names)
            )
            config['training_params']['class_weights'] = class_weights
            
        results = self.driver_detector.train_with_mlflow(
            data_yaml=config['data_yaml'],
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 16),
            experiment_name='driver_detection',
            **config.get('training_params', {})
        )
        
        return results.results_dict
        
    def train_all(self, optimize: bool = True) -> Dict[str, Dict]:
        """Train all models in sequence."""
        results = {
            'car_detection': self.train_car_detector(optimize),
            'team_detection': self.train_team_detector(optimize),
            'driver_detection': self.train_driver_detector(optimize)
        }
        
        # Log overall pipeline metrics
        overall_metrics = {
            'car_map50': results['car_detection']['metrics/mAP50(B)'],
            'team_map50': results['team_detection']['metrics/mAP50(B)'],
            'driver_map50': results['driver_detection']['metrics/mAP50(B)']
        }
        
        with mlflow.start_run(run_name='full_pipeline'):
            mlflow.log_metrics(overall_metrics)
            
        return results
    
    def _calculate_class_weights(self, data_yaml: str, num_classes: int) -> torch.Tensor:
        """Calculate inverse class weights from dataset."""
        import numpy as np
        from collections import Counter
        
        # Load dataset labels
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        train_path = data_config['train']
        if isinstance(train_path, str) and 'labels' in train_path:
            label_files = os.listdir(train_path)
            all_classes = []
            
            for label_file in label_files:
                if label_file.endswith('.txt'):
                    with open(os.path.join(train_path, label_file), 'r') as f:
                        for line in f:
                            class_id = int(line.split()[0])
                            all_classes.append(class_id)
                            
            # Calculate inverse class weights
            class_counts = Counter(all_classes)
            total_samples = sum(class_counts.values())
            weights = torch.zeros(num_classes)
            
            for class_id in range(num_classes):
                count = class_counts.get(class_id, 0)
                if count > 0:
                    weights[class_id] = total_samples / (num_classes * count)
                else:
                    weights[class_id] = 1.0
                    
            return weights
        
        return None