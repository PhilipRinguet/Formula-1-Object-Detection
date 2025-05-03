from typing import Dict, List, Optional, Tuple
import mlflow
import torch
from .base_detector import BaseDetector

class TeamDetector(BaseDetector):
    """F1 team detection model for classifying cars into their respective teams."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        super().__init__(model_path, config_path)
        self.class_names = [
            'mercedes', 'red_bull', 'ferrari', 'mclaren', 
            'alpine', 'aston_martin', 'williams', 'alpha_tauri',
            'alfa_romeo', 'haas'
        ]
        
    def predict_with_confidence(self, image, conf_threshold: float = 0.5):
        """
        Predict team with confidence threshold.
        
        Args:
            image: Input image
            conf_threshold (float): Confidence threshold for predictions
            
        Returns:
            List of predictions above threshold
        """
        results = self.predict(image, conf=conf_threshold)
        return results
        
    def train_with_mlflow(self, 
                         data_yaml: str,
                         epochs: int = 100,
                         batch_size: int = 16,
                         experiment_name: str = "team_detection",
                         **kwargs):
        """
        Train the team detector with MLflow tracking.
        
        Args:
            data_yaml (str): Path to data.yaml file
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            experiment_name (str): MLflow experiment name
        """
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run():
            mlflow.log_params({
                'model_type': 'YOLOv11',
                'epochs': epochs,
                'batch_size': batch_size,
                'num_classes': len(self.class_names),
                **kwargs
            })
            
            # Train with class weights to handle imbalance
            class_weights = kwargs.pop('class_weights', None)
            if class_weights:
                kwargs['cls_weights'] = torch.tensor(class_weights)
            
            results = self.train(
                data=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                **kwargs
            )
            
            # Log metrics
            metrics = {
                'mAP50': results.results_dict['metrics/mAP50(B)'],
                'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
                'precision': results.results_dict['metrics/precision(B)'],
                'recall': results.results_dict['metrics/recall(B)']
            }
            
            # Log per-class metrics if available
            for i, class_name in enumerate(self.class_names):
                if f'metrics/precision({i})' in results.results_dict:
                    metrics[f'{class_name}_precision'] = results.results_dict[f'metrics/precision({i})']
                    metrics[f'{class_name}_recall'] = results.results_dict[f'metrics/recall({i})']
            
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(self.model.last)
            
        return results
        
    def optimize_hyperparameters(self, data_yaml: str, n_trials: int = 100):
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            data_yaml (str): Path to data.yaml file
            n_trials (int): Number of optimization trials
        """
        import optuna
        
        def objective(trial):
            # Define hyperparameter search space
            params = {
                'lr0': trial.suggest_float('lr0', 1e-5, 1e-3, log=True),
                'lrf': trial.suggest_float('lrf', 0.01, 0.1),
                'momentum': trial.suggest_float('momentum', 0.8, 0.99),
                'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.001),
                'warmup_epochs': trial.suggest_int('warmup_epochs', 1, 5),
                'warmup_momentum': trial.suggest_float('warmup_momentum', 0.5, 0.95),
                # Classification specific parameters
                'cls': trial.suggest_float('cls', 0.2, 2.0),
                'cls_pw': trial.suggest_float('cls_pw', 0.5, 2.0),
                'cls_tau': trial.suggest_float('cls_tau', 0.0, 1.0),
            }
            
            try:
                results = self.train(
                    data=data_yaml,
                    epochs=1,
                    batch_size=16,
                    **params
                )
                return results.results_dict['metrics/mAP50(B)']
            except Exception as e:
                return float('-inf')
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value