from typing import Dict, List, Optional, Tuple
import mlflow
from .base_detector import BaseDetector

class CarDetector(BaseDetector):
    """F1 car detection model specialized for detecting Formula 1 cars in video frames."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        super().__init__(model_path, config_path)
        self.class_names = ['f1_car']
        
    def train_with_mlflow(self, 
                         data_yaml: str,
                         epochs: int = 100,
                         batch_size: int = 16,
                         experiment_name: str = "car_detection",
                         **kwargs):
        """
        Train the car detector with MLflow tracking.
        
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
                **kwargs
            })
            
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
            mlflow.log_metrics(metrics)
            
            # Log model
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
                'box': trial.suggest_float('box', 5.0, 10.0),
                'cls': trial.suggest_float('cls', 0.1, 1.0),
                'hsv_h': trial.suggest_float('hsv_h', 0.0, 0.1),
                'hsv_s': trial.suggest_float('hsv_s', 0.0, 0.9),
                'hsv_v': trial.suggest_float('hsv_v', 0.0, 0.9),
            }
            
            # Train with current hyperparameters
            try:
                results = self.train(
                    data=data_yaml,
                    epochs=1,  # Use fewer epochs for optimization
                    batch_size=16,
                    **params
                )
                return results.results_dict['metrics/mAP50(B)']
            except Exception as e:
                return float('-inf')
        
        # Create study
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        return study.best_params, study.best_value