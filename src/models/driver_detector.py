from typing import Dict, List, Optional, Tuple
import mlflow
import torch
from .base_detector import BaseDetector

class DriverDetector(BaseDetector):
    """F1 driver detection model for identifying individual drivers within teams."""
    
    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        super().__init__(model_path, config_path)
        # Initialize team-driver mapping
        self.team_drivers = {
            'mercedes': ['hamilton', 'russell'],
            'red_bull': ['verstappen', 'perez'],
            'ferrari': ['leclerc', 'sainz'],
            'mclaren': ['norris', 'piastri'],
            'alpine': ['gasly', 'ocon'],
            'aston_martin': ['alonso', 'stroll'],
            'williams': ['albon', 'sargeant'],
            'alpha_tauri': ['tsunoda', 'ricciardo'],
            'alfa_romeo': ['bottas', 'zhou'],
            'haas': ['magnussen', 'hulkenberg']
        }
        
        # Flatten driver names for model classes
        self.class_names = [
            driver for drivers in self.team_drivers.values()
            for driver in drivers
        ]
        
    def predict_with_hierarchy(self, image, team_conf: float = 0.5, driver_conf: float = 0.3):
        """
        Predict drivers using hierarchical detection (team first, then driver).
        
        Args:
            image: Input image
            team_conf (float): Confidence threshold for team detection
            driver_conf (float): Confidence threshold for driver detection
            
        Returns:
            List of predictions with team and driver information
        """
        results = self.predict(image, conf=driver_conf)
        
        # Post-process predictions to ensure team-driver consistency
        processed_results = []
        for pred in results:
            cls_id = pred.cls
            conf = pred.conf
            driver_name = self.class_names[int(cls_id)]
            
            # Find team for driver
            team = next(
                (team for team, drivers in self.team_drivers.items() 
                 if driver_name in drivers),
                None
            )
            
            if team and conf >= driver_conf:
                processed_results.append({
                    'team': team,
                    'driver': driver_name,
                    'confidence': float(conf),
                    'box': pred.boxes.xyxy.tolist()[0]
                })
                
        return processed_results
        
    def train_with_mlflow(self, 
                         data_yaml: str,
                         epochs: int = 100,
                         batch_size: int = 16,
                         experiment_name: str = "driver_detection",
                         **kwargs):
        """
        Train the driver detector with MLflow tracking.
        
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
            
            # Handle class imbalance with weights
            class_weights = kwargs.pop('class_weights', None)
            if class_weights:
                kwargs['cls_weights'] = torch.tensor(class_weights)
            
            results = self.train(
                data=data_yaml,
                epochs=epochs,
                batch_size=batch_size,
                **kwargs
            )
            
            # Log overall metrics
            metrics = {
                'mAP50': results.results_dict['metrics/mAP50(B)'],
                'mAP50-95': results.results_dict['metrics/mAP50-95(B)'],
                'precision': results.results_dict['metrics/precision(B)'],
                'recall': results.results_dict['metrics/recall(B)']
            }
            
            # Log per-driver metrics
            for i, driver_name in enumerate(self.class_names):
                if f'metrics/precision({i})' in results.results_dict:
                    metrics[f'{driver_name}_precision'] = results.results_dict[f'metrics/precision({i})']
                    metrics[f'{driver_name}_recall'] = results.results_dict[f'metrics/recall({i})']
            
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
                # Additional parameters for fine-grained classification
                'mosaic': trial.suggest_float('mosaic', 0.0, 1.0),
                'mixup': trial.suggest_float('mixup', 0.0, 1.0),
                'copy_paste': trial.suggest_float('copy_paste', 0.0, 1.0),
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
    
    def process_video_with_tracking(self, video_path: str, output_path: str, **kwargs):
        """
        Process video with object tracking for consistent driver identification.
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save processed video
        """
        if self.model is None:
            raise ValueError("No model loaded")
            
        return self.model.track(
            source=video_path,
            save=True,
            project=output_path,
            tracker="bytetrack",  # Use ByteTrack for better tracking
            **kwargs
        )