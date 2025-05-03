import os
import yaml
import mlflow
import torch
import random
import numpy as np
import time
from pathlib import Path
from ultralytics import YOLO, settings, solutions
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.callbacks.mlflow import on_train_end
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import prepare_datasets 
from src.training.train import TrainingPipeline
from src.visualization.visualize import visualize_results

def train_pipeline(config):
    """
    Runs the complete training pipeline for F1 car, team and driver detection
    """
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    
    # Prepare datasets 
    prepare_datasets()

    EXPERIMENT_NAME = "f1_detection_pipeline"

    # Initialize MLflow
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        print(f" Experiment '{EXPERIMENT_NAME}' not found. Creating new experiment...")
        mlflow.create_experiment(EXPERIMENT_NAME)
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    mlflow.set_experiment(experiment_name=EXPERIMENT_NAME)

    # Initialize model with YOLOv11
    model = YOLO('yolov11s.pt')
    model.add_callback("on_train_end", on_train_end)

    # Initialize training pipeline
    pipeline = TrainingPipeline(config_path='configs/training_configs/training_pipeline.yaml')
    
    # Initialize tensorboard writer
    TENSORBOARD_LOG_DIR = "runs/detect"
    tb_writer = SummaryWriter(TENSORBOARD_LOG_DIR)

    # Add loss logging callback
    class LossLoggingCallback:
        def __init__(self, tb_writer):
            self.tb_writer = tb_writer

        def on_fit_epoch_end(self, trainer: BaseTrainer):
            """Runs at the end of each epoch to log loss."""
            epoch = trainer.epoch
            loss_dict = trainer.loss_items

            if loss_dict is not None and isinstance(loss_dict, torch.Tensor):
                loss_values = loss_dict.tolist()

                if len(loss_values) >= 3:
                    self.tb_writer.add_scalar("Loss/Box", loss_values[0], epoch)
                    self.tb_writer.add_scalar("Loss/Class", loss_values[1], epoch)
                    self.tb_writer.add_scalar("Loss/DFL", loss_values[2], epoch)
                    self.tb_writer.flush()

                    mlflow.log_metric("loss_box", loss_values[0], step=epoch)
                    mlflow.log_metric("loss_class", loss_values[1], step=epoch)
                    mlflow.log_metric("loss_dfl", loss_values[2], step=epoch)

    model.add_callback("on_fit_epoch_end", LossLoggingCallback(tb_writer).on_fit_epoch_end)

    # Run training
    train_params = {
        "data": "configs/f1_car.yaml",
        "epochs": 30,
        "batch": 8, 
        "lr0": 1e-3,
        "optimizer": "AdamW",
        "augment": False,
        "save_period": 0,
        "save": False,
        "verbose": True,
    }

    if mlflow.active_run():
        mlflow.end_run()

    with mlflow.start_run(run_name="YOLOv11_F1_Detection_Pipeline") as run:
        mlflow.log_params(train_params)

        start_time = time.time()
        car_model, team_model, driver_model = pipeline.train_all()
        end_time = time.time()

        mlflow.log_metric("training_time_sec", end_time - start_time)
        mlflow.log_artifact(TENSORBOARD_LOG_DIR, artifact_path="tensorboard_logs")

    print(" Training with MLflow & TensorBoard Completed!")

    tb_writer.close()
    mlflow.end_run()

    return car_model, team_model, driver_model

def main():
    # Load config
    config_path = Path('configs/training_configs/training_pipeline.yaml')
    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Run training pipeline
    car_model, team_model, driver_model = train_pipeline(config)

    # Visualize results
    visualize_results(car_model, team_model, driver_model)

if __name__ == '__main__':
    main()