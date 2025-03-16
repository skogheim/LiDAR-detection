# Importing necessary libraries
from ultralytics import YOLO  # Import YOLO model from Ultralytics library
import os  # Import OS library for interacting with the operating system
import wandb  # Import Weights & Biases for experiment tracking and optimization

def setup_directories(base_dir):
    """
    Create necessary directories for storing weights and logs within the base directory.
    
    Args:
    base_dir (str): The root directory under which to create subdirectories.
    
    Creates:
    - base_dir: Main directory if it does not exist.
    - weights: Subdirectory to store model weights.
    - logs: Subdirectory to store training logs.
    """
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'weights'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'logs'), exist_ok=True)

def train_and_tune():
    """
    Initialize the Weights & Biases project, set up directories, perform hyperparameter tuning, and train the YOLO model.
    Utilizes early stopping based on validation metrics to prevent overfitting.
    """
    # Initialize a new Weights & Biases run with a configuration dictionary
    wandb.init(project="FINAL_RUN", entity="Skogheim", config={
        "epochs": 100,
        "batch_size": 32,
        "image_size": 640
    })

    # Base directory for the run
    base_dir = 'yolov8_mainRUN'
    setup_directories(base_dir)

    # Load a pretrained YOLO model
    model = YOLO('yolov8s.pt')
    data_config = 'config.yaml'  # Path to the dataset configuration file

    # Perform quick hyperparameter tuning before training
    print("Starting quick hyperparameter tuning...")
    model.tune(data=data_config, epochs=3, iterations=8, optimizer='AdamW', plots=True, save=False, val=True, name='hyperparam_tuning')
    print("Hyperparameter tuning completed.")

    # Start the training process with optimized hyperparameters
    print("Starting training with potentially better hyperparameters...")
    best_map50 = 0  # Initialize the best mAP@0.5 metric
    best_model_path = os.path.join(base_dir, 'weights', 'best_model.pt')
    no_improve_epoch = 0
    early_stopping_threshold = 15  # Threshold for early stopping

    # Train the model for a configured number of epochs
    for epoch in range(1, 101):
        model.train(data=data_config, epochs=1, batch=32, imgsz=640, name=f'training_epoch_{epoch}', exist_ok=True)
        val_metrics = model.val(data=data_config, batch=32, imgsz=640, name=f'validation_epoch_{epoch}', exist_ok=True)
        current_map50 = val_metrics.box.map50

        # Save the best model based on mAP@0.5 metric
        if current_map50 > best_map50:
            best_map50 = current_map50
            model.save(best_model_path)
            print(f"Epoch {epoch}: mAP@0.5 {current_map50}. Best model saved at {best_model_path}.")
            no_improve_epoch = 0
        else:
            no_improve_epoch += 1
            if no_improve_epoch >= early_stopping_threshold:
                print("Early stopping initiated.")
                break

    # Finish the Weights & Biases run
    wandb.finish()

if __name__ == "__main__":
    train_and_tune()
