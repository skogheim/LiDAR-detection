# YOLOv8 Object Detection with LiDAR Data

This project utilizes YOLOv8 for object detection in LiDAR datasets, specifically designed to leverage the capabilities of the NAPLab-LiDAR Dataset. The model is trained using customized augmentations to improve detection accuracy on sparse and scale-sensitive LiDAR data.

## This file structure
 **Code**:
    Include the code relvants part for this project, such ass ass the traning scripts, config, parameters etc...

 **Stats**:
    Includes the stats from the traning both form the best result and graphs showing results elvolve thrue the whole traing, and the stats from the hyperparameters tuning

## Installation

Follow these instructions to set up the environment and install all necessary dependencies:

1. **Install Anaconda or Miniconda**:  
   Download and install Anaconda or Miniconda from [here](https://www.anaconda.com/products/distribution).

2. **Create a Conda Environment**:  
   conda create --name tdt4265 python=3.8

3. **Activate the Environment**:
    conda activate tdt4265

4. **Install Dependencies**:
    pip install -r requirements.txt

## Configuration

The project uses a YAML configuration file for setting up data paths and model parameters. Update the config.yaml with the correct paths to your dataset and desired training parameters.

## Running the Model

Submit the job to a SLURM scheduler with a configured environment on a cluster:

- **Load the SLURM Script**:
    Review and modify the SLURM script run.slurm as necessary, adjusting job names, partition, and resource requests according to your cluster's configuration.

- **Submit the Job**:
    sbatch run.slurm

- **Output**
    The script will output logs to yolov8.out, and the best model weights will be saved in the specified directory. Check these outputs to monitor the training progress and results.

## Project Structure
- **run.slrum**: SLURM script to run the training on a GPU-enabled cluster.
- **config.yaml**: Contains all configuration settings for training and validation.
- **main.py**: The main Python script that initializes and starts the training process.
- **requirements.txt**: Lists all Python dependencies.

