#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=250G
#SBATCH --job-name="yolov8-2"
#SBATCH --output=yolov8.out


cd "${SLURM_SUBMIT_DIR}"

module purge
module load Anaconda3/2023.09-0
conda activate tdt4265

srun python main.py