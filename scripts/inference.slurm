#!/bin/bash
#SBATCH --job-name=collect_activations     # create a short name for your job
#SBATCH --nodes=1                          # node count
#SBATCH --ntasks=1                         # total number of tasks across all nodes
#SBATCH --gres=gpu:1                       # number of gpus per node   
#SBATCH --time=01:20:00                    # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=32G                  # memory per cpu-core (4G is default)
#SBATCH --output slurm/inference.out       # location to save output
#SBATCH --account=jdc                      # account for compute

module purge
module load anaconda3/2024.6
conda activate transformers
export HYDRA_FULL_ERROR=1  # show full error traceback

python run_vlm.py task="scene_description" model="qwen" model.probe_layers=null