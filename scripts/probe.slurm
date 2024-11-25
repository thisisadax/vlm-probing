#!/bin/bash
#SBATCH --job-name=probe_activations       # create a short name for your job
#SBATCH --nodes=1                          # node count
#SBATCH --ntasks=1                         # total number of tasks across all nodes
#SBATCH --gres=gpu:1                       # number of gpus per node
#SBATCH --constraint=gpu80                 # gpu model   
#SBATCH --time=00:20:00                    # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=32G                  # memory per cpu-core (4G is default)
#SBATCH --output slurm/probe.out           # location to save output
#SBATCH --account=jdc                      # account for compute

module purge
module load anaconda3/2024.6
conda activate transformers

python run_probes.py dataset="counting" probe="pooled_probe"