#!/bin/bash
#SBATCH --job-name=probe_activations       # create a short name for your job
#SBATCH --nodes=1                          # node count
#SBATCH --ntasks=1                         # total number of tasks across all nodes
#SBATCH --gres=gpu:1                       # number of gpus per node
#SBATCH --time=08:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=128G                 # memory per cpu-core (4G is default)
#SBATCH --output slurm/probe.out           # location to save output
#SBATCH --account=jdc                      # account for compute

module purge
module load anaconda3/2024.6
conda activate transformers
export HYDRA_FULL_ERROR=1  # show full error traceback

python run_probes.py \
        dataset="tpr_classification" \
        probe="pooled_tpr_probe" \
        dataset.task_name="search" \
        callbacks.model_checkpoint.monitor="val_acc" \
        trainer.max_epochs=100 \
        probe.lr=5e-4 \
        dataset.batch_size=256 \
        dataset.layer_name=null 