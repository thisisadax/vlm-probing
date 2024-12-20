#!/bin/bash
#SBATCH --job-name=probe_activations       # create a short name for your job
#SBATCH --nodes=1                          # node count
#SBATCH --ntasks=1                         # total number of tasks across all nodes
#SBATCH --gres=gpu:1                       # number of gpus per node
#SBATCH --time=05:00:00                    # total run time limit (HH:MM:SS)
#SBATCH --mem-per-cpu=128G                 # memory per cpu-core (4G is default)
#SBATCH --output slurm/both.out            # location to save output
#SBATCH --account=jdc                      # account for compute

module purge
module load anaconda3/2024.6
conda activate transformers
export HYDRA_FULL_ERROR=1  # show full error traceback

python run_vlm.py task="scene_description" model="qwen" model.probe_layers=null

python run_probes.py dataset="tpr_classification" \
        dataset.task_name="scene_description" \
        probe="pooled_tpr_probe" \
        callbacks.model_checkpoint.monitor="val_acc" \
        trainer.max_epochs=5 \
        probe.lr=5e-4 \
        dataset.batch_size=512 \
        dataset.layer_name=null

