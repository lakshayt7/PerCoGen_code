#!/bin/bash
#SBATCH --nodes=1
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=16G
#SBATCH --output=logs/train_dreambooth_%j.out
#SBATCH --error=logs/train_dreambooth%j.err
#SBATCH --gres=gpu:a100:1

echo -e "GPUS = $CUDA_VISIBLE_DEVICES\n"
nvidia-smi

source /gpfs/scratch/lt2504/bert/bin/activate

python inference_sdxl.py