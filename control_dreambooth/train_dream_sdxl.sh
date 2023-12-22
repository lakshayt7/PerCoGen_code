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

export INSTANCE_DIR="/gpfs/home/lt2504/dreambooth/Dreambooth-Stable-Diffusion/training_images/Sam_Altman"
export OUTPUT_DIR="/gpfs/scratch/lt2504/diffusers_out/dreambooth-sdxl-sam-5000"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_PATH="madebyollin/sdxl-vae-fp16-fix"

source /gpfs/scratch/lt2504/bert/bin/activate

accelerate launch train_dreambooth_lora_sdxl.py  --pretrained_model_name_or_path=$MODEL_NAME    --instance_data_dir=$INSTANCE_DIR --class_data_dir="class_dir"  --output_dir=$OUTPUT_DIR   --instance_prompt="image of <sks>"  --train_batch_size=1   --gradient_accumulation_steps=1   --learning_rate=5e-6   --lr_scheduler="constant"   --lr_warmup_steps=0   --max_train_steps=6000  --with_prior_preservation   --mixed_precision="fp16"  --resolution=1024 --pretrained_vae_model_name_or_path=$VAE_PATH   --validation_prompt="image of <sks> realistic,high quality" --validation_epochs=25  --class_prompt="high quality, realistic image of a man"

