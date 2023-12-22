#Dreambooth finetuning of ControlNet code

This is the code for dreambooth finetuning of ControlNet model.

All of the slurm scripts are organized where they can be run by editing the following variables in the bash file

    export INSTANCE_DIR="location/to/images/of/instance/"
    export OUTPUT_DIR="location/where/model/should/be/stored/after/training"
    export MODEL_NAME="Name/of/model/for/finetuning/sd1.5/2"

To run the simple dreambooth finetuning use the script after editing the above variables

    sbatch train_dream.sh


For dreambooth finetuning of sdxl using LORA use

    sbatch train_dream_sdxl.sh

For dreambooth finetuning of ControlNet model after unfreezing controlnet layers use the script after editing the above variables

    sbatch train_dream_unfreeqe.sh


