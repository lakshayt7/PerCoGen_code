from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

import os

from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
)

pipeline = pipeline.to(accelerator.device)
generator = torch.Generator(device=accelerator.device).manual_seed(args.seed) if args.seed else None
images = [
    pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
    for _ in range(args.num_validation_images)
]

base_model_path = "/gpfs/scratch/lt2504/diffusers_out/dreambooth-sdxl-sam/"
controlnet_path = "lllyasviel/sd-controlnet-openpose"

out_folder = "sdxl_3000"

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

sd = StableDiffusionXLPipeline.load_lora_weights(
    pretrained_model_name_or_path_or_dict = base_model_path
)

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)

'''
pipe = StableDiffusionXLControlNetPipeline.load_lora_weights(
    sd, controlnet=controlnet, torch_dtype=torch.float16
)
'''
# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("/gpfs/home/lt2504/dreambooth/Dreambooth-Stable-Diffusion/controlnet/images/control.png")
prompt = "a photo of sks man"

# generate image


generator = torch.manual_seed(0)
images = pipe(
    prompt, num_inference_steps=30, generator=generator, image=control_image, num_images_per_prompt = 20
).images
for i, image in enumerate(images):
    image.save(f"{out_folder}/output_{i}.png")
