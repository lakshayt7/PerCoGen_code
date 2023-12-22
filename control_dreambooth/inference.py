from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from diffusers.utils import load_image
import torch

base_model_path = "/gpfs/scratch/lt2504/diffusers_out/dreambooth-control-1500step-prior-preserve/"
controlnet_path = "lllyasviel/sd-controlnet-openpose"

controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path, controlnet=controlnet, torch_dtype=torch.float16
)

# speed up diffusion process with faster scheduler and memory optimization
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

# memory optimization.
pipe.enable_model_cpu_offload()

control_image = load_image("/gpfs/home/lt2504/dreambooth/Dreambooth-Stable-Diffusion/controlnet/images/control.png")
prompt = "a photo of sks"

# generate image


generator = torch.manual_seed(0)
images = pipe(
    prompt, num_inference_steps=30, generator=generator, image=control_image, num_images_per_prompt = 20
).images
for i, image in enumerate(images):
    image.save(f"gen_images_1500_prior_preserve/output_{i}.png")
