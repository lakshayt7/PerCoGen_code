# Grounded-ControlNet Instructions

The code is structured as a fork of the diffusers library.

Use the following command to install dependencies

    pip install -e . 

The training script for the grounded-ControlNet model is available as `examples/controlNet/train_controlnet_grounded.py`

The grounded ControlNet model can be used using the `StableDiffusionControlNetGroundedPipeline` similar to the ControlNet pipeline


## Generate Dataset.
1. Setup [GLIP](https://github.com/microsoft/GLIP) for generating grounding data.
2. Setup [CLIP](https://github.com/openai/CLIP.git) for choosing best caption among available.
3. Install [ControlNet_AUX](https://github.com/patrickvonplaten/controlnet_aux) if you want openpose style skeleton support, else use the COCO skeletons.
4. Download [COCO2014](https://cocodataset.org/#download).
5. Use `personalize_gen/make_dataset_grounded.py` to generate dataset. This generated `coco14.json` for HF style dataset.
6. Use `personalize_gen/coco14.py`  and combine with generated `coco14.json` to generated HuggingFace Dataset.

