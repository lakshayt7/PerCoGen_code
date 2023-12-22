# Grounded-ControlNet Instructions

The code is structured as a fork of the diffusers library.

Use the following command to install dependencies

    pip install -e . 




## Generate Dataset.
1. Setup [GLIP](https://github.com/microsoft/GLIP) for generating grounding data.
2. Setup [CLIP](https://github.com/openai/CLIP.git) for choosing best caption among available.
3. Install [ControlNet_AUX](https://github.com/patrickvonplaten/controlnet_aux) if you want openpose style skeleton support, else use the COCO skeletons.
4. Download [COCO2014](https://cocodataset.org/#download).
5. Use `make_dataset_grounded.py` from [from](https://github.com/ravich3373/personalize_gen) to generate dataset. This generated `coco14.json` for HF style dataset.
6. Use `coco14.py` from the same to repo and combine with generated `coco14.json` to generated HuggingFace Dataset.

