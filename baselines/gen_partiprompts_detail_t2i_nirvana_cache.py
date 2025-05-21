import os
import random
random.seed(0)

import torch

from utils import read_prompts
from diffusers import AutoPipelineForText2Image

import warnings
warnings.filterwarnings("ignore")

seed = 10
sd_generator = torch.manual_seed(seed)

if __name__ == "__main__":
    prompts = read_prompts(num_prompts=-1)

    image_folder = "./images_sdxl_t2i_nirvana_cache"
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0", 
        torch_dtype=torch.float16, 
        variant="fp16", 
        use_safetensors=True
    ).to("cuda")

    for prompt_id, prompt in enumerate(prompts):
        print(f"{prompt_id}: {prompt}")
        image = pipeline_text2image(prompt=prompt, generater=sd_generator).images[0]
        image.save(f"{image_folder}/image_{prompt_id}.png")
