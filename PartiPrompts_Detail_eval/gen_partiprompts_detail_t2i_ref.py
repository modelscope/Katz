import random
random.seed(0)
import os

from diffusers import AutoPipelineForText2Image
import torch
import numpy as np

from PIL import Image
from diffusers.utils import load_image

import warnings
warnings.filterwarnings("ignore")

seed = 0
sd_generator = torch.manual_seed(seed)

from transformers import DPTFeatureExtractor, DPTForDepthEstimation
depth_estimator = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas").to("cuda")
feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-hybrid-midas")
def get_depth_map(image):
    image = feature_extractor(images=image, return_tensors="pt").pixel_values.to("cuda")
    with torch.no_grad(), torch.autocast("cuda"):
        depth_map = depth_estimator(image).predicted_depth

    depth_map = torch.nn.functional.interpolate(
        depth_map.unsqueeze(1),
        size=(1024, 1024),
        mode="bicubic",
        align_corners=False,
    )
    depth_min = torch.amin(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_max = torch.amax(depth_map, dim=[1, 2, 3], keepdim=True)
    depth_map = (depth_map - depth_min) / (depth_max - depth_min)
    image = torch.cat([depth_map] * 3, dim=1)

    image = image.permute(0, 2, 3, 1).cpu().numpy()[0]
    image = Image.fromarray((image * 255.0).clip(0, 255).astype(np.uint8))
    return image

def process_prompt(prefix, prompt, suffix):
    return prefix + prompt + suffix

def read_prompts(num_prompts=30):
    prompts = []
    with open("./PartiPrompts_Detail.tsv", 'r') as fr:
        for line in fr:
            parts = line.strip().split("\t")
            assert len(parts) >= 3, parts
            prompts.append(parts[0])
        random.shuffle(prompts)
    
    return prompts[:num_prompts]

if __name__ == "__main__":
    prompts = read_prompts(num_prompts=600)

    image_folder = "./images_sdxl_t2i"
    if not os.path.isdir(image_folder):
        os.mkdir(image_folder)
    # else:
    #     print("{} exists".format(image_folder))

    args = {
        "model_name": "stabilityai/stable-diffusion-xl-base-1.0",
        "prompt_prefix": "",
        "prompt_suffix": "",
    }
    print("args", args)

    pipeline_text2image = AutoPipelineForText2Image.from_pretrained(
        args["model_name"], torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    ).to("cuda")

    for prompt_id, prompt in enumerate(prompts):
        prompt = process_prompt(args["prompt_prefix"], prompt, args["prompt_suffix"])
        print(prompt_id, prompt)
        image = pipeline_text2image(prompt=prompt, generater=sd_generator).images[0]
        image.save(f"{image_folder}/image_{prompt_id}.png")

        # Generate depth reference image
        image = load_image(f"{image_folder}/image_{prompt_id}.png")
        depth_image = get_depth_map(image)
        depth_image.save(f"{image_folder}/image_{prompt_id}_depth.png")    