import os
import time
import argparse
import random

import cv2
import numpy as np
from PIL import Image

import torch

from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineBaseline
from utils import process_prompt, read_prompts

import warnings
warnings.filterwarnings("ignore")

# use generator to make the sampling deterministic
seed = 0
random.seed(seed)
sd_generator = torch.manual_seed(seed)

#####################

parser = argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=-1, help="Number of prompts to generate. -1 means all prompts")
parser.add_argument("--ref-image-path", type=str, default="/project/infattllm/slida/katz-ae-images/images_sdxl_t2i", help="the path to the reference image folder")
parser.add_argument("--output-image-path", type=str, default=None, help="the path to the output image folder")
parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard"], help="which serve mode to use")
parser.add_argument("--lora-mode", type=str, default="full", choices=["full"], help="which lora mode to use")
parser.add_argument("--load-lora-mode", type=str, default="default", choices=["default"], help="how to load lora")
parser.add_argument("--num-controlnets", type=int, default=1, choices=[0,1,2,3], help="number of controlnets in the pipeline")
parser.add_argument("--num-loras", type=int, default=1, choices=[0,1,2], help="number of controlnets in the pipeline")
parser.add_argument("--skipped-steps", type=int, default=0, choices=[0], help="must be 0")
serve_args = parser.parse_args() 
print("Args", serve_args)

controlnet_conditioning_scale = 0.5  # recommended for good generalization
if serve_args.num_controlnets > 1:
    controlnet_conditioning_scale = [ controlnet_conditioning_scale ] * serve_args.num_controlnets

if serve_args.num_controlnets <= 1:
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    )
elif serve_args.num_controlnets > 1:
    controlnet = [ ControlNetModel.from_pretrained(
        "diffusers/controlnet-depth-sdxl-1.0",
        variant="fp16",
        use_safetensors=True,
        torch_dtype=torch.float16,
    ) for _ in range(serve_args.num_controlnets) ]

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipelineBaseline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
    serve_args=serve_args,
)
pipe = pipe.to("cuda")

##################### Inference #####################
num_prompts = serve_args.num_prompts
prompts = read_prompts(num_prompts=num_prompts)
num_prompts = len(prompts)

if serve_args.num_loras == 1:
    prompt_prefix = "papercut -subject/scene-"
elif serve_args.num_loras == 2:
    prompt_prefix = "by william eggleston, "
elif serve_args.num_loras == 0:
    prompt_prefix = ""
prompt_suffix = ", 4k, clean background"
negative_prompt = "low quality, bad quality, sketches, numbers, letters"

ref_image_folder = serve_args.ref_image_path
assert os.path.isdir(ref_image_folder), "ref image folder not exists"

output_image_folder = serve_args.output_image_path
if output_image_folder is not None:
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

for prompt_id, prompt in enumerate(prompts):
    if prompt_id == 1:
        e2e_serving_start = time.time()
    # Process prompt
    prompt = process_prompt(prompt_prefix, prompt, prompt_suffix)
    print(prompt_id, prompt)

    # Load ref image
    ref_image = cv2.imread(f"{ref_image_folder}/image_{prompt_id}_depth.png")
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_image = ref_image[:, :, None]
    ref_image = np.concatenate([ref_image, ref_image, ref_image], axis=2)
    ref_image = Image.fromarray(ref_image)

    if serve_args.num_controlnets > 1:
        ref_image = [ ref_image ] * serve_args.num_controlnets

    inference_start = time.time()
    ##################### Load LoRAs #####################
    load_lora_start = time.time()
    lora_model_repos = []
    if serve_args.num_loras == 1:
        lora_model_repos = ["TheLastBen/Papercut_SDXL"]
    elif serve_args.num_loras == 2:
        lora_model_repos = ["TheLastBen/Filmic", "TheLastBen/William_Eggleston_Style_SDXL"]
    assert len(lora_model_repos) == serve_args.num_loras, "lora_model_repos: {}".format(lora_model_repos)

    if serve_args.num_loras > 0:
        for lora_id in range(serve_args.num_loras):
            pipe.load_lora_weights(lora_model_repos[lora_id], adapter_name=lora_model_repos[lora_id], serve_args=serve_args)
        pipe.set_adapters(lora_model_repos, adapter_weights=[1.0] * len(lora_model_repos))
        pipe.fuse_lora(lora_scale=1.0)
    load_lora_end = time.time()

    images = pipe(
        prompt, negative_prompt=negative_prompt, image=ref_image, controlnet_conditioning_scale=controlnet_conditioning_scale,
        generater=sd_generator,
        num_inference_steps = 50,
        serve_args=serve_args,
    ).images
    inference_end = time.time()
    print("Load LoRA latency: {:.2f}".format(load_lora_end - load_lora_start))
    print("End2End inference latency: {:.2f}".format(inference_end - inference_start))
    print("==============================")

    # Save images
    if output_image_folder is not None:
        images[0].save(f"{output_image_folder}/image_{prompt_id}.png")

    ##################### Unload LoRAs #####################
    if serve_args.num_loras > 0:
        pipe.unfuse_lora()
        pipe.unload_lora_weights()

e2e_serving_end = time.time()
print("End2End inference latency: {:.2f}".format(e2e_serving_end - e2e_serving_start))
print("Number of prompts: {}".format(len(prompts) - 1))
print("End2End Throughput: {:.2f}".format((e2e_serving_end - e2e_serving_start)/(len(prompts) - 1)))
