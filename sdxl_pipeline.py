import os
import time
import random
import argparse
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

import torch
import torch.distributed as dist

from lora_utils import patch_on_lora
from baselines.utils import process_prompt, read_prompts

from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineKatz

import warnings
warnings.filterwarnings("ignore")


seed = 0
sd_generator = torch.manual_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

num_gpus = torch.cuda.device_count()
print(f"Number of GPUs: {num_gpus}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"Available GPU {i}: {torch.cuda.get_device_properties(i)}")

# use generator to make the sampling deterministic
seed = 0
sd_generator = torch.manual_seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser()
parser.add_argument("--num-prompts", type=int, default=-1, help="Number of prompts to generate. -1 means all prompts")
parser.add_argument("--ref-image-path", type=str, default="/project/infattllm/slida/katz-ae-images/images_sdxl_t2i", help="the path to the reference image folder")
parser.add_argument("--output-image-path", type=str, default=None, help="the path to the output image folder")
parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard", "zmq", "nvlink", "without"], help="which serve mode to use")  # deprecated
# distributed
parser.add_argument("--world-size", type=int, default=2, help="world-size")
parser.add_argument("--gpu-id", type=int, default=0, help="which gpu to use")
parser.add_argument("--rank", type=int, default=0)
# ControlNet
parser.add_argument("--controlnet-parallel", action="store_true")
parser.add_argument("--controlnet-parallel-rank", type=int, nargs='+', default=[])  # Example: [2, 4, 6]
parser.add_argument("--num-controlnets", type=int, default=0)
# LoRA
parser.add_argument("--lora-mode", type=str, default="without", choices=["without", "full", "sync"])
parser.add_argument("--load-lora-mode", type=str, default="default", choices=["default", "async"])
parser.add_argument("--max-lora-num", type=int, default=0, choices=[0, 1, 2])
# Base Model
parser.add_argument("--latent-parallel", action="store_true")
# others
parser.add_argument("--verbose", "-v", action="store_true")
serve_args = parser.parse_args() 
print(f"Args: {serve_args}")

if serve_args.latent_parallel or serve_args.controlnet_parallel:
    assert serve_args.rank == 0, "This script should be run on GPU 0"
    if serve_args.controlnet_parallel:
        assert len(serve_args.controlnet_parallel_rank) == serve_args.num_controlnets, "controlnet_parallel_rank must have the same length as num_controlnets"
    torch.cuda.set_device(serve_args.gpu_id)
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8000', rank=serve_args.rank, world_size=serve_args.world_size)
    print(f"World size: {torch.distributed.get_world_size()}, rank: {torch.distributed.get_rank()}")

enable_channels_last = os.getenv("ENABLE_CHANNELS_LAST", "0") == "1"

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipelineKatz.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    serve_args=serve_args,
)
pipe = pipe.to("cuda")
if enable_channels_last:
    pipe.to(memory_format=torch.channels_last)

# copy state_dict to restore later
unet_state_dict_copy = deepcopy(pipe.unet.state_dict())
text_encoder_state_dict_copy = deepcopy(pipe.text_encoder.state_dict())
text_encoder_2_state_dict_copy = deepcopy(pipe.text_encoder_2.state_dict())

unet_state_dict_file = "default_unet_state_dict.pt"
if enable_channels_last:
    unet_state_dict_file = "default_unet_state_dict_channels_last.pt"
default_unet_state_dict = torch.load(unet_state_dict_file)
for key in default_unet_state_dict:
    default_unet_state_dict[key] = default_unet_state_dict[key].to("cuda")
print("default_unet_state_dict length", len(default_unet_state_dict.keys()))
pipe.unet.load_state_dict(deepcopy(default_unet_state_dict), strict=False)

ref_image_folder = serve_args.ref_image_path
assert os.path.isdir(ref_image_folder), "ref image folder not exists"

num_prompts = serve_args.num_prompts

prompts = read_prompts(num_prompts=num_prompts)
num_prompts = len(prompts)
prompt_prefix = ""
if serve_args.max_lora_num == 1:
    prompt_prefix = "papercut -subject/scene-"
elif serve_args.max_lora_num == 2:
    prompt_prefix = "by william eggleston, "
prompt_suffix = ", 4k, clean background"
negative_prompt = "low quality, bad quality, sketches, numbers, letters"

# output_image_folder = "./output_images"
output_image_folder = serve_args.output_image_path
if output_image_folder is not None:
    if not os.path.exists(output_image_folder):
        os.makedirs(output_image_folder)

for prompt_id, prompt in enumerate(prompts):
    if prompt_id == 1:
        e2e_serving_start = time.time()

    unet_state_dict_copy = deepcopy(default_unet_state_dict)

    # Process prompt
    prompt = process_prompt(prompt_prefix, prompt, prompt_suffix)

    ref_image_path = f"{ref_image_folder}/image_{prompt_id}_depth.png"
    print(f"Prompt: {prompt}")
    print(f"Reference image: {ref_image_path}")
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2GRAY)
    ref_image = ref_image[:, :, None]
    ref_image = np.concatenate([ref_image, ref_image, ref_image], axis=2)
    ref_image = Image.fromarray(ref_image)

    inference_start = time.time()

    # Load LoRAs
    load_lora_start = time.time()

    lora_model_repos = []
    if serve_args.max_lora_num == 1:
        lora_model_repos = ["TheLastBen/Papercut_SDXL"]
    elif serve_args.max_lora_num == 2:
        lora_model_repos = ["TheLastBen/William_Eggleston_Style_SDXL", "TheLastBen/Filmic"]
    if serve_args.lora_mode == "full":
        patch_on_lora(pipe, lora_model_repos)

    load_lora_end = time.time()

    images = pipe(
        prompt, negative_prompt=negative_prompt,
        image=ref_image,
        controlnet_conditioning_scale=0.5,  # recommended for good generalization
        generater=sd_generator,
        num_inference_steps=50,
        lora_model_repos=lora_model_repos,
        default_unet_state_dict=unet_state_dict_copy,
    ).images

    inference_end = time.time()

    print("Load LoRA latency: {:.2f}".format(load_lora_end - load_lora_start))
    print("End2End inference latency: {:.2f}".format(inference_end - inference_start))
    print("==============================")

    # Save images
    if output_image_folder is not None:
        output_image_path = f"{output_image_folder}/image_{prompt_id}.png"
        images[0].save(output_image_path)

    # Unload LoRAs
    if serve_args.load_lora_mode == "default":
        pipe.unfuse_lora()
        pipe.unload_lora_weights()
    elif serve_args.load_lora_mode == "async":
        # restore parameters and shm
        pipe.clear_shm()
    # pipe.unet.load_state_dict(unet_state_dict_copy)
    pipe.unet.load_state_dict(deepcopy(default_unet_state_dict), strict=False)

    if serve_args.max_lora_num == 2:
        pipe.text_encoder.load_state_dict(text_encoder_state_dict_copy)
        pipe.text_encoder_2.load_state_dict(text_encoder_2_state_dict_copy)

e2e_serving_end = time.time()
print(f"End2End inference latency: {e2e_serving_end - e2e_serving_start:.2f}")
print(f"Number of prompts: {num_prompts}")
print(f"End2End Throughput: {(e2e_serving_end - e2e_serving_start)/num_prompts:.2f}")

# stop all lora loaders
if serve_args.load_lora_mode == "async":
    for i in range(serve_args.max_lora_num):
        pipe.shm_dict["start_loading_flag_np"][i] = 100
    print("Shut down all LoRA loaders", pipe.shm_dict["start_loading_flag_np"])
