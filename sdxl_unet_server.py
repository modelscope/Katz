import os
import time
import json
import random
import argparse

from copy import deepcopy

import numpy as np

import torch
import torch.distributed as dist

from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineKatz

from lora_utils import patch_on_lora

total_gpu_num = torch.cuda.device_count()
assert total_gpu_num >= 2, "Must have more than one GPU"
print(f"Total GPUs in the world: {total_gpu_num}")
print(f"torch.distributed.is_available(): {torch.distributed.is_available()}")
print(f"torch.distributed.is_nccl_available(): {torch.distributed.is_nccl_available()}")

parser = argparse.ArgumentParser()
# distributed
parser.add_argument("--world-size", type=int, default=2, help="world size")
parser.add_argument("--gpu-id", type=int, default=1, help="which gpu to use")
parser.add_argument("--rank", type=int, default=1)
# ControlNet
parser.add_argument("--controlnet-parallel", action="store_true")
parser.add_argument("--controlnet-parallel-rank", type=int, nargs='+', default=[]) # Example: [3, 5, 7]
parser.add_argument("--num-controlnets", type=int, default=0)
# LoRA
parser.add_argument("--lora-mode", type=str, default="without", choices=["without", "full", "sync"])
parser.add_argument("--load-lora-mode", type=str, default="default", choices=["default", "async"])
parser.add_argument("--max-lora-num", type=int, default=0, choices=[0, 1, 2])
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args()
args.latent_parallel = True
print(f"Args: {args}")
assert args.rank == 1 and args.gpu_id == 1, "Currently, this script should be run on GPU 1."

torch.cuda.set_device(args.gpu_id)
gpu_device = torch.cuda.current_device()

dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8000', rank=args.rank, world_size=args.world_size)
print(f"World size: {dist.get_world_size()}, rank: {dist.get_rank()}")

enable_channels_last = os.getenv("ENABLE_CHANNELS_LAST", "0") == "1"

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)

args.serve_mode = "standard"
pipe = StableDiffusionXLControlNetPipelineKatz.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    # use_safetensors=True,
    serve_args=args,
)
pipe = pipe.to(gpu_device)
pipe.controlnet = pipe.controlnet.to(gpu_device)
pipe.unet = pipe.unet.to(gpu_device)
if enable_channels_last:
    pipe.to(memory_format=torch.channels_last)

lora_model_repos = []
if args.max_lora_num == 1:
    lora_model_repos = ["TheLastBen/Papercut_SDXL"]
elif args.max_lora_num == 2:
    lora_model_repos = ["TheLastBen/William_Eggleston_Style_SDXL", "TheLastBen/Filmic"]
num_lora_model_repos = len(lora_model_repos)

# unet_state_dict_copy = deepcopy(pipe.unet.state_dict())
text_encoder_state_dict_copy = deepcopy(pipe.text_encoder.state_dict())
text_encoder_2_state_dict_copy = deepcopy(pipe.text_encoder_2.state_dict())

unet_state_dict_file = "default_unet_state_dict.pt"
if enable_channels_last:
    unet_state_dict_file = "default_unet_state_dict_channels_last.pt"
default_unet_state_dict = torch.load(unet_state_dict_file)
for key in default_unet_state_dict:
    default_unet_state_dict[key] = default_unet_state_dict[key].to("cuda")
pipe.unet.load_state_dict(deepcopy(default_unet_state_dict), strict=False)

print(f"Start UNet server on GPU {args.gpu_id}")

with torch.no_grad():
    while True:
        unet_state_dict_copy = deepcopy(default_unet_state_dict)
        if args.lora_mode == "full":
            patch_on_lora(pipe, lora_model_repos)

        batch_size_tensor = torch.empty((1), dtype=torch.int).to(gpu_device)
        dist.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor)

        num_inference_steps = torch.empty((1), dtype=torch.int).to(gpu_device)
        dist.broadcast(num_inference_steps, src=0)

        controlnet_cond = torch.empty((batch_size, 3, 1024, 1024), dtype=torch.float16).to(gpu_device)
        dist.broadcast(controlnet_cond, src=0)

        # get prompt_embeds, add_text_embeds, add_time_ids from rank 0
        prompt_embeds = torch.empty([batch_size, 77, 2048], dtype=torch.float16).to(gpu_device)
        add_text_embeds = torch.empty([batch_size, 1280], dtype=torch.float16).to(gpu_device)
        add_time_ids = torch.empty([batch_size, 6], dtype=torch.float16).to(gpu_device)
        dist.recv(prompt_embeds, src=0)
        dist.recv(add_text_embeds, src=0)
        dist.recv(add_time_ids, src=0)
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        if args.controlnet_parallel:
            for i in args.controlnet_parallel_rank:
                dist.send(prompt_embeds, dst=i)
            for i in args.controlnet_parallel_rank:
                dist.send(add_text_embeds, dst=i)
            for i in args.controlnet_parallel_rank:
                dist.send(add_time_ids, dst=i)

        latent_model_input = torch.empty([batch_size, 4, 128, 128]).type(torch.float16).to(gpu_device)
        t = torch.tensor(0.0).type(torch.float32).to(gpu_device)

        if not args.controlnet_parallel:
            down_block_res_samples = [
                torch.zeros([batch_size, 320, 128, 128]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 320, 128, 128]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 320, 128, 128]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 320, 64, 64]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 640, 64, 64]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 640, 64, 64]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 640, 32, 32]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 1280, 32, 32]).type(torch.float16).to(gpu_device),
                torch.zeros([batch_size, 1280, 32, 32]).type(torch.float16).to(gpu_device),
            ]
            mid_block_res_sample = torch.zeros([batch_size, 1280, 32, 32]).type(torch.float16).to(gpu_device)

        if args.verbose:
            print(f"Start unet inference")
        patched_lora = False
        for step in range(int(num_inference_steps)):
            if not patched_lora:
                if args.lora_mode == "sync" and args.load_lora_mode == "async":
                    # shm state: 0: not loading, 1-9: invoke loading LoRAs, 10: loaded
                    lora_info_shm_folder = "lora_info_shm"
                    status = torch.tensor(0).type(torch.int).to(gpu_device)
                    dist.recv(status, src=0)
                    if int(status) == num_lora_model_repos * 10:
                        # means all LoRAs have been loaded
                        state_dict = {}
                        for lora_loader_id in range(num_lora_model_repos): 
                            lora_info_file = "{}/{}.json".format(lora_info_shm_folder, "_".join(lora_model_repos[lora_loader_id].split("/")))
                            lora_network_alpha_file = "{}/{}_alphas.json".format(lora_info_shm_folder, "_".join(lora_model_repos[lora_loader_id].split("/")))

                            # load LoRA state dict
                            with open(lora_info_file, 'r') as fr:
                                lora_info_dict = json.load(fr)
                            for key in lora_info_dict.keys():
                                if state_dict.get(key, None) is None:
                                    state_dict[key] = torch.from_numpy(pipe.shm_dict["{}_np_{}".format(key, lora_loader_id)]).clone()
                                else:
                                    state_dict[key] += torch.from_numpy(pipe.shm_dict["{}_np_{}".format(key, lora_loader_id)]).clone()

                            # load network alphas
                            with open(lora_network_alpha_file, 'r') as fr:
                                network_alphas = json.load(fr)

                        pipe.patch_on_lora(state_dict, lora_model_repos, unet_state_dict_copy)
                        patched_lora = True
                        print(f"Patched {lora_model_repos} at step {step}")

            dist.broadcast(latent_model_input, src=0)
            dist.broadcast(t, src=0)

            if args.num_controlnets > 0:
                if args.controlnet_parallel:
                    down_block_res_samples = None
                    mid_block_res_sample = None
                else:
                    down_block_res_samples, mid_block_res_sample = pipe.controlnet(
                        latent_model_input,
                        t,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        added_cond_kwargs=added_cond_kwargs,
                    )

            noise_pred_uncond = pipe.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=None,
                cross_attention_kwargs=None,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
                controlnet_skt=None,
                serve_mode="standard",
                time_stats={},
                controlnet_num=0,
            )[0]

            dist.send(noise_pred_uncond.contiguous(), dst=0)

        pipe.unet.load_state_dict(deepcopy(default_unet_state_dict), strict=False)
        if args.max_lora_num == 2:
            pipe.text_encoder.load_state_dict(deepcopy(text_encoder_state_dict_copy))
            pipe.text_encoder_2.load_state_dict(deepcopy(text_encoder_2_state_dict_copy))
