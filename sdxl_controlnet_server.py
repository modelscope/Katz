import os
import time
import argparse

import numpy as np

import torch
import torch.distributed as dist

from diffusers import ControlNetModel

import warnings
warnings.filterwarnings("ignore")

total_gpu_num = torch.cuda.device_count()

print("Total GPUs in the world:", total_gpu_num)
print("torch.distributed.is_available():", torch.distributed.is_available())
print("torch.distributed.is_nccl_available():", torch.distributed.is_nccl_available())

if total_gpu_num < 2:
    assert False, "Must have more than one GPU"

parser = argparse.ArgumentParser()
# distributed
parser.add_argument("--world-size", type=int, default=2, help="world size")
parser.add_argument("--gpu-id", type=int, default=1, help="which gpu to use")
parser.add_argument("--rank", type=int, default=1)
parser.add_argument("--dst-rank", type=int, default=0)
# controlnet
parser.add_argument("--controlnet-parallel-rank", type=int, nargs='+', default=[]) # Example: [2, 4, 6] or [3, 5, 7]
parser.add_argument("-v", "--verbose", action="store_true")
args = parser.parse_args() 
print(f"Args: {args}")

dst_rank = args.dst_rank
assert dst_rank < 2, "dst_rank must be less than 2"

# set the gpu to be used
torch.cuda.set_device(args.gpu_id)
gpu_device = torch.cuda.current_device()

# initialize the process group
dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:8000', rank=args.rank, world_size=args.world_size)
print(f"World size: {dist.get_world_size()}, rank: {dist.get_rank()}")

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    variant="fp16",
    use_safetensors=True,
    torch_dtype=torch.float16,
)
controlnet = controlnet.to(gpu_device)
enable_channels_last = os.getenv("ENABLE_CHANNELS_LAST", "0") == "1"
if enable_channels_last:
    controlnet.to(memory_format=torch.channels_last)

print(f"Start ControlNet server on GPU {args.gpu_id}")

with torch.no_grad():
    while True:
        batch_size_tensor = torch.empty((1), dtype=torch.int).to(gpu_device)
        dist.broadcast(batch_size_tensor, src=0)
        batch_size = int(batch_size_tensor)

        num_inference_steps = torch.empty((1), dtype=torch.int).to(gpu_device)
        dist.broadcast(num_inference_steps, src=0)

        controlnet_cond = torch.empty((batch_size, 3, 1024, 1024), dtype=torch.float16).to(gpu_device)
        dist.broadcast(controlnet_cond, src=0)

        prompt_embeds = torch.empty([batch_size, 77, 2048], dtype=torch.float16).to(gpu_device)
        add_text_embeds = torch.empty([batch_size, 1280], dtype=torch.float16).to(gpu_device)
        add_time_ids = torch.empty([batch_size, 6], dtype=torch.float16).to(gpu_device)
        dist.recv(prompt_embeds, src=dst_rank)
        dist.recv(add_text_embeds, src=dst_rank)
        dist.recv(add_time_ids, src=dst_rank)
        added_cond_kwargs = {
            "text_embeds": add_text_embeds,
            "time_ids": add_time_ids,
        }

        control_model_input = torch.empty([batch_size, 4, 128, 128]).type(torch.float16).to(gpu_device)
        t = torch.tensor(0.0).type(torch.float32).to(gpu_device)

        controlnet_compute_time = []
        for step in range(int(num_inference_steps)):
            dist.broadcast(control_model_input, src=0)
            dist.broadcast(t, src=0)

            torch.cuda.synchronize()
            controlnet_start = time.time()

            down_block_res_samples, mid_block_res_sample = controlnet(
                control_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                controlnet_cond=controlnet_cond,
                added_cond_kwargs=added_cond_kwargs,
            )

            torch.cuda.synchronize()
            controlnet_end = time.time()
            controlnet_compute_time.append(controlnet_end - controlnet_start)

            # send down_block_res_samples
            for i in range(len(down_block_res_samples)):
                # must use contiguous: https://github.com/pytorch/pytorch/issues/77554
                down_block_res_samples[i] = down_block_res_samples[i].contiguous()
                dist.send(down_block_res_samples[i], dst=dst_rank)
            # send mid_block_res_sample
            mid_block_res_sample = mid_block_res_sample.contiguous()
            dist.send(mid_block_res_sample, dst=dst_rank)

        print(f"Avg: {np.mean(controlnet_compute_time):.4f}, Sum: {np.sum(controlnet_compute_time):.4f}")
