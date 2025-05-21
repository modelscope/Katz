import os
import argparse
import warnings

import torch

from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineKatz


warnings.filterwarnings("ignore")
# use generator to make the sampling deterministic
seed = 0
sd_generator = torch.manual_seed(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--num_images", type=int, default=1)
parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard", "zmq", "nvlink", "without"], help="which serve mode to use") # deprecated
# distributed
parser.add_argument("--world-size", type=int, default=2, help="world-size")
parser.add_argument("--gpu-id", type=int, default=0, help="which gpu to use")
parser.add_argument("--rank", type=int, default=0)
# ControlNet
parser.add_argument("--controlnet-parallel", action="store_true")
parser.add_argument("--controlnet-parallel-rank", type=int, nargs='+', default=[]) # Example: [2, 4, 6]
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
print("Args", serve_args)

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
if not enable_channels_last:
    torch.save(pipe.unet.state_dict(), "./default_unet_state_dict.pt")
else:
    pipe.unet.to(memory_format=torch.channels_last)
    torch.save(pipe.unet.state_dict(), "./default_unet_state_dict_channels_last.pt")
