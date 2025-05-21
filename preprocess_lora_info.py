import time
import argparse

import torch

from diffusers.loaders import LoraLoaderMixin
from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineKatz

parser = argparse.ArgumentParser()
parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard", "zmq", "nvlink"], 
                    help="which serve mode to use")
parser.add_argument("--lora_mode", type=str, default="full", choices=["without", "full", "sync", "async"], 
                    help="which lora mode to use")
parser.add_argument("--load_lora_mode", type=str, default="default", choices=["default", "async"], 
                    help="how to load lora")
serve_args = parser.parse_args() 
print("Args", serve_args)

if __name__ == "__main__":
    # Load the state dict
    controlnet = ControlNetModel.from_pretrained(
        "diffusers/controlnet-canny-sdxl-1.0",
        torch_dtype=torch.float16
    )
    vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
    pipe = StableDiffusionXLControlNetPipelineKatz.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet=controlnet,
        vae=vae,
        torch_dtype=torch.float16,
        serve_args=serve_args,
    )

    lora_model_repo = "TheLastBen/Papercut_SDXL"
    lora_model_repo = "TheLastBen/Filmic"
    # lora_model_repo = "TheLastBen/William_Eggleston_Style_SDXL"

    load_start = time.time()
    state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_model_repo, unet_config=pipe.unet.config, num_partitions=1, partition_id=0)
    load_end = time.time()
    print("Load latency: {:.2f}".format(load_end - load_start))
    
    lora_info_dict = {}
    for key in state_dict.keys():
        cur_weight_np = state_dict[key].numpy()
        lora_info_dict[key] = {}
        lora_info_dict[key]["shape"] = cur_weight_np.shape
        lora_info_dict[key]["nbytes"] = cur_weight_np.nbytes
    assert len(lora_info_dict.keys()) == len(state_dict.keys())

    lora_info_folder = "./lora_info_shm"

    import json
    with open("{}/{}.json".format(lora_info_folder, "_".join(lora_model_repo.split("/"))), "w") as fw:
        json.dump(lora_info_dict, fw)

    with open("{}/{}_alphas.json".format(lora_info_folder, "_".join(lora_model_repo.split("/"))), "w") as fw:
        json.dump(network_alphas, fw)

    print( cur_weight_np.dtype )
    print( type(network_alphas), len(network_alphas.keys()) )
    for key in network_alphas.keys():
        assert network_alphas[key] == 20000.0 or network_alphas[key] == 64, "{}, {}".format(key, network_alphas[key])
