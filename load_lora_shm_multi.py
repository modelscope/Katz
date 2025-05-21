import os
import json
import time
import argparse

import torch
import numpy as np
from multiprocessing import shared_memory

from diffusers.loaders import LoraLoaderMixin
from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineKatz


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard", "zmq", "nvlink"], 
                        help="which serve mode to use")
    parser.add_argument("--lora_mode", type=str, default="full", choices=["without", "full", "sync", "async"], 
                        help="which lora mode to use")
    parser.add_argument("--load_lora_mode", type=str, default="default", choices=["default", "async"], 
                        help="how to load lora")
    parser.add_argument("--lora_loader_id", type=int, default=0, choices=[0, 1], 
                        help="lora loader id")
    parser.add_argument("--lora_loader_num", type=int, default=2, choices=[1, 2], 
                        help="lora loader num")
    serve_args = parser.parse_args() 
    print("Args", serve_args)

    lora_info_shm_folder = "lora_info_shm"
    if serve_args.lora_loader_id == 0:
        lora_model_repo = "TheLastBen/Papercut_SDXL"
    elif serve_args.lora_loader_id == 1:
        lora_model_repo = "TheLastBen/Filmic"
    else:
        raise ValueError("Invalid lora_loader_id")

    if not os.path.exists(lora_info_shm_folder):
        raise ValueError("lora_info_shm_folder does not exist")
    lora_info_file = "{}/{}.json".format(lora_info_shm_folder, "_".join(lora_model_repo.split("/")))
    lora_info_dict = {}
    load_json_start = time.time()
    with open(lora_info_file, "r") as fr:
        lora_info_dict = json.load(fr)
    load_json_end = time.time()
    print("Load Json latency: {:.6f}".format(load_json_end - load_json_start))

    # create shm placeholder
    shm_dict = {}
    if serve_args.lora_loader_id == 0:
        shm_dict["start_loading_flag_shm"] = shared_memory.SharedMemory(name="start_loading_flag", create=True, size=serve_args.lora_loader_num)
    else:
        shm_dict["start_loading_flag_shm"] = shared_memory.SharedMemory(name="start_loading_flag")
    shm_dict["start_loading_flag_np"]  = np.ndarray( (serve_args.lora_loader_num, ), dtype=np.int8, buffer=shm_dict["start_loading_flag_shm"].buf)
    shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] = 0

    for key in lora_info_dict.keys():
        shm_dict["{}_shm".format(key)] = shared_memory.SharedMemory(name="{}_{}".format(key, serve_args.lora_loader_id), \
                                                                    create=True, size=lora_info_dict[key]["nbytes"])
        shm_dict["{}_np".format(key)] = np.ndarray(lora_info_dict[key]["shape"], dtype=np.float16, \
                                                    buffer=shm_dict["{}_shm".format(key)].buf)
        assert np.sum(shm_dict["{}_np".format(key)]) == 0, "shm not cleared"

    # Prepare to load the LoRA weights
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

    ### Warm LoRA file loading
    warm_repos = ["TheLastBen/Papercut_SDXL", "TheLastBen/Filmic", "TheLastBen/William_Eggleston_Style_SDXL"]
    for warm_repo in warm_repos:
        _, _ = LoraLoaderMixin.lora_state_dict(warm_repo, unet_config=pipe.unet.config, num_partitions=1, partition_id=0)
    print("Warm LoRA file loading done")

    print("Before entering busy waiting", shm_dict["start_loading_flag_np"])
    # Load LoRA weights
    while shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] != 100:
        # busy waiting to start loading
        print("Waiting to be invoked to load LoRA")
        while True:
            # have not been invoked to load LoRA
            cur_state = shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ]
            if cur_state == 0:
                continue
            elif cur_state == 10:
                continue
            else:
                print("Invoked with", cur_state)
                break
        if shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] == 100:
            break

        print("Invoked", shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ])

        if serve_args.lora_loader_id == 0:
            # The first loader is for Papercut_SDXL or William_Eggleston_Style_SDXL
            if shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] == 1:
                lora_model_repo = "TheLastBen/Papercut_SDXL"
            elif shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] == 2:
                lora_model_repo = "TheLastBen/William_Eggleston_Style_SDXL"
        else:
            # The second loader is for Filmic
            if shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] == 1:
                lora_model_repo = "TheLastBen/Filmic"

        print("Starting LoRA loading")
        state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_model_repo, unet_config=pipe.unet.config, num_partitions=1, partition_id=0)
        for key in state_dict.keys():
            cur_weight_np = state_dict[key].numpy()
            shm_dict["{}_np".format(key)][:] = cur_weight_np[:]

        print("========= Complete LoRA loading =========")
        shm_dict["start_loading_flag_np"][ serve_args.lora_loader_id ] = 10  # 10 means loading is done

    time.sleep(10)
    for key in shm_dict:
        if "shm" in key:
            shm_dict[key].close()
            shm_dict[key].unlink()
    print("shm cleared")
