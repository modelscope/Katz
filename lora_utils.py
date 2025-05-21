import time
import json

import torch

from diffusers.loaders import LoraLoaderMixin


def patch_on_lora(pipe, lora_model_repos):
    if not isinstance(lora_model_repos, list):
        lora_model_repos = [lora_model_repos]

    lora_weights = {}
    for lora_model_repo in lora_model_repos:
        load_start = time.time()
        lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_model_repo, unet_config=pipe.unet.config, num_partitions=1, partition_id=0)
        load_end = time.time()
        lora_weights[lora_model_repo] = lora_state_dict
        print("Load LoRA file latency: {:.2f}".format(load_end - load_start))

    unet_state_dict = pipe.unet.state_dict()
    scale = torch.tensor(20000.0/64.0, dtype=torch.float16).cuda()
    patch_start = time.time()
    print("Fuse LoRA to unet")
    for lora_model_repo in lora_model_repos:
        unet_lora_match_keys = {}
        with open("./lora_info_shm/{}_key_match_unet.json".format( lora_model_repo.replace("/", "_") ),  "r") as fr:
            unet_lora_match_keys = json.load(fr)
        for key in unet_lora_match_keys:
            cur_down = lora_weights[lora_model_repo][ unet_lora_match_keys[key]["lora_down_key"] ].to("cuda", non_blocking=True)
            cur_up   = lora_weights[lora_model_repo][ unet_lora_match_keys[key]["lora_up_key"] ].to("cuda", non_blocking=True)
            unet_state_dict[key] +=  cur_up @ cur_down * scale
    pipe.unet.load_state_dict(unet_state_dict, strict=False)

    print("Fuse LoRA to text_encoder and text_encoder_2")
    for lora_model_repo in lora_model_repos:
        if lora_model_repo == "TheLastBen/Filmic":
            text_encoder_state_dict = pipe.text_encoder.state_dict()
            with open("./lora_info_shm/{}_key_match_text_encoder.json".format( lora_model_repo.replace("/", "_") ), "r") as fr:
                unet_lora_match_keys_text_encoder = json.load(fr)
            for key in unet_lora_match_keys_text_encoder:
                cur_down = lora_weights[lora_model_repo][ unet_lora_match_keys_text_encoder[key]["lora_down_key"] ].to("cuda", non_blocking=True)
                cur_up   = lora_weights[lora_model_repo][ unet_lora_match_keys_text_encoder[key]["lora_up_key"] ].to("cuda", non_blocking=True)
                text_encoder_state_dict[key] +=  cur_up @ cur_down
            pipe.text_encoder.load_state_dict(text_encoder_state_dict, strict=False)

            text_encoder_2_state_dict = pipe.text_encoder_2.state_dict()
            with open("./lora_info_shm/{}_key_match_text_encoder_2.json".format( lora_model_repo.replace("/", "_") ),  "r") as fr:
                unet_lora_match_keys_text_encoder_2 = json.load(fr)
            for key in unet_lora_match_keys_text_encoder_2:
                cur_down = lora_weights[lora_model_repo][ unet_lora_match_keys_text_encoder_2[key]["lora_down_key"] ].to("cuda", non_blocking=True)
                cur_up   = lora_weights[lora_model_repo][ unet_lora_match_keys_text_encoder_2[key]["lora_up_key"] ].to("cuda", non_blocking=True)
                text_encoder_2_state_dict[key] +=  cur_up @ cur_down
            pipe.text_encoder_2.load_state_dict(text_encoder_2_state_dict, strict=False)

        patch_end = time.time()
        print("Patch latency: {:.2f}".format(patch_end - patch_start))
