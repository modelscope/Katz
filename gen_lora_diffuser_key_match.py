import time 
import json
import argparse

import warnings
warnings.filterwarnings("ignore")

import torch

from diffusers import ControlNetModel, AutoencoderKL, StableDiffusionXLControlNetPipelineSY

# use generator to make the sampling deterministic
seed = 0
sd_generator = torch.manual_seed(seed)


parser = argparse.ArgumentParser()
parser.add_argument("--serve_mode", type=str, default="standard", choices=["standard"], 
                    help="which serve mode to use")
parser.add_argument("--lora_mode", type=str, default="full", choices=["full"], 
                    help="which lora mode to use")
parser.add_argument("--load_lora_mode", type=str, default="default", choices=["default"], 
                    help="how to load lora")
serve_args = parser.parse_args() 
print("Args", serve_args)

controlnet_conditioning_scale = 0.5  # recommended for good generalization

controlnet = ControlNetModel.from_pretrained(
    "diffusers/controlnet-canny-sdxl-1.0",
    torch_dtype=torch.float16
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLControlNetPipelineSY.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=controlnet,
    vae=vae,
    torch_dtype=torch.float16,
    serve_args=serve_args,
)

pipe_togpu_start = time.time()
pipe = pipe.to("cuda")
pipe_togpu_end = time.time()
print("Pipe to GPU latency: {:.2f}".format(pipe_togpu_end - pipe_togpu_start))

# lora_model_repo = "TheLastBen/Papercut_SDXL"
# lora_model_repo = "TheLastBen/William_Eggleston_Style_SDXL"
lora_model_repo = "TheLastBen/Filmic"

from diffusers.loaders import LoraLoaderMixin
load_start = time.time()
lora_state_dict, network_alphas = LoraLoaderMixin.lora_state_dict(lora_model_repo, unet_config=pipe.unet.config, num_partitions=1, partition_id=0)
load_end = time.time()
print("Load LoRA file latency: {:.2f}".format(load_end - load_start))

unet_state_dict_keys = pipe.unet.state_dict().keys()
text_encoder_state_dict_keys = pipe.text_encoder.state_dict().keys()
text_encoder_2_state_dict_keys = pipe.text_encoder_2.state_dict().keys()
print("unet state key length", len( unet_state_dict_keys ))
print("text_encoder state key length", len( text_encoder_state_dict_keys ))
print("text_encoder_2 state key length", len( text_encoder_2_state_dict_keys ))


lora_state_keys = list(lora_state_dict.keys())
print("lora_state_keys length", len(lora_state_keys))

lora_unet_keys = [ key for key in lora_state_keys if "unet." in key ]
lora_text_encoder_keys = [ key for key in lora_state_keys if "text_encoder." in key ]
lora_text_encoder_2_keys = [ key for key in lora_state_keys if "text_encoder_2." in key ]
print("lora_unet_keys length", len(lora_unet_keys))
print("lora_text_encoder_keys length", len(lora_text_encoder_keys))
print("lora_text_encoder_2_keys length", len(lora_text_encoder_2_keys))
print("=========")

def match_text_encoder(diffuser_lora_key_match, text_encoder_state_dict_keys, lora_text_encoder_keys, prefix):
    counter = 0
    for key in text_encoder_state_dict_keys:
        if ".weight" in key and "layer_norm" not in key and "embedding" not in key:
            if "self_attn." in key:
                items = key.split(".")
                qkvo = items[-2].split("_")[0]
                lora_down_key = "{}.{}.to_{}_lora.down.weight".format( prefix, ".".join(items[:-2]), qkvo )
                lora_up_key   = "{}.{}.to_{}_lora.up.weight".format( prefix, ".".join(items[:-2]), qkvo )

                if lora_down_key in lora_text_encoder_keys or lora_up_key in lora_text_encoder_keys:
                    diffuser_lora_key_match[key] = {}
                    diffuser_lora_key_match[key]["lora_down_key"] = lora_down_key
                    diffuser_lora_key_match[key]["lora_up_key"] = lora_up_key

                    if lora_down_key in lora_text_encoder_keys:
                        lora_text_encoder_keys.remove(lora_down_key)
                        counter += 1
                    else:
                        print(key, lora_down_key)
                    
                    if lora_up_key in lora_text_encoder_keys:
                        lora_text_encoder_keys.remove(lora_up_key)
                        counter += 1
                    else:
                        print(key, lora_up_key)

            else:
                items = key.split(".")
                lora_down_key = "{}.{}.lora_linear_layer.down.weight".format( prefix, ".".join(items[:-1]) )
                lora_up_key = "{}.{}.lora_linear_layer.up.weight".format( prefix, ".".join(items[:-1]) )

                if lora_down_key in lora_text_encoder_keys or lora_up_key in lora_text_encoder_keys:
                    diffuser_lora_key_match[key] = {}
                    diffuser_lora_key_match[key]["lora_down_key"] = lora_down_key
                    diffuser_lora_key_match[key]["lora_up_key"] = lora_up_key

                    if lora_down_key in lora_text_encoder_keys:
                        lora_text_encoder_keys.remove(lora_down_key)
                        counter += 1
                    else:
                        print(key, lora_down_key)
                    
                    if lora_up_key in lora_text_encoder_keys:
                        lora_text_encoder_keys.remove(lora_up_key)
                        counter += 1
                    else:
                        print(key, lora_up_key)

    print("counter", counter)
    print("lora_text_encoder_keys length", len(lora_text_encoder_keys))
    # assert counter == len(lora_text_encoder_keys)
    assert len(lora_text_encoder_keys) == 0


def match_unet(diffuser_lora_key_match, unet_state_dict_keys, lora_unet_keys):
    for key in unet_state_dict_keys:
        if ".weight" in key and "conv" not in key:
            if ".attn" in key:
                items = key.split(".")
                if "to_out.0" not in key:
                    if "mid_block" in key:
                        lora_down_key = "unet.{}.processor.{}.down.weight".format( ".".join(items[:6]), items[-2]+"_lora" )
                        lora_up_key   = "unet.{}.processor.{}.up.weight".format( ".".join(items[:6]), items[-2]+"_lora" )
                    else:
                        lora_down_key = "unet.{}.processor.{}.down.weight".format( ".".join(items[:7]), items[-2]+"_lora" )
                        lora_up_key   = "unet.{}.processor.{}.up.weight".format( ".".join(items[:7]), items[-2]+"_lora" )
                else:
                    if "mid_block" in key:
                        lora_down_key = "unet.{}.processor.{}.down.weight".format( ".".join(items[:6]), items[-3]+"_lora" )
                        lora_up_key   = "unet.{}.processor.{}.up.weight".format( ".".join(items[:6]), items[-3]+"_lora" )
                    else:
                        lora_down_key = "unet.{}.processor.{}.down.weight".format( ".".join(items[:7]), items[-3]+"_lora" )
                        lora_up_key   = "unet.{}.processor.{}.up.weight".format( ".".join(items[:7]), items[-3]+"_lora" )  
            else:
                items = key.split(".")
                lora_down_key = "unet.{}.lora.down.weight".format( ".".join(items[:-1]) )
                lora_up_key = "unet.{}.lora.up.weight".format( ".".join(items[:-1]) )
            

            if lora_down_key in lora_unet_keys and lora_up_key in lora_state_keys:

                diffuser_lora_key_match[key] = {}
                diffuser_lora_key_match[key]["lora_down_key"] = lora_down_key
                diffuser_lora_key_match[key]["lora_up_key"] = lora_up_key

diffuser_lora_key_match_text_encoder = {}
match_text_encoder(diffuser_lora_key_match_text_encoder, text_encoder_state_dict_keys, lora_text_encoder_keys, "text_encoder")
with open("./lora_info_shm/{}_key_match_text_encoder.json".format(lora_model_repo.replace("/", "_")),  "w") as fw:
    json.dump(diffuser_lora_key_match_text_encoder, fw)

diffuser_lora_key_match_text_encoder_2 = {}
match_text_encoder(diffuser_lora_key_match_text_encoder_2, text_encoder_2_state_dict_keys, lora_text_encoder_2_keys, "text_encoder_2")
with open("./lora_info_shm/{}_key_match_text_encoder_2.json".format(lora_model_repo.replace("/", "_")),  "w") as fw:
    json.dump(diffuser_lora_key_match_text_encoder_2, fw)

diffuser_lora_key_match_unet = {}
match_unet(diffuser_lora_key_match_unet, unet_state_dict_keys, lora_unet_keys)
with open("./lora_info_shm/{}_key_match_unet.json".format(lora_model_repo.replace("/", "_")),  "w") as fw:
    json.dump(diffuser_lora_key_match_unet, fw)

# with open("./lora_info_shm/{}_key_match.json".format( "TheLastBen/Papercut_SDXL".replace("/", "_") ),  "r") as fr:
#     unet_lora_match_keys = json.load(fr)
# for key in unet_lora_match_keys:
#     cur_down = lora_state_dict[ unet_lora_match_keys[key]["lora_down_key"] ].to("cuda", non_blocking=True)
#     cur_up   = lora_state_dict[ unet_lora_match_keys[key]["lora_up_key"] ].to("cuda", non_blocking=True)

#     unet_state_dict[key] +=  cur_up @ cur_down * scale
# pipe.unet.load_state_dict(unet_state_dict, strict=False)
