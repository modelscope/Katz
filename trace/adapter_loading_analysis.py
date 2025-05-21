import os
import json
import argparse
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
from collections import defaultdict, OrderedDict


def adapter_loading_analysis(trace_data, scene_id, cache_policy):
    with open(trace_data, "r") as f:
        req_list = json.load(f)
        # filter the requests based on the scene id
        req_list = [req for req in req_list if req["scene_id"] in scene_id]

    matplotlib.rcdefaults()
    matplotlib.rcParams['pdf.fonttype'] = 42

    matplotlib.rcParams.update({"font.size": 34}) # for 24 for (8,6), 16 for (4,3)
    matplotlib.rcParams['lines.linewidth'] = 2 # 2.5

    BASE_FIG_DIR = "figures"
    if not os.path.exists(BASE_FIG_DIR):
        os.makedirs(BASE_FIG_DIR)

    ADAPTER_LOADING_FIG_PATH = os.path.join(BASE_FIG_DIR, f"adapter_loading_{cache_policy}_{scene_id}.pdf")
    print(f"ADAPTER_LOADING_FIG_PATH: {ADAPTER_LOADING_FIG_PATH}")

    # Get the unique IPs
    ip_list = []
    for req in req_list:
        ip_list.append(req["ip"])
    unique_ips = list(set(ip_list))
    print(f"Unique IPs: {len(unique_ips)}")

    ip_req_dict = {}
    for req in req_list:
        if req["ip"] not in ip_req_dict:
            ip_req_dict[req["ip"]] = []
        ip_req_dict[req["ip"]].append(req)

    # sort req of each ip by req's timestamp
    for ip in ip_req_dict:
        ip_req_dict[ip].sort(key=lambda x: x["timestamp"])

    print("=====Adapter Loading Analysis=====")
    CANDIDATE_ADDON_CACHE_SIZES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Cache policy: {cache_policy}")
    if cache_policy == "LRU":
        df = pd.DataFrame()
        for cache_size in CANDIDATE_ADDON_CACHE_SIZES:
            overall_switch_controlnet_times = 0
            overall_switch_lora_times = 0
            total_req_cnt = 0

            for ip in ip_req_dict:
                controlnet_cache = OrderedDict()
                switch_controlnet_times = 0
                for req_id, req in enumerate(ip_req_dict[ip]):
                    controlnets = req["controlnets"]
                    if controlnets is None or len(controlnets) == 0:
                        continue
                    for controlnet in controlnets:
                        model_id = controlnet
                        if model_id in controlnet_cache:
                            controlnet_cache.move_to_end(model_id)
                            continue
                        if len(controlnet_cache) >= cache_size and len(controlnet_cache) > 0:
                            controlnet_cache.popitem(last=False)
                        controlnet_cache[model_id] = 1
                        switch_controlnet_times += 1
                overall_switch_controlnet_times += switch_controlnet_times

                lora_cache = OrderedDict()
                switch_lora_times = 0
                for req_id, req in enumerate(ip_req_dict[ip]):
                    loras = req["loras"]
                    if loras is None or len(loras) == 0:
                        continue
                    for lora in loras:
                        if lora in lora_cache:
                            lora_cache.move_to_end(lora)
                            continue
                        if len(lora_cache) >= cache_size and len(lora_cache) > 0:
                            lora_cache.popitem(last=False)
                        lora_cache[lora] = 1
                        switch_lora_times += 1
                overall_switch_lora_times += switch_lora_times

                total_req_cnt += len(ip_req_dict[ip])

            print(f"Cache size: {cache_size}, overall switch controlnet times: {overall_switch_controlnet_times}, per request: {overall_switch_controlnet_times/total_req_cnt:.4f}")
            print(f"Cache size: {cache_size}, overall switch lora times: {overall_switch_lora_times}, per request: {overall_switch_lora_times/total_req_cnt:.4f}")
            df = pd.concat([df, pd.DataFrame({
                "cache_size": [cache_size], 
                "overall_switch_controlnet_times": [overall_switch_controlnet_times], 
                "average_switch_controlnet_times": [1.0*overall_switch_controlnet_times/total_req_cnt], 
                "overall_switch_lora_times": [overall_switch_lora_times], 
                "average_switch_lora_times": [1.0*overall_switch_lora_times/total_req_cnt]
            })], ignore_index=True)

    elif cache_policy == "LFU":
        df = pd.DataFrame()
        for cache_size in CANDIDATE_ADDON_CACHE_SIZES:

            overall_switch_controlnet_times = 0
            overall_switch_lora_times = 0
            total_req_cnt = 0

            for ip in ip_req_dict:
                controlnet_cache = {}  # Stores model_id -> frequency
                controlnet_freq_map = defaultdict(OrderedDict)  # Tracks frequency -> {model_id: None}
                min_freq = 0
                switch_controlnet_times = 0

                for req_id, req in enumerate(ip_req_dict[ip]):
                    controlnets = req["controlnets"]
                    if controlnets is None or len(controlnets) == 0:
                        continue
                    for controlnet in controlnets:
                        model_id = controlnet

                        # If the model_id is already in the cache, update its frequency
                        if model_id in controlnet_cache:
                            freq = controlnet_cache[model_id]
                            controlnet_freq_map[freq].pop(model_id)
                            if not controlnet_freq_map[freq]:
                                del controlnet_freq_map[freq]
                                if min_freq == freq:
                                    min_freq += 1
                            controlnet_cache[model_id] += 1
                            new_freq = controlnet_cache[model_id]
                            controlnet_freq_map[new_freq][model_id] = None
                        else:
                            # If the cache is full, remove the least frequently used item
                            if len(controlnet_cache) >= cache_size:
                                lfu_model_id, _ = controlnet_freq_map[min_freq].popitem(last=False)
                                if not controlnet_freq_map[min_freq]:
                                    del controlnet_freq_map[min_freq]
                                del controlnet_cache[lfu_model_id]

                            # Add the new model_id to the cache with frequency 1
                            controlnet_cache[model_id] = 1
                            controlnet_freq_map[1][model_id] = None
                            min_freq = 1
                            switch_controlnet_times += 1

                overall_switch_controlnet_times += switch_controlnet_times

                # LoRA LFU Cache Implementation
                lora_cache = {}  # Stores model_id -> frequency
                lora_freq_map = defaultdict(OrderedDict)  # Tracks frequency -> {model_id: None}
                min_freq = 0
                switch_lora_times = 0

                for req_id, req in enumerate(ip_req_dict[ip]):
                    loras = req["loras"]
                    if loras is None or len(loras) == 0:
                        continue
                    for lora in loras:
                        model_id = lora

                        # If the model_id is already in the cache, update its frequency
                        if model_id in lora_cache:
                            freq = lora_cache[model_id]
                            lora_freq_map[freq].pop(model_id)
                            if not lora_freq_map[freq]:
                                del lora_freq_map[freq]
                                if min_freq == freq:
                                    min_freq += 1
                            lora_cache[model_id] += 1
                            new_freq = lora_cache[model_id]
                            lora_freq_map[new_freq][model_id] = None
                        else:
                            # If the cache is full, remove the least frequently used item
                            if len(lora_cache) >= cache_size:
                                lfu_model_id, _ = lora_freq_map[min_freq].popitem(last=False)
                                if not lora_freq_map[min_freq]:
                                    del lora_freq_map[min_freq]
                                del lora_cache[lfu_model_id]

                            # Add the new model_id to the cache with frequency 1
                            lora_cache[model_id] = 1
                            lora_freq_map[1][model_id] = None
                            min_freq = 1
                            switch_lora_times += 1

                overall_switch_lora_times += switch_lora_times

                total_req_cnt += len(ip_req_dict[ip])

            print(f"Cache size: {cache_size}, overall switch controlnet times: {overall_switch_controlnet_times}, per request: {overall_switch_controlnet_times/total_req_cnt:.4f}")
            print(f"Cache size: {cache_size}, overall switch lora times: {overall_switch_lora_times}, per request: {overall_switch_lora_times/total_req_cnt:.4f}")
            df = pd.concat([df, pd.DataFrame({
                "cache_size": [cache_size], 
                "overall_switch_controlnet_times": [overall_switch_controlnet_times], 
                "average_switch_controlnet_times": [1.0*overall_switch_controlnet_times/total_req_cnt], 
                "overall_switch_lora_times": [overall_switch_lora_times], 
                "average_switch_lora_times": [1.0*overall_switch_lora_times/total_req_cnt]
            })], ignore_index=True)


    # Plot the average loading times for ControlNet and LoRA
    plt.figure(figsize=(7, 3), dpi=120)
    plt.plot(df["cache_size"], df["average_switch_controlnet_times"], marker='o', label="ControlNet", markersize=10, linewidth=4, linestyle='-')
    plt.plot(df["cache_size"], df["average_switch_lora_times"], marker='^', label="LoRA", markersize=10, linewidth=4, linestyle='--')
    plt.xticks(CANDIDATE_ADDON_CACHE_SIZES)
    if scene_id == "A":
        plt.ylim(None, 2.0)
    plt.xlabel("Cache size (# adapters)")
    plt.ylabel("Avg. # \n loading times", loc='top')
    if scene_id == "A":
        plt.legend(loc='lower right', frameon=False, ncol=1, fontsize=32, borderaxespad=0, labelspacing=0.1, handletextpad=0.1, handlelength=2, markerscale=1)
    elif scene_id == "B":
        plt.legend(loc='upper right', frameon=False, ncol=1, fontsize=32, borderaxespad=0, labelspacing=0.1, handletextpad=0.1, handlelength=2, markerscale=1)
    else:
        raise ValueError("Invalid scene_id")
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig(ADAPTER_LOADING_FIG_PATH, bbox_inches='tight')
    print(f"Figure saved to {ADAPTER_LOADING_FIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="diffusion_model_request_trace.json", help="Path to the trace file")
    parser.add_argument("--service", type=str, default="A", help="Service name", choices=["A", "B"])
    parser.add_argument("--cache-policy", type=str, default="LRU", help="Cache policy", choices=["LRU", "LFU"])

    args = parser.parse_args()
    trace_file = args.trace_file
    service = args.service
    cache_policy = args.cache_policy

    adapter_loading_analysis(trace_file, service, cache_policy)
