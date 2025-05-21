import os
import json
import argparse
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt


def unique_adapter_loading_analysis(trace_data, scene_id):
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

    UNIQUE_ADDON_FIG_PATH = os.path.join(BASE_FIG_DIR, f"unique_loras_{scene_id}.pdf")
    print(f"UNIQUE_ADDON_FIG_PATH: {UNIQUE_ADDON_FIG_PATH}")

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


    # Analyze the unique addon numbers
    print("=====Unique LoRAs=====")
    df = pd.DataFrame()
    for ip in ip_req_dict:
        unique_controlnets = set()
        for req in ip_req_dict[ip]:
            controlnets = req["controlnets"]
            if controlnets is None or len(controlnets) == 0:
                continue
            unique_controlnets.update([controlnet for controlnet in controlnets])

        unique_loras = set()
        for req in ip_req_dict[ip]:
            loras = req["loras"]
            if loras is None or len(loras) == 0:
                continue
            unique_loras.update([lora for lora in loras])

        df = pd.concat([df, pd.DataFrame({
            "ip": [ip], 
            "num_requests": [len(ip_req_dict[ip])], 
            "unique_controlnets": [len(unique_controlnets)], 
            "unique_loras": [len(unique_loras)]
        })], ignore_index=True)

    # Plot the unique addon numbers
    plt.figure(figsize=(7, 3), dpi=120)
    plt.scatter(df["num_requests"], df["unique_loras"], marker='^', label="LoRA")
    plt.xlabel("# of requests")
    plt.ylabel("# of unique\nLoRAs", loc='top')
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig(UNIQUE_ADDON_FIG_PATH, bbox_inches='tight')
    print(f"Figure saved to {UNIQUE_ADDON_FIG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="diffusion_model_request_trace.json", help="Path to the trace file")
    parser.add_argument("--service", type=str, default="A", help="Service name", choices=["A", "B"])

    args = parser.parse_args()
    trace_file = args.trace_file
    service = args.service

    unique_adapter_loading_analysis(trace_file, service)
