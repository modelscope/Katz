import os
import json
import argparse
import matplotlib
from matplotlib import pyplot as plt


def controlnet_analysis(trace_file):
    with open(trace_file, "r") as f:
        req_list = json.load(f)

    matplotlib.rcdefaults()
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams.update({"font.size": 34})
    matplotlib.rcParams['lines.linewidth'] = 2

    BASE_FIG_DIR = "figures"
    if not os.path.exists(BASE_FIG_DIR):
        os.makedirs(BASE_FIG_DIR)

    # Path to save the figure
    CONTROLNET_INVOCATION_FIG_PATH = os.path.join(BASE_FIG_DIR, f"controlnet_invocation.pdf")
    print(f"CONTROLNET_INVOCATION_FIG_PATH: {CONTROLNET_INVOCATION_FIG_PATH}")


    # Service A
    print("=====Service A=====")

    controlnet_A_list = []
    for req in req_list:
        if req["scene_id"] == "A":
            if req["controlnets"] is None:
                continue
            for controlnet in req["controlnets"]:
                controlnet_A_list.append(controlnet)
    print(f"Unique controlnet ids: {len(set(controlnet_A_list))}")

    controlnet_A_dict = {}
    for controlnet_id in controlnet_A_list:
        if controlnet_id not in controlnet_A_dict:
            controlnet_A_dict[controlnet_id] = 0
        controlnet_A_dict[controlnet_id] += 1
    controlnet_A_dict = {k: v for k, v in sorted(controlnet_A_dict.items(), key=lambda item: item[1], reverse=True)}

    for controlnet_id, count in controlnet_A_dict.items():
        if count/len(controlnet_A_list) > 0.01:
            print(f"ControlNet ID: {controlnet_id}, Count: {count}, Percentage: {count/len(controlnet_A_list):.4f}")

    total_count = sum(controlnet_A_dict.values())

    cumulative_count = 0
    cumulative_percentages_A = [0]
    for count in controlnet_A_dict.values():
        cumulative_count += count
        cumulative_percentage = cumulative_count / total_count * 100
        cumulative_percentages_A.append(cumulative_percentage)

    controlnet_ids_A = list(controlnet_A_dict.keys())


    # Service B
    print("=====Service B=====")

    controlnet_B_list = []
    for req in req_list:
        if req["scene_id"] == "B":
            if req["controlnets"] is None:
                continue
            for controlnet in req["controlnets"]:
                controlnet_B_list.append(controlnet)
    print(f"Unique controlnet ids: {len(set(controlnet_B_list))}")

    controlnet_B_dict = {}
    for controlnet_id in controlnet_B_list:
        if controlnet_id not in controlnet_B_dict:
            controlnet_B_dict[controlnet_id] = 0
        controlnet_B_dict[controlnet_id] += 1

    controlnet_B_dict = {k: v for k, v in sorted(controlnet_B_dict.items(), key=lambda item: item[1], reverse=True)}

    for controlnet_id, count in controlnet_B_dict.items():
        if count/len(controlnet_B_list) > 0.01:
            print(f"ControlNet ID: {controlnet_id}, Count: {count}, Percentage: {count/len(controlnet_B_list):.4f}")

    total_count = sum(controlnet_B_dict.values())

    cumulative_count = 0
    cumulative_percentages_B = [0]
    for count in controlnet_B_dict.values():
        cumulative_count += count
        cumulative_percentage = cumulative_count / total_count * 100
        cumulative_percentages_B.append(cumulative_percentage)

    controlnet_ids_B = list(controlnet_B_dict.keys())


    # Plotting
    plt.figure(figsize=(7, 3))
    plt.plot(range(len(controlnet_ids_A)+1), cumulative_percentages_A, marker='o', markersize=10, linestyle='-', color='tab:green', label="Service A", linewidth=4)
    plt.plot(range(len(controlnet_ids_B)+1), cumulative_percentages_B, marker='s', markersize=10, linestyle='--', color='tab:orange', label="Service B", linewidth=4)
    plt.xlabel("Top-k popular ControlNets", loc='right')
    plt.ylabel("Invocation \n Percent (%)")
    plt.xlim(0, 15)
    plt.ylim(0, 100)
    plt.legend(loc='lower right', frameon=False, ncol=1, fontsize=32, labelspacing=0.1, handletextpad=0.1, handlelength=2)
    plt.grid(alpha=.3, linestyle='--')
    plt.savefig(CONTROLNET_INVOCATION_FIG_PATH, bbox_inches="tight")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="diffusion_model_request_trace.json", help="Path to the trace file")

    args = parser.parse_args()
    trace_file = args.trace_file

    controlnet_analysis(trace_file)
