import os
import json
import argparse
import matplotlib
from matplotlib import pyplot as plt


def lora_analysis(trace_file):
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
    LORA_INVOCATION_FIG_PATH = os.path.join(BASE_FIG_DIR, f"lora_invocation.pdf")
    print(f"LORA_INVOCATION_FIG_PATH: {LORA_INVOCATION_FIG_PATH}")


    # Service A
    print("=====Service A=====")
    lora_ids_list_A = []
    for req in req_list:
        if req["scene_id"] == "A":
            if req["loras"] is None:
                continue
            for lora_id in req["loras"]:
                if lora_id is not None:
                    lora_ids_list_A.append(lora_id)
    print(f"Number of unique lora ids: {len(set(lora_ids_list_A))}")

    lora_ids_dict_A = {}
    for lora_id in lora_ids_list_A:
        if lora_id not in lora_ids_dict_A:
            lora_ids_dict_A[lora_id] = 0
        lora_ids_dict_A[lora_id] += 1

    lora_ids_dict_A = {k: v for k, v in sorted(lora_ids_dict_A.items(), key=lambda item: item[1], reverse=True)}

    low_freq_lora_nums = 0
    for lora_id, count in lora_ids_dict_A.items():
        threshold = 0.01
        if count/len(lora_ids_list_A) >= threshold:
            print(f"Count: {count}, Percentage: {1.0*count/len(lora_ids_list_A):.4f}")
        else:
            low_freq_lora_nums += 1
    print(f"Low frequency LoRA nums: {low_freq_lora_nums}")

    cumulative_count = 0
    cumulative_percentages_A = [0]
    flag25 = False
    flag50 = False
    flag75 = False
    lora_id_cnt = 0
    for count in lora_ids_dict_A.values():
        cumulative_count += count
        cumulative_percentage = cumulative_count / len(lora_ids_list_A) * 100
        lora_id_cnt += 1
        if cumulative_percentage >= 25 and not flag25:
            print(f"25%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag25 = True
        if cumulative_percentage >= 50 and not flag50:
            print(f"50%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag50 = True
        if cumulative_percentage >= 75 and not flag75:
            print(f"75%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag75 = True
        cumulative_percentages_A.append(cumulative_percentage)

    lora_ids_A = list(lora_ids_dict_A.keys())


    # Service B
    print("=====Service B=====")
    lora_ids_B = []
    for req in req_list:
        if req["scene_id"] == "B":
            if req["loras"] is None:
                continue
            for lora_id in req["loras"]:
                if lora_id is not None:
                    lora_ids_B.append(lora_id)
    print(f"Number of unique lora ids: {len(set(lora_ids_B))}")

    lora_ids_dict_B = {}
    for lora_id in lora_ids_B:
        if lora_id not in lora_ids_dict_B:
            lora_ids_dict_B[lora_id] = 0
        lora_ids_dict_B[lora_id] += 1

    lora_ids_dict_B = {k: v for k, v in sorted(lora_ids_dict_B.items(), key=lambda item: item[1], reverse=True)}

    low_freq_lora_nums = 0
    for lora_id, count in lora_ids_dict_B.items():
        threshold = 0.01
        if count/len(lora_ids_B) >= threshold:
            print(f"Count: {count}, Percentage: {1.0*count/len(lora_ids_B):.4f}")
        else:
            low_freq_lora_nums += 1
    print(f"Low frequency LoRA nums: {low_freq_lora_nums}")

    cumulative_count = 0
    cumulative_percentages_B = [0]
    flag25 = False
    flag50 = False
    flag75 = False
    lora_id_cnt = 0
    for count in lora_ids_dict_B.values():
        cumulative_count += count
        cumulative_percentage = cumulative_count / len(lora_ids_B) * 100
        lora_id_cnt += 1
        if cumulative_percentage >= 25 and not flag25:
            print(f"25%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag25 = True
        if cumulative_percentage >= 50 and not flag50:
            print(f"50%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag50 = True
        if cumulative_percentage >= 75 and not flag75:
            print(f"75%: {cumulative_count}, cnt: {lora_id_cnt}")
            flag75 = True
        cumulative_percentages_B.append(cumulative_percentage)

    lora_ids_B = list(lora_ids_dict_B.keys())


    # Plotting
    plt.figure(figsize=(7, 3))
    plt.plot(range(len(lora_ids_A)+1), cumulative_percentages_A, color='tab:green', linestyle='-', label="Service A", linewidth=4)
    plt.plot(range(len(lora_ids_B)+1), cumulative_percentages_B, linestyle='--', color='tab:blue', label="Service B", linewidth=4)
    plt.xlabel("Top-k popular LoRAs", loc='right')
    plt.ylabel("Invocation \n Percent (%)")
    plt.xticks(range(0, 10005, 1500))
    plt.yticks(range(0, 101, 25))
    plt.xlim(None, 4500)
    plt.ylim(0, 100)
    plt.legend(loc='lower right', frameon=False, ncol=1, fontsize=32, labelspacing=0.1, handletextpad=0.1, handlelength=2)
    plt.grid(alpha=.3, linestyle='--')

    plt.savefig(LORA_INVOCATION_FIG_PATH, bbox_inches='tight')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="diffusion_model_request_trace.json", help="Path to the trace file")

    args = parser.parse_args()
    trace_file = args.trace_file

    lora_analysis(trace_file)
