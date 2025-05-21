import os

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

one_controlnet_switch_latency = 0.28

def get_latency(foldername):
    latencies = []
    with open(f"{foldername}/sdxl_pipeline.log", "r") as f:
        for line in f:
            if "End2End inference latency:" in line:
                latencies.append(float(line.split(":")[-1].strip()))
    latencies = latencies[1:-1]
    return latencies

foldernames = {
    "Diffusers": {
        "0C/0L": "./logs/diffusers-0c-0l",
        "1C/0L": "./logs/diffusers-1c-0l",
        "0C/1L": "./logs/diffusers-0c-1l",
        "1C/1L": "./logs/diffusers-1c-1l",
        "3C/0L": "./logs/diffusers-3c-0l",
        "2C/2L": "./logs/diffusers-2c-2l",
        "3C/2L": "./logs/diffusers-3c-2l",
    },
    "Nirvana-10": {
        "0C/0L": "./logs/nirvana-0c-0l-skip10",
        "1C/0L": "./logs/nirvana-1c-0l-skip10",
        "0C/1L": "./logs/nirvana-0c-1l-skip10",
        "1C/1L": "./logs/nirvana-1c-1l-skip10",
        "3C/0L": "./logs/nirvana-3c-0l-skip10",
        "2C/2L": "./logs/nirvana-2c-2l-skip10",
        "3C/2L": "./logs/nirvana-3c-2l-skip10",
    },
    "Nirvana-20": {
        "0C/0L": "./logs/nirvana-0c-0l-skip20",
        "1C/0L": "./logs/nirvana-1c-0l-skip20",
        "0C/1L": "./logs/nirvana-0c-1l-skip20",
        "1C/1L": "./logs/nirvana-1c-1l-skip20",
        "3C/0L": "./logs/nirvana-3c-0l-skip20",
        "2C/2L": "./logs/nirvana-2c-2l-skip20",
        "3C/2L": "./logs/nirvana-3c-2l-skip20",
    },
    "DistriFusion": {
        "0C/0L": 0.0,
        "1C/0L":0.0,
        "0C/1L": 0.0,
        "1C/1L": 0.0,
        "3C/0L": 0.0,
        "2C/2L": 0.0,
        "3C/2L": 0.0,
    },
    "Katz": {
        "0C/0L": "./logs/katz-0c-0l",
        "1C/0L": "./logs/katz-1c-0l",
        "0C/1L": "./logs/katz-0c-1l",
        "1C/1L": "./logs/katz-1c-1l",
        "3C/0L": "./logs/katz-3c-0l",
        "2C/2L": "./logs/katz-2c-2l",
        "3C/2L": "./logs/katz-3c-2l",
    },
}

latencies = {}
for baseline, foldername in foldernames.items():
    if baseline not in latencies:
        latencies[baseline] = {}
    if baseline != "DistriFusion":
        for setting, folder in foldername.items():
            if setting not in latencies[baseline]:
                latencies[baseline][setting] = 0.0
            latencies[baseline][setting] = np.mean(get_latency(folder))
    else:
        # parse the latencies from the DistriFusion
        distriFusionFolder = "./logs/distrifuser_benchmark_logs"
        for setting in foldernames[baseline]:
            # print(f"Setting: {setting}")
            setting_filename = setting.replace("/", "_")
            with open(f"{distriFusionFolder}/distrifuser_{setting_filename}.log", "r") as fr:
                # print(f"{distriFusionFolder}/distrifuser_{ setting_filename }.log")
                end2end_inference_times = []
                for line in fr:
                    if "End2End inference latency" in line:
                        end2end_inference_times.append(float(line.split(":")[-1]))

                latencies[baseline][setting] = np.mean(end2end_inference_times[:-1])

fig, ax = plt.subplots(figsize=(14, 5))
fontsize = 32

barWidth = 0.18
bars = []
hatches = ['-', '\\', "/", "o", "x"]
# matplotlib default colors
bar_colors = ["tab:blue", "tab:orange", "tab:green", "tab:purple", "tab:red"]

for i, (baseline, latencies) in enumerate(latencies.items()):
    x = np.arange(len(latencies))
    plot_latencies = np.array(list(latencies.values()))

    num_controlnets = np.array( [ int(item.split("/")[0][0]) for item in list(latencies.keys()) ] )
    controlnet_switch_latency = num_controlnets * one_controlnet_switch_latency
    if "Katz" not in baseline:
        plot_latencies += controlnet_switch_latency
    if "Nirvana" in baseline:
        plot_latencies += 0.35  #

    print(baseline, plot_latencies)
    bars.append(ax.bar(x + i * barWidth - 1.5*barWidth, plot_latencies, barWidth, \
                       label=baseline, hatch=hatches[i], color=bar_colors[i] , linewidth=1.5))

    # add bar height
    if baseline == "Diffusers" or baseline == "Katz":
        for j, latency in enumerate(plot_latencies):
            ax.text(x[j] + 1.05 * i * barWidth - 1.5*barWidth, latency + 0.3, f"{latency:.1f}", fontsize=fontsize-12, ha='center')

ax.grid(axis='y')
ax.set_ylim([1, 17.8])

ax.set_xlabel("Adapter settings", fontsize=fontsize)
ax.set_ylabel('Latency (s)', fontsize=fontsize)
ax.set_xticks(np.arange(len(latencies)))
ax.set_xticklabels(latencies.keys(), fontsize=fontsize)
# set y tick labels
ax.yaxis.set_ticks([2,5,8,11,14,17])

ax.tick_params(axis='y', labelsize=fontsize)

ax.legend(loc='upper left', fontsize=fontsize, frameon=False, borderaxespad=0.1, labelspacing=0.1, ncol=2, columnspacing=0.2, handletextpad=0.1, handlelength=2)
fig.tight_layout()

figure_path = "./figures"
if not os.path.exists(figure_path):
    os.makedirs(figure_path)
latency_figure = f"{figure_path}/end2end_latency.pdf"
plt.savefig(latency_figure, format='pdf', bbox_inches='tight', pad_inches=0.03)
print(f"Saved figure to {latency_figure}")
