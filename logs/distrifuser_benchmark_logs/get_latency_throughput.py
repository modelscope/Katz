import os
import numpy as np

def parse_log_file(filename):
    # Load LoRA latency: 0.00
    # End2End inference latency: 1.85
    load_lora_times = []
    end2end_inference_times = []
    throughput = None
    with open(filename, "r") as f:
        for line in f:
            if "Load LoRA latency" in line:
                load_lora_times.append(float(line.split(":")[-1]))
            if "End2End inference latency" in line:
                end2end_inference_times.append(float(line.split(":")[-1]))
            if "End2End Throughput" in line:
                throughput = float(line.split(":")[-1])

    return load_lora_times, end2end_inference_times[:-1], throughput

log_files = [ item for item in os.listdir("./") if "distrifuser" in item ]
for log_file in log_files:
    load_lora_times, end2end_inference_times, throughput = parse_log_file(log_file)
    print(f"Log file: {log_file}")
    print(f"Mean Load LoRA latency: {np.mean(load_lora_times)}, {len(load_lora_times)}")
    print(f"Mean End2End inference latency: {np.mean(end2end_inference_times)}, {len(end2end_inference_times)}")
    print(f"Throughput: {throughput}")
    print("==============================")
