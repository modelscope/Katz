import json
import argparse


def analysis(trace_file, service):
    with open(trace_file, "r") as f:
        req_list = json.load(f)

        # filter the requests based on the service
        req_list = [req for req in req_list if req["scene_id"] == service]

    print(f"Total number of requests: {len(req_list)}")


    # Analyze the distribution of base model
    print("=====Base Model=====")
    base_model_id_list = []
    for req in req_list:
        if req["base_model_id"] is not None:
            base_model_id_list.append(req["base_model_id"])
    print(f"Number of unique base models: {len(set(base_model_id_list))}")

    base_model_dict = {}
    for base_model_id in base_model_id_list:
        if base_model_id not in base_model_dict:
            base_model_dict[base_model_id] = 0
        base_model_dict[base_model_id] += 1
    base_model_dict = {k: v for k, v in sorted(base_model_dict.items(), key=lambda item: item[1], reverse=True)}

    for base_model_id, count in base_model_dict.items():
        threshold = 0.01
        if count/len(base_model_id_list) > threshold:
            print(f"Base Model: {base_model_id}, Count: {count}, Percentage: {count/len(base_model_id_list)*100:.1f}%")


    # Analyze the distribution of controlnet number
    print("=====ControlNet=====")
    controlnet_num_list = []
    for req in req_list:
        controlnet_num = len(req["controlnets"]) if req["controlnets"] is not None else 0
        controlnet_num_list.append(controlnet_num)
    print(f"Possible number of ControlNet in a request: {len(set(controlnet_num_list))}")

    controlnet_num_dict = {}
    for controlnet_num in controlnet_num_list:
        if controlnet_num not in controlnet_num_dict:
            controlnet_num_dict[controlnet_num] = 0
        controlnet_num_dict[controlnet_num] += 1

    controlnet_num_dict = {k: v for k, v in controlnet_num_dict.items() if v >= 100}

    controlnet_num_dict = {k: v for k, v in sorted(controlnet_num_dict.items(), key=lambda item: item[0], reverse=False)}
    for controlnet_num, count in controlnet_num_dict.items():
        print(f"No. ControlNets: {controlnet_num}, Count: {count}, Percentage: {count/len(controlnet_num_list)*100:.1f}%")


    # Analyze the distribution of lora number
    print(f"=====LoRA=====")
    lora_num_list = []
    for req in req_list:
        lora_num = len(req["loras"]) if req["loras"] is not None else 0
        lora_num_list.append(lora_num)
    print(f"Possible numbers of LoRA in a request: {len(set(lora_num_list))}")

    lora_num_dict = {}
    for lora_num in lora_num_list:
        if lora_num not in lora_num_dict:
            lora_num_dict[lora_num] = 0
        lora_num_dict[lora_num] += 1

    lora_num_dict = {k: v for k, v in sorted(lora_num_dict.items(), key=lambda item: item[0], reverse=False)}
    for lora_num, count in lora_num_dict.items():
        print(f"No. LoRA: {lora_num}, Count: {count}, Percentage: {count/len(lora_num_list)*100:.1f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-file", type=str, default="diffusion_model_request_trace.json", help="Path to the trace file")
    parser.add_argument("--service", type=str, default="A", help="Service name", choices=["A", "B"])

    args = parser.parse_args()

    trace_file = args.trace_file
    service = args.service

    analysis(trace_file, service)
