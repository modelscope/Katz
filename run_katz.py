import os
import sys
import subprocess

import time

import yaml

import torch
import torch.distributed as dist


def find_available_cpus():
    proc = subprocess.Popen("numactl -s", stdout=subprocess.PIPE, shell=True)
    proc.wait()
    out = proc.communicate()[0].decode("utf-8")
    for item in out.split("\n"):
        if "physcpubind:" in item:
            cpus = item.split(":")[1].strip().split(" ")
    available_cpus = [int(item) for item in cpus]
    return available_cpus


def create_lora_loader(env, log_folder, available_cpus, config):
    # nohup python3 -u load_lora_shm_multi.py --lora_loader_id 1 2>&1 &
    max_lora_num = config["max_lora_num"]

    cpu_ids = []
    cpus_per_lora = 3
    for i in range(max_lora_num):
        if (len(available_cpus) < cpus_per_lora):
            raise ValueError(f"Not enough CPUs for lora {i}, available_cpus: {available_cpus}")
        lora_cpus = []
        for _ in range(cpus_per_lora):
            lora_cpus.append(available_cpus.pop(0))
        cpu_ids.append(f"{','.join([str(cpu) for cpu in lora_cpus])}")
    print(f"CPU IDs for LoRA Loaders: {cpu_ids}")

    processes = []

    for lora_loader_id in range(max_lora_num):
        log_path = f"{log_folder}/lora_loader_{lora_loader_id}.log"

        command = f"exec numactl --physcpubind={cpu_ids[lora_loader_id]} python -u load_lora_shm_multi.py "
        command += f"--lora_loader_id {lora_loader_id} "
        command += f"> {log_path} 2>&1"
        print(f"Running LoRA Loader {lora_loader_id}: {command}")

        p = subprocess.Popen(command, shell=True, env=env)
        time.sleep(10)

        processes.append(p)
    return processes


def create_controlnets(env, log_folder, available_cpus, config):
    # nohup python3 -u sdxl_controlnet_server_nvlink.py --world_size 3 --gpu_id 2 --rank 2 > sdxl_controlnet_server_nvlink_2.log 2>&1 &
    # nohup python -u sdxl_controlnet_server.py --world-size 8 --gpu-id 2 --rank 2 --dst-rank 0 --controlnet-parallel-rank 2 4 6 > controlnet_server_2.log 2>&1 &
    
    world_size = config["world_size"]

    num_controlnets = config["num_controlnets"]

    latent_parallel = config["latent_parallel"]

    cpu_ids = []
    cpus_per_controlnet = 2
    for i in range(num_controlnets):
        if (len(available_cpus) < cpus_per_controlnet):
            raise ValueError(f"Not enough CPUs for controlnet {i}, available_cpus: {available_cpus}")

        controlnet_cpus = []
        for _ in range(cpus_per_controlnet):
            controlnet_cpus.append(available_cpus.pop(0))
        cpu_ids.append(f"{','.join([str(cpu) for cpu in controlnet_cpus])}")

        if latent_parallel:
            controlnet_cpus = []
            for _ in range(cpus_per_controlnet):
                controlnet_cpus.append(available_cpus.pop(0))
            cpu_ids.append(f"{','.join([str(cpu) for cpu in controlnet_cpus])}")
    print(f"CPU IDs for ControlNets: {cpu_ids}")

    processes = []

    if not latent_parallel:
        dst_rank = 0
        controlnet_rank_list = [1 + i for i in range(num_controlnets)]
        controlnet_parallel_rank = f"{' '.join([str(rank) for rank in controlnet_rank_list])}"
        for controlnet_id in controlnet_rank_list:
            log_path = f"{log_folder}/controlnet_{controlnet_id}.log"

            command = f"exec numactl --physcpubind={cpu_ids[controlnet_id-1]} python -u sdxl_controlnet_server.py "
            command += f"--world-size {world_size} --gpu-id {controlnet_id} --rank {controlnet_id} --dst-rank {dst_rank} "
            command += f"--controlnet-parallel-rank {controlnet_parallel_rank} "
            command += f"> {log_path} 2>&1"
            print(f"Running ControlNet {controlnet_id}: {command}")
            p = subprocess.Popen(command, shell=True, env=env)

            processes.append(p)
    else:
        dst_rank = 0
        controlnet_rank_list = [(2 + 2 * i) for i in range(num_controlnets)]
        controlnet_parallel_rank = f"{' '.join([str(rank) for rank in controlnet_rank_list])}"
        for controlnet_id in controlnet_rank_list:
            log_path = f"{log_folder}/controlnet_{controlnet_id}.log"

            command = f"exec numactl --physcpubind={cpu_ids[controlnet_id-2]} python -u sdxl_controlnet_server.py "
            command += f"--world-size {world_size} --gpu-id {controlnet_id} --rank {controlnet_id} --dst-rank {dst_rank} --controlnet-parallel-rank {controlnet_parallel_rank} "
            command += f"> {log_path} 2>&1"
            print(f"Running ControlNet {controlnet_id}: {command}")
            p = subprocess.Popen(command, shell=True, env=env)

            processes.append(p)

        dst_rank = 1
        controlnet_rank_list = [(3 + 2 * i) for i in range(num_controlnets)]
        controlnet_parallel_rank = f"{' '.join([str(rank) for rank in controlnet_rank_list])}"
        print(f"ControlNet Parallel Rank: {controlnet_parallel_rank}")
        for controlnet_id in controlnet_rank_list:
            log_path = f"{log_folder}/controlnet_{controlnet_id}.log"

            command = f"exec numactl --physcpubind={cpu_ids[controlnet_id-2]} python -u sdxl_controlnet_server.py "
            command += f"--world-size {world_size} --gpu-id {controlnet_id} --rank {controlnet_id} --dst-rank {dst_rank} --controlnet-parallel-rank {controlnet_parallel_rank} "
            command += f"> {log_path} 2>&1"
            print(f"Running ControlNet {controlnet_id}: {command}")
            p = subprocess.Popen(command, shell=True, env=env)

            processes.append(p)

    return processes


def create_sdxl_unet_server(env, log_folder, available_cpus, config):
    # nohup python -u sdxl_unet_server.py --world-size 8 --gpu-id 1 --rank 1 --controlnet-parallel --controlnet-parallel-rank 3 5 7 --num-controlnets 3 > unet_server_1.log 2>&1 &

    world_size = config["world_size"]

    controlnet_parallel = config["controlnet_parallel"]
    num_controlnets = config["num_controlnets"]

    lora_mode = config["lora_mode"]
    load_lora_mode = config["load_lora_mode"]
    max_lora_num = config["max_lora_num"]

    cpus_per_unet = 6
    if (len(available_cpus) < cpus_per_unet):
        raise ValueError(f"Not enough CPUs for unet, available_cpus: {available_cpus}")
    unet_cpus = []
    for _ in range(cpus_per_unet):
        unet_cpus.append(available_cpus.pop(0))
    cpu_id = f"{','.join([str(cpu) for cpu in unet_cpus])}"
    print(f"CPU IDs for UNets: {cpu_id}")

    log_path = f"{log_folder}/unet_server.log"
    command = f"exec numactl --physcpubind={cpu_id} python -u sdxl_unet_server.py "
    command += f"--world-size {world_size} --gpu-id 1 --rank 1 "
    # ControlNet
    if controlnet_parallel:
        command += "--controlnet-parallel "
        controlnet_rank_list = [(3 + 2 * i) for i in range(num_controlnets)]
        command += f"--controlnet-parallel-rank {' '.join([str(rank) for rank in controlnet_rank_list])} "
    command += f"--num-controlnets {num_controlnets} "
    # LoRA
    command += f"--lora-mode {lora_mode} --load-lora-mode {load_lora_mode} --max-lora-num {max_lora_num} "
    command += f"> {log_path} 2>&1"

    print(f"Running SDXL Unet Server: {command}")

    p = subprocess.Popen(command, shell=True, env=env)
    return [p]


def create_sdxl_pipeline(env, log_folder, available_cpus, config):
    # python sdxl_pipeline.py --world-size 8 --gpu-id 0 --rank 0 --latent-parallel --controlnet-parallel --controlnet-parallel-rank 2 4 6 --num-controlnets 3

    num_prompts = config["num_prompts"] if "num_prompts" in config else -1
    ref_image_path = config["ref_image_path"] if "ref_image_path" in config else env["ref_image_path"]
    assert ref_image_path != ""
    output_image_path = config["output_image_path"] if "output_image_path" in config else None

    world_size = config["world_size"]

    controlnet_parallel = config["controlnet_parallel"]
    num_controlnets = config["num_controlnets"]

    lora_mode = config["lora_mode"]
    load_lora_mode = config["load_lora_mode"]
    max_lora_num = config["max_lora_num"]

    latent_parallel = config["latent_parallel"]

    cpus_per_pipeline = 6
    if (len(available_cpus) < cpus_per_pipeline):
        raise ValueError(f"Not enough CPUs for pipeline, available_cpus: {available_cpus}")
    pipeline_cpus = []
    for _ in range(cpus_per_pipeline):
        pipeline_cpus.append(available_cpus.pop(0))
    cpu_id = f"{','.join([str(cpu) for cpu in pipeline_cpus])}"
    print(f"CPU IDs for SDXL Pipeline: {cpu_id}")

    log_path = f"{log_folder}/sdxl_pipeline.log"
    command = f"exec numactl --physcpubind={cpu_id} python sdxl_pipeline.py "
    command += f"--num-prompts {num_prompts} --ref-image-path {ref_image_path} "
    if output_image_path is not None:
        command += f"--output-image-path {output_image_path} "
    command += f"--world-size {world_size} --gpu-id 0 --rank 0 "
    # ControlNet
    if controlnet_parallel:
        command += "--controlnet-parallel "
        controlnet_rank_list = [(2 + 2 * i) for i in range(num_controlnets)] if latent_parallel else [i + 1 for i in range(num_controlnets)]
        command += f"--controlnet-parallel-rank {' '.join([str(rank) for rank in controlnet_rank_list])} "
    command += f"--num-controlnet {num_controlnets} "
    # LoRA
    command += f"--lora-mode {lora_mode} --load-lora-mode {load_lora_mode} --max-lora-num {max_lora_num} "
    # Base Model
    if latent_parallel:
        command += "--latent-parallel "
    command += f"> {log_path} 2>&1"

    print(f"Running SDXL pipeline: {command}")

    p = subprocess.Popen(command, shell=True, env=env)
    return [p]


def cleanup(all_procs):
    print("Cleaning up...")
    # kill all
    for p in all_procs:
        print(f"kill {p.pid}")
        # maybe we need kill using sigkill?
        os.system(f"kill -TERM {p.pid} > /dev/null 2>&1")


def main():
    assert len(sys.argv) == 2, "python run_swiftdiffusion.py <config_path>"

    project_path = os.getcwd()
    print(f"Project Path: {project_path}")

    available_cpus = find_available_cpus()
    print(f"Available CPUs: {available_cpus}")
    
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    assert isinstance(config, dict), "Failed to load the config"
    print(f"Config: {config}")

    world_size = config["world_size"]

    # ControlNet
    controlnet_parallel = config["controlnet_parallel"]
    assert type(controlnet_parallel) == bool, "controlnet_parallel: {}".format(controlnet_parallel)
    num_controlnets = config["num_controlnets"]
    assert num_controlnets in [0,1,2,3], f"num_controlnets: {num_controlnets}"

    # LoRA
    lora_mode = config["lora_mode"] # how to serve lora
    assert lora_mode in ["without", "full", "sync"], "lora_mode: {}".format(lora_mode)

    load_lora_mode = config["load_lora_mode"] # how to load lora
    assert load_lora_mode in ["default", "async"], "load_lora_mode: {}".format(load_lora_mode)

    max_lora_num = config["max_lora_num"]
    assert max_lora_num in [0,1,2], "max_lora_num: {}".format(max_lora_num)

    latent_parallel = config["latent_parallel"]
    assert type(latent_parallel) == bool, f"latent_parallel: {latent_parallel}"

    total_gpu_num = torch.cuda.device_count()
    assert world_size <= total_gpu_num, "world_size: {}; total_gpu_num: {}".format(world_size, total_gpu_num)

    print(f"Total GPUs in the world: {total_gpu_num}")
    print(f"torch.distributed.is_available(): {dist.is_available()}")
    print(f"torch.distributed.is_nccl_available(): {dist.is_nccl_available()}")

    sdxl_env = os.environ.copy()
    sdxl_env['ENABLE_CUDA_GRAPH']      = config['ENABLE_CUDA_GRAPH']
    sdxl_env['ENABLE_FUSED_GEGLU']     = config['ENABLE_FUSED_GEGLU']
    sdxl_env['ENABLE_CHANNELS_LAST']   = config['ENABLE_CHANNELS_LAST']
    sdxl_env['ENABLE_FUSED_NORM_SILU'] = config['ENABLE_FUSED_NORM_SILU']

    # create log folder
    log_folder = config.get("log_folder", None)
    if log_folder is None:
        log_root_dir = "logs"
        current_datetime = time.strftime("%m%d_%H%M%S")
        log_folder = f"{current_datetime}_{config_path.split('/')[-1].split('.')[0]}"
        log_folder = os.path.join(log_root_dir, log_folder)
    print(f"Log folder: {log_folder}")
    os.makedirs(log_folder, exist_ok=True)
    os.system('cp {} {}'.format(sys.argv[1], log_folder)) # cp config file to log folder

    all_procs = []
    try:
        if lora_mode == "sync" and load_lora_mode == "async":
            loader_processes = create_lora_loader(sdxl_env, log_folder, available_cpus, config)
            all_procs += loader_processes
            time.sleep(20)

        if controlnet_parallel:
            controlnet_processes = create_controlnets(sdxl_env, log_folder, available_cpus, config)
            all_procs += controlnet_processes
            time.sleep(3)

        if latent_parallel:
            unet_processes = create_sdxl_unet_server(sdxl_env, log_folder, available_cpus, config)
            all_procs += unet_processes
            time.sleep(5)

        sdxl_processes = create_sdxl_pipeline(sdxl_env, log_folder, available_cpus, config)
        all_procs += sdxl_processes

        sdxl_processes[0].wait()
    except Exception as e:
        print("Error:", e)
    finally:
        cleanup(all_procs)

if __name__ == '__main__':
    main()
