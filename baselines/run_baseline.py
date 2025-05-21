import os
import subprocess
import time
import re
import sys
import json
import yaml


def cleanup(all_procs):
    print("Cleaning up...")
    # kill all
    for p in all_procs:
        print("kill", p.pid)
        # maybe we need kill using sigkill?
        os.system(f"kill -TERM {p.pid} > /dev/null 2>&1")

def create_sdxl_pipeline(env_, logFolder, py_script_name, configs):
    num_prompts = configs["num_prompts"] if "num_prompts" in configs else -1
    ref_image_path = configs["ref_image_path"] if "ref_image_path" in configs else env_["ref_image_path"]
    assert ref_image_path != ""
    output_image_path = configs["output_image_path"] if "output_image_path" in configs else None

    num_controlnets = configs["num_controlnets"]
    num_loras = configs["num_loras"]
    skipped_steps = configs["skipped_steps"] if "skipped_steps" in configs else 0

    logpath = f"{logFolder}/sdxl_pipeline.log"
    test_command = f"exec python3 {py_script_name} "
    test_command += f"--num-prompts {num_prompts} --ref-image-path {ref_image_path} "
    if output_image_path is not None:
        test_command += f"--output-image-path {output_image_path} "
    test_command += f"--num-controlnets {num_controlnets} --num-loras {num_loras} --skipped-steps {skipped_steps} "
    test_command += f"> {logpath} 2>&1"
    print(f"Command: {test_command}")

    p = subprocess.Popen(test_command, shell=True, env=env_)
    return [p]

def main():
    assert len(sys.argv) >= 2, "python run_baseline.py <config_path>"
    project_path = os.getcwd()
    print("Project Path", project_path)

    ################# Configs #################
    config_path = sys.argv[1]
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    assert isinstance(configs, dict), "Config load failed"
    ################# Configs #################
    print("Configs", configs)

    # create log folder
    log_folder = configs.get("log_folder", None)
    if log_folder is None:
        log_root_dir = "logs"
        current_datetime = time.strftime("%m%d_%H%M%S")
        log_folder = f"{current_datetime}_{config_path.split('/')[-1].split('.')[0]}"
        log_folder = os.path.join(log_root_dir, log_folder)
    print(f"Log folder: {log_folder}")
    os.makedirs(log_folder, exist_ok=True)
    os.system('cp {} {}'.format(sys.argv[1], log_folder)) # cp config file to log folder

    py_script_name = configs['script']
    assert os.path.exists(py_script_name), f"Script {py_script_name} does not exist"

    sdxl_env = os.environ.copy()

    all_procs = []
    try:
        sdxl_processes = create_sdxl_pipeline(sdxl_env, log_folder, py_script_name, configs)

        all_procs += sdxl_processes

        sdxl_processes[0].wait()
    except Exception as e:
        print("Error:", e)
    finally:
        cleanup(all_procs)

if __name__ == '__main__':
    main()
