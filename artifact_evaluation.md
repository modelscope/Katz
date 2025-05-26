# Artifact Evaluation for ATC 2025

This guide provides step-by-step instructions to reproduce the experiments and results presented in our Katz paper. Follow these steps to validate our claims regarding **performance improvements** and **image quality preservation**.

To simplify reproducibility, we provide an off-the-shelf Docker image, `mental2008/katz-ae:latest`, which includes all the dependencies and configurations required to run the experiments. This eliminates the need for complex environment setup. You can pull the image from [Docker Hub](https://hub.docker.com/repository/docker/mental2008/katz-ae/general) and use it as follows.

## 🚀 Run Katz with Docker

**We have pulled the image on the provided machine**, as its size is nearly 100 GiB.

```bash
# We have pulled the image on the provided machine. You can skip this. On your machine, you can pull the prebuilt Docker image with the following command.
$ docker pull mental2008/katz-ae:latest

# Run the container in detached mode with GPU support
$ docker run -d --gpus all --rm --name katz-ae --shm-size 8G mental2008/katz-ae:latest sleep infinity

# Access the container's shell
$ docker exec -it katz-ae bash

# Inside the container, activate the Conda environment
$ source activate katz

# Set the path to reference images
$ export ref_image_path=/workspace/Katz-cached-image-results/images/images_sdxl_t2i
```

## 📋 Prerequisites

### 1. Generate UNet State Dictionaries

Before running experiments, go to the project directory and generate state dictionaries for the UNet model (used for restoration):

```bash
# Go to the project directory
$ cd /workspace/Katz

# Generate default state dictionary (standard layout). Note that this can take minutes.
$ ENABLE_CHANNELS_LAST=0 python gen_unet_state_dict.py  # Generates default_unet_state_dict.pt

# Generate state dictionary with channels-last memory format. Note that this can take minutes.
$ ENABLE_CHANNELS_LAST=1 python gen_unet_state_dict.py  # Generates default_unet_state_dict_channels_last.pt
```

### 2. Reference Image Path

Export the environment variable pointing to the reference images (available in our [Modelscope repository](https://modelscope.cn/datasets/mental2008/Katz-cached-image-results)):

```bash
# The command is: export ref_image_path=/path/to/Katz-cached-image-results/images/images_sdxl_t2i
# In the provided container, run the following command.
$ export ref_image_path=/workspace/Katz-cached-image-results/images/images_sdxl_t2i
```

In the provided container, the ``ref_image_path=/workspace/Katz-cached-image-results/images/images_sdxl_t2i``.

## 🚀 End-to-End Performance

### Katz Evaluation

Evaluate Katz under different adapter configurations using provided configuration files (available in `configs/`):

```bash
# Example: 3 ControlNets and 2 LoRAs
$ python run_katz.py configs/katz-3c-2l.yml

# Other configurations
$ python run_katz.py configs/katz-0c-0l.yml  # 0 ControlNet, 0 LoRA
$ python run_katz.py configs/katz-1c-1l.yml  # 1 ControlNet, 1 LoRA
$ python run_katz.py configs/katz-2c-2l.yml  # 2 ControlNets, 2 LoRAs
```

### Baseline Comparisons

#### 1. Diffusers

```bash
# Same example: 3 ControlNets and 2 LoRAs
$ python baselines/run_baseline.py configs/diffusers-3c-2l.yml

# Other configurations
$ python baselines/run_baseline.py configs/diffusers-0c-0l.yml  # 0 ControlNet, 0 LoRA
$ python baselines/run_baseline.py configs/diffusers-1c-1l.yml  # 1 ControlNet, 1 LoRA
$ python baselines/run_baseline.py configs/diffusers-2c-2l.yml  # 2 ControlNets, 2 LoRAs
```

#### 2. Nirvana

First, generate the cached images required by Nirvana. The cached images will be saved in the `./images_sdxl_t2i_nirvana_cache/` directory.

**You can skip this step.** Note that the process of generating these images takes hours. Therefore, we have generated them and cached them in the provided container. 

```bash
$ python baselines/gen_partiprompts_detail_t2i_nirvana_cache.py
```

Then run Nirvana with different configurations:

```bash
# Example: 3 ControlNets, 2 LoRAs, 10 skipped steps
python baselines/run_baseline.py configs/nirvana-3c-2l-skip10.yml

# Other configurations
$ python baselines/run_baseline.py configs/nirvana-0c-0l-skip10.yml  # 0 ControlNet, 0 LoRA, 10 skipped steps
$ python baselines/run_baseline.py configs/nirvana-1c-1l-skip10.yml  # 1 ControlNet, 1 LoRA, 10 skipped steps
$ python baselines/run_baseline.py configs/nirvana-2c-2l-skip10.yml  # 2 ControlNet, 2 LoRA, 10 skipped steps
$ python baselines/run_baseline.py configs/nirvana-0c-0l-skip20.yml  # 0 ControlNets, 0 LoRAs, 20 skipped steps
$ python baselines/run_baseline.py configs/nirvana-1c-1l-skip20.yml  # 1 ControlNets, 1 LoRAs, 20 skipped steps
$ python baselines/run_baseline.py configs/nirvana-2c-2l-skip20.yml  # 2 ControlNets, 2 LoRAs, 20 skipped steps
$ python baselines/run_baseline.py configs/nirvana-3c-2l-skip20.yml  # 3 ControlNets, 2 LoRAs, 20 skipped steps
```

#### 3. DistriFusion

For the DistriFusion baseline, refer to the [GitHub repository](https://github.com/Suyi32/distrifuser-controlnet). More information is available in the [README](https://github.com/Suyi32/distrifuser-controlnet/blob/main/AE_README.md).

> Note: DistriFusion also modifies Diffusers library and should be run in a *separate* Conda environment.

### Performance Visualization

Generate performance comparison plots using the following commands. Note that we plot the results of setting 0C/0L, 1C/1L, 2C/2L, and 3C/2C, here. You can compare the results with those in Figure 14.

```bash
$ cd /workspace/Katz
$ python plot_end2end_latency.py
```

The script generates a PDF file: `./figures/end2end_latency.pdf`, showing performance across all tested configurations.

## 🖼️ Image Quality Assessment

Our image quality evaluation ensures that **performance optimizations** do not compromise **output quality**. Pre-generated images are also available in our [Modelscope repository](https://modelscope.cn/datasets/mental2008/Katz-cached-image-results) for reference.

### 1. Generate Test Images

**We have cached the generated images.** **And you can skip this step.** Generating all images will take hours. You can run the following command to generate images with different configurations.

```bash
# Generate images with Katz (1 ControlNet, 2 LoRAs example)
$ python run_katz.py configs/gen-images-katz-1c-2l.yml

# Generate images with other baselines
$ python baselines/run_baseline.py configs/gen-images-diffusers-1c-1l.yml  # Diffusers
$ python baselines/run_baseline.py configs/gen-images-nirvana-1c-1l-skip10.yml  # Nirvana
```

### 2. Evaluate Image Quality

Install CLIP for evaluation:

```bash
$ conda activate katz
$ pip install git+https://github.com/openai/CLIP.git
```

#### Single LoRA Evaluation (Papercut)

*It takes minutes to calculate those scores.*

```bash
# CLIP score
$ python baselines/eval_images.py --metric clip --lora-num 1 --root-dir /workspace/Katz-cached-image-results/images

# FID score
$ python baselines/eval_images.py --metric fid --lora-num 1 --root-dir /workspace/Katz-cached-image-results/images

# SSIM score
$ python baselines/eval_images.py --metric ssim --lora-num 1 --root-dir /workspace/Katz-cached-image-results/images
```

#### Two LoRAs Evaluation (Filmic + Photography)

*It takes minutes to calculate those scores.*

```bash
# CLIP score
$ python baselines/eval_images.py --metric clip --lora-num 2 --root-dir /workspace/Katz-cached-image-results/images

# FID score
$ python baselines/eval_images.py --metric fid --lora-num 2 --root-dir /workspace/Katz-cached-image-results/images

# SSIM score
$ python baselines/eval_images.py --metric ssim --lora-num 2 --root-dir /workspace/Katz-cached-image-results/images
```
