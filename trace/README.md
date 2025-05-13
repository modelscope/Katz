# Trace Analysis on Diffusion Model Serving

## Dataset Overview

This dataset captures the **invocation characteristics** of approximately **550,000 text-to-image requests** collected over a span of **20 days**.

Key features of the dataset include:
- Base Models: The primary models used for text-to-image generation.
- ControlNet: A collection of ControlNet models applied to refine outputs.
- LoRA: A list of LoRA (Low-Rank Adaptation) models utilized for fine-tuning.
- Services: Data spans two distinct services (Service A and Service B).

Each request entry includes the following fields:
- timestamp: The time when the request was made.
- ip: The name of the serving instance.
- scene_id: The service identifier (Service A or Service B).
- base_model_id: The anonymized ID of the base model.
- controlnets: A list of anonymized ControlNet model IDs applied.
- loras: A list of anonymized LoRA model IDs utilized.

The dataset, named `diffusion_model_request_trace.json`, is available for download at: [ModelScope](https://modelscope.cn/datasets/mental2008/T2I-Model-Serving-Request-Trace)

## Download Instructions

To download the dataset, run the following command:

```bash
$ modelscope download --dataset 'mental2008/T2I-Model-Serving-Request-Trace' diffusion_model_request_trace.json
```

Ensure you have the modelscope CLI installed. If not, install it via:

```bash
$ pip install modelscope[framework]
```

## Statistical Analysis

You can perform statistical analysis on requests originating from either Service A or Service B using the following commands:

```bash
# For Service A
$ python statistical_analysis.py --service A

# For Service B
$ python statistical_analysis.py --service B
```

## Invocation Analysis

### ControlNet

Analyze the invocation patterns of ControlNet models with:

```bash
$ python controlnet_invocation_analysis.py
```

Result location: [`figures/controlnet_invocation.pdf`](./figures/controlnet_invocation.pdf).

### LoRA

Analyze the invocation patterns of LoRA models with:

```bash
$ python lora_invocation_analysis.py
```

Result location: [`figures/lora_invocation.pdf`](./figures/lora_invocation.pdf).

## Adapter Loading Analysis

Investigate adapter loading behavior under different caching policies (LRU and LFU) for both services:

```bash
# LRU policy
$ python adapter_loading_analysis.py --service A --cache-policy LRU
$ python adapter_loading_analysis.py --service B --cache-policy LRU

# LFU policy
$ python adapter_loading_analysis.py --service A --cache-policy LFU
$ python adapter_loading_analysis.py --service B --cache-policy LFU
```

Result locations:
- LRU policy
  - [`figures/adapter_loading_LRU_A.pdf`](./figures/adapter_loading_LRU_A.pdf)
  - [`figures/adapter_loading_LRU_B.pdf`](./figures/adapter_loading_LRU_B.pdf)
- LFU policy
  - [`figures/adapter_loading_LFU_A.pdf`](./figures/adapter_loading_LFU_A.pdf)
  - [`figures/adapter_loading_LFU_B.pdf`](./figures/adapter_loading_LFU_B.pdf)

Additionally, results showing the count of unique LoRAs loaded per instance are available here:
- [`figures/unique_loras_A.pdf`](./figures/unique_loras_A.pdf)
- [`figures/unique_loras_B.pdf`](./figures/unique_loras_B.pdf)
