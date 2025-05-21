# Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters

Katz is a high-performance serving system designed specifically for diffusion model workflows with multiple adapters. It dramatically improves inference efficiency while maintaining image quality.

## 🌟 Key Features

- **ControlNet-as-a-Service**: Decouples ControlNets from the base model for independent scaling.
- **Bounded Asynchronous LoRA Loading**: Overlaps LoRA loading with base model execution for reduced latency.
- **Latent Parallelism**: Accelerates base model execution across multiple GPUs.
- **Performance Gains**: Up to $7.8 \times$ latency reduction and $1.7 \times$ throughput improvement.

## 🎬 Demo

![](./assets/katz_demo.gif)

**Prompt**: papercut -subject/scene-a shiba inu wearing a beret and black turtleneck, 4k, clean background

**Negative prompt**: low quality, bad quality, sketches, numbers, letters

This image was generated with **1 ControlNet** with depth guidance and **1 LoRA** for the papercut style. The depth reference image used for guidance is available [here](./assets/demo_image_depth.png).

## 🚀 Getting Started

### System Requirements

- NVIDIA GPUs (H800 recommended for best performance)
- CUDA 11.8+
- Python 3.10+

### 🚧 Environment Setup

```bash
$ conda create -n katz python=3.10
$ conda activate katz
$ pip install -r requirements.txt
# Install our customized diffusers package
$ pushd ./diffusers-hf && pip install -e . && popd
# Install fast-kernel
$ pushd ./diffusers-hf/src/fast_kernel/ && git submodule update --init --recursive && pip install . && popd
```

## 🔥 Quickstart Example

Coming soon.

## 🔮 Artifact Evaluation

For detailed benchmarking instructions and reproducing our results, see the [artifact evaluation](./artifact_evaluation.md) guide.

## 🗄️ Production Trace Analysis

We provide tools and datasets for analyzing real-world production traces in the [trace](./trace/README.md) directory.

## 📝 Citation

Please cite our paper if it is helpful to your research.

```bibtex
@inproceedings{Katz2025,
  title = {Katz: Efficient Workflow Serving for Diffusion Models with Many Adapters},
  author = {Li, Suyi and Yang, Lingyun and Jiang, Xiaoxiao and Lu, Hanfeng and An, Dakai and Di, Zhipeng and Lu, Weiyi and Chen, Jiawei and Liu, Kan and Yu, Yinghao and Lan, Tao and Yang, Guodong and Qu, Lin and Zhang, Liping and Wang, Wei},
  booktitle = {Proc. USENIX ATC},
  year = {2025}
}
```

## 🙏🏻 Acknowledgement

We thank the contributors of [🤗 Diffusers](https://github.com/huggingface/diffusers) for their foundational work. 

## 📬 Contact

For questions and support, please open an issue or contact the authors.
