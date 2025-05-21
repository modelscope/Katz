import os 
import argparse

import torch
_ = torch.manual_seed(42)

from torchvision.io import read_image, ImageReadMode

from torchmetrics.functional.multimodal import clip_score
from torchmetrics.multimodal.clip_score import CLIPScore

from cleanfid import fid
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image import StructuralSimilarityIndexMeasure

from utils import process_prompt, read_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", type=str, default="/project/infattllm/slida/katz-ae-images/Katz-cached-image-results/images")
    parser.add_argument("--num-images", type=int, default=-1, help="Number of images to evaluate. -1 means all images")
    parser.add_argument("--metric", type=str, default="clip", choices=["clip", "fid", "lpips", "ssim"])
    parser.add_argument("--lora-num", type=int, default=1, choices=[1, 2], help="Number of LoRAs")
    args = parser.parse_args()

    root_dir = args.root_dir
    num_images = args.num_images
    metric = args.metric
    lora_num = args.lora_num
    print(f"Root dir: {root_dir}, num images: {num_images}, metric: {metric}, lora_num: {lora_num}")

    if metric == "clip":
        # CLIP score
        # https://lightning.ai/docs/torchmetrics/stable/multimodal/clip_score.html#module-interface
        clip_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16")
    elif metric == "lpips":
        # lpips score
        lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='vgg')

    if metric == "clip":
        if lora_num == 1:
            target_folders = {
                "Diffusers": f"{root_dir}/diffusers-1c-1l",
                "NoLoRA": f"{root_dir}/images_nolora",
                "Nirvana-10": f"{root_dir}/nirvana-1c-1l-skip10",
                "Nirvana-20": f"{root_dir}/nirvana-1c-1l-skip20",
                "DistriFusion": f"{root_dir}/distrifusion-1c-1l",
                "Katz": f"{root_dir}/katz-1c-1l",
            }
        elif lora_num == 2:
            target_folders = {
                "Diffusers": f"{root_dir}/diffusers-1c-2l",
                "NoLoRA": f"{root_dir}/images_nolora",
                "Nirvana-10": f"{root_dir}/nirvana-1c-2l-skip10",
                "Nirvana-20": f"{root_dir}/nirvana-1c-2l-skip20",
                "DistriFusion": f"{root_dir}/distrifusion-1c-2l",
                "Katz": f"{root_dir}/katz-1c-2l",
            }
        else:
            raise ValueError("Invalid LoRA number")
    else:
        if lora_num == 1:
            target_folders = {
                "NoLoRA": [f"{root_dir}/diffusers-1c-1l", f"{root_dir}/images_nolora"],
                "Nirvana-10": [f"{root_dir}/diffusers-1c-1l", f"{root_dir}/nirvana-1c-1l-skip10"],
                "Nirvana-20": [f"{root_dir}/diffusers-1c-1l", f"{root_dir}/nirvana-1c-1l-skip20"],
                "DistriFusion": [f"{root_dir}/diffusers-1c-1l", f"{root_dir}/distrifusion-1c-1l"],
                "Katz": [f"{root_dir}/diffusers-1c-1l", f"{root_dir}/katz-1c-1l"],
            }
        elif lora_num == 2:
            target_folders = {
                "NoLoRA": [f"{root_dir}/diffusers-1c-2l", f"{root_dir}/images_nolora"],
                "Nirvana-10": [f"{root_dir}/diffusers-1c-2l", f"{root_dir}/nirvana-1c-2l-skip10"],
                "Nirvana-20": [f"{root_dir}/diffusers-1c-2l", f"{root_dir}/nirvana-1c-2l-skip20"],
                "DistriFusion": [f"{root_dir}/diffusers-1c-2l", f"{root_dir}/distrifusion-1c-2l"],
                "Katz": [f"{root_dir}/diffusers-1c-2l", f"{root_dir}/katz-1c-2l"],
            }
        else:
            raise ValueError("Invalid LoRA number")

    prompts = read_prompts(num_prompts=num_images)
    if lora_num == 1:
        prompt_prefix = "papercut -subject/scene-"
    elif lora_num == 2:
        prompt_prefix = "by william eggleston, "
    else:
        raise ValueError("Invalid foldername")
    prompt_suffix = ", 4k, clean background"

    for baseline, foldernames in target_folders.items():
        print("=====================================")
        if metric == "clip":
            if num_images != -1:
                prompts = prompts[:num_images]

            folder = foldernames

            clip_prompts = []
            clip_images = []
            for prompt_id, prompt in enumerate(prompts):
                prompt = process_prompt(prompt_prefix, prompt, prompt_suffix)
                image_filename = os.path.join(folder, f"image_{prompt_id}.png")
                image = read_image(image_filename)  # shape: (3, H, W)

                clip_prompts.append(prompt)
                clip_images.append(image)

            assert len(clip_images) == len(clip_prompts), (len(clip_images), len(clip_prompts))
            clip_score = clip_metric(clip_images, clip_prompts)
            print(f"Baseline: \033[32m{baseline}\033[0m, CLIP score: \033[31m{clip_score.detach():.1f}\033[0m")
        else:  # other metrics (e.g., fid, lpips, ssim)
            gold_foldername = foldernames[0]
            pred_foldername = foldernames[1]
            print(f"Ground truth folder: {gold_foldername}, Prediction folder: {pred_foldername}")

            gold_images = []
            num_gold_images = len([item for item in os.listdir(gold_foldername) if "image_" in item]) if num_images == -1 else num_images
            for image_id in range(num_gold_images):
                try:
                    image_filename = os.path.join(gold_foldername, f"image_{image_id}.png")
                    gold_images.append(read_image(image_filename,  mode=ImageReadMode.RGB))  # shape: (3, H, W)
                except:
                    break

            pred_images = []
            num_pred_images = len([item for item in os.listdir(pred_foldername) if "image_" in item]) if num_images == -1 else num_images
            for image_id in range(num_pred_images):
                try:
                    image_filename = os.path.join(pred_foldername, f"image_{image_id}.png")
                    pred_images.append(read_image(image_filename,  mode=ImageReadMode.RGB))
                except:
                    break

            # print(f"Number of ground truth images: {len(gold_images)}, number of prediction images: {len(pred_images)}")
            test_image_num = min(len(gold_images), len(pred_images))

            if metric == "fid":
                fid_score_vit = fid.compute_fid(gold_foldername, pred_foldername, model_name="clip_vit_b_32")

                print(f"Baseline: \033[32m{baseline}\033[0m, FID score: \033[31m{fid_score_vit:.1f}\033[0m")
            elif metric == "lpips":
                lpips_score = lpips_metric(torch.stack(gold_images[:test_image_num], dim=0) / 255.0, torch.stack(pred_images[:test_image_num], dim=0) / 255.0 )

                print(f"Baseline: \033[32m{baseline}\033[0m, lpips score: \033[31m{lpips_score.detach():.2f}\033[0m")
            elif metric == "ssim":
                def cal_ssim(**kwargs):
                    gold_images = kwargs["gold_images"]
                    pred_images = kwargs["pred_images"]
                    ssim = StructuralSimilarityIndexMeasure(data_range=(0.0, 255.0))

                    # convert to BxCxHxW
                    gold_images = torch.stack(gold_images, dim=0) * 1.0
                    pred_images = torch.stack(pred_images, dim=0) * 1.0

                    ssim_score = ssim(pred_images, gold_images)
                    return ssim_score

                ssim_score = cal_ssim(gold_images=gold_images[:test_image_num], pred_images=pred_images[:test_image_num])

                print(f"Baseline: \033[32m{baseline}\033[0m, SSIM scores: \033[31m{ssim_score:.2f}\033[0m")
            else:
                raise ValueError("Invalid metric")
