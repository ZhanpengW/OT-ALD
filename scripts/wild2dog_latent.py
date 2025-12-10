"""
Class-conditional image translation from one ImageNet class to another.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import (
    load_source_data_for_domain_translation_2,
    get_image_filenames_for_label
)
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import ot
import torch.distributed as dist
from openpyxl import load_workbook

def main():
    args = create_argparser().parse_args()
    logger.log(f"arguments: {args}")
    truncated_step = args.truncated_step

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model_source, diffusion_source = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_source.load_state_dict(
        dist_util.load_state_dict(args.source_model_path, map_location="cpu")
    )
    model_source.to(dist_util.dev())
    if args.use_fp16:
        model_source.convert_to_fp16()
    model_source.eval()

    model_target, diffusion_target = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model_target.load_state_dict(
        dist_util.load_state_dict(args.target_model_path, map_location="cpu")
    )
    model_target.to(dist_util.dev())
    if args.use_fp16:
        model_target.convert_to_fp16()
    model_target.eval()
    ###########################################################
    logger.log("running source image encoding...")
    data_source = load_source_data_for_domain_translation_2(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_source_dir,
    )

    for i, (batch, extra) in enumerate(data_source):

        batch = batch.to(dist_util.dev())

        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise_source = diffusion_target.ddim_reverse_sample_loop(
            model_target,
            batch,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} source samples...")
        logger.log(f"latent with mean {noise_source.mean()} and std {noise_source.std()}")
        noise_source = ((noise_source + 1) * 127.5).clamp(0, 255)#.to(th.uint8)
        noise_source=noise_source.cpu().numpy()
        if i==0:
            break
    ###########################################################
    logger.log("running target image encoding...")
    data_target = load_source_data_for_domain_translation_2(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_target_dir,
    )

    for i, (batch, extra) in enumerate(data_target):

        batch = batch.to(dist_util.dev())

        # First, use DDIM to encode to latents.
        logger.log("encoding the target images.")
        noise_target = diffusion_target.ddim_reverse_sample_loop(
            model_target,
            batch,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} target samples...")
        logger.log(f"latent with mean {noise_target.mean()} and std {noise_target.std()}")
        noise_target = ((noise_target + 1) * 127.5).clamp(0, 255)#.to(th.uint8)
        noise_target=noise_target.cpu().numpy()
        if i==0:
            break
    ###########################################################
    DIS=[]
    for i in range(args.batch_size):
        noise_source_i = noise_source[i].flatten()
        noise_target_i = noise_target[i].flatten()
        noise_source_i=np.expand_dims(noise_source_i, axis=1)
        noise_target_i=np.expand_dims(noise_target_i, axis=1)
        N=int(len(noise_target_i)/9000)
        noise_source_i=noise_source_i[::N]
        noise_target_i=noise_target_i[::N]
        print(noise_source_i.shape)
        if dist.is_initialized():
            dist.destroy_process_group()
    
        M = ot.dist(noise_source_i, noise_target_i, metric='euclidean')
        # 均匀权重 (假设每个点的概率相等)
        a = np.ones(len(noise_source_i)) / len(noise_source_i)  # 源分布权重
        b = np.ones(len(noise_target_i)) / len(noise_target_i)  # 目标分布权重
        # 计算瓦瑟斯坦距离
        distance = ot.emd2(a, b, M)
        logger.log(f"    Wasserstein distance: {distance}")
        DIS.append(distance)
    
    # 加载已有 Excel 文件
    wb = load_workbook("experiments/DIS_LATENT_WILD2DOG.xlsx")
    ws = wb.active  # 选择默认工作表
    
    # 追加数据
    ws.append(DIS)
    
    # 保存修改
    wb.save("experiments/DIS_LATENT_WILD2DOG.xlsx")
    
    dist.barrier()
    logger.log(f"The difference calculation of latent code distributions {i} and {j} is complete.\n\n")
    
def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=20,
        eta=0.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_model_path",
        type=str,
        default="trained_models/Wild_model2480000.pt",
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        default="trained_models/Dog_model3860000.pt",
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--data_source_dir",
        type=str,
        default="datasets/afhq/val/wild",
        help="The local directory containing ImageNet validation dataset, "
             "containing filenames like ILSVRC2012_val_000XXXXX.JPG."
    )
    parser.add_argument(
        "--data_target_dir",
        type=str,
        default="datasets/afhq/val/dog",
        help="The local directory containing ImageNet validation dataset, "
             "containing filenames like ILSVRC2012_val_000XXXXX.JPG."
    )
    parser.add_argument(
        "--truncated_step",
        type=int,
        default=1000,
        help="Diffusion truncated steps."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
