"""
This script reproduces the *cycle-consistent translation* experiments in the paper.
"""

import argparse
import os
import pathlib

import numpy as np
import torch.distributed as dist

from common import read_model_and_diffusion
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import model_and_diffusion_defaults_2d, add_dict_to_argparser
from guided_diffusion.synthetic_datasets import scatter, heatmap, load_2d_data, Synthetic2DType

import ot
from openpyxl import load_workbook


def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample synthetic data to demonstrate cycle consistency.")

    code_folder = os.getcwd()
    image_folder = os.path.join(code_folder, f"experiments/images")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    truncated_step = args.truncated_step
    i = args.source
    j = args.target
    logger.log(f"reading models for synthetic data...")

    source_dir = os.path.join(code_folder, f"models/synthetic/log2D{i}")
    source_model, diffusion = read_model_and_diffusion(args, source_dir)

    target_dir = os.path.join(code_folder, f"models/synthetic/log2D{j}")
    target_model, _ = read_model_and_diffusion(args, target_dir)

    image_subfolder = os.path.join(image_folder, f"translation_cycle_{i}_{j}")
    pathlib.Path(image_subfolder).mkdir(parents=True, exist_ok=True)

    sources = []
    latents = []
    targets = []
    latents_2 = []
    sources_2 = []

    shape_s = list(Synthetic2DType)[i]
    data = load_2d_data(n_samples=9000, batch_size=args.batch_size, shape=shape_s, training=False)

    for k, (source, extra) in enumerate(data):
        logger.log(f"translating {i}->{j}, batch {k}, shape {source.shape}...")
        logger.log(f"device: {dist_util.dev()}")

        source = source.to(dist_util.dev())

        noise = diffusion.ddim_reverse_sample_loop(
            source_model, source,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained latent representation for {source.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        target = diffusion.ddim_sample_loop(
            target_model, (args.batch_size, 2),
            noise=noise,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"translated to target {target.shape}")

        noise_2 = diffusion.ddim_reverse_sample_loop(
            target_model, target,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained the second latents")
        logger.log(f"latent with mean {noise_2.mean()} and std {noise_2.std()}")

        source_2 = diffusion.ddim_sample_loop(
            source_model, (args.batch_size, 2),
            noise=noise_2,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"finished cycle")

        sources.append(source.cpu().numpy())
        latents.append(noise.cpu().numpy())
        targets.append(target.cpu().numpy())
        latents_2.append(noise_2.cpu().numpy())
        sources_2.append(source_2.cpu().numpy())

    sources = np.concatenate(sources, axis=0)
    sources_path = os.path.join(image_subfolder, 'source.npy')
    np.save(sources_path, sources)
    sources_image_path = os.path.join(image_subfolder, 'scatter_source.png')
    scatter(sources, sources_image_path)
    sources_image_path = os.path.join(image_subfolder, 'heatmap_source.png')
    heatmap(sources, sources_image_path)

    latents = np.concatenate(latents, axis=0)
    latents_path = os.path.join(image_subfolder, 'latent.npy')
    np.save(latents_path, latents)
    latents_image_path = os.path.join(image_subfolder, 'scatter_latent.png')
    scatter(latents, latents_image_path)
    latents_image_path = os.path.join(image_subfolder, 'heatmap_latent.png')
    heatmap(latents, latents_image_path)

    targets = np.concatenate(targets, axis=0)
    targets_path = os.path.join(image_subfolder, 'target.npy')
    np.save(targets_path, targets)
    targets_image_path = os.path.join(image_subfolder, 'scatter_target.png')
    scatter(targets, targets_image_path)
    targets_image_path = os.path.join(image_subfolder, 'heatmap_target.png')
    heatmap(targets, targets_image_path)

    latents_2 = np.concatenate(latents_2, axis=0)
    latents_2_path = os.path.join(image_subfolder, 'latent_2.npy')
    np.save(latents_2_path, latents_2)
    latents_2_image_path = os.path.join(image_subfolder, 'scatter_latent_2.png')
    scatter(latents_2, latents_2_image_path)
    latents_2_image_path = os.path.join(image_subfolder, 'heatmap_latent_2.png')
    heatmap(latents_2, latents_2_image_path)

    sources_2 = np.concatenate(sources_2, axis=0)
    sources_2_path = os.path.join(image_subfolder, 'source_2.npy')
    np.save(sources_2_path, sources_2)
    sources_2_image_path = os.path.join(image_subfolder, 'scatter_source_2.png')
    scatter(sources_2, sources_2_image_path)
    sources_2_image_path = os.path.join(image_subfolder, 'heatmap_source_2.png')
    heatmap(sources_2, sources_2_image_path)

    DIS=[]
    # distance = np.linalg.norm(sources - sources_2, axis=1).max()
    distance = np.linalg.norm(sources - sources_2, axis=1).mean()
    logger.log(f"computing the max L2 distance between original data points and round-trip:")
    logger.log(f"    L2 distance: {distance}")
    DIS.append(distance)
    
    M = ot.dist(sources, sources_2, metric='euclidean')
    # 均匀权重 (假设每个点的概率相等)
    a = np.ones(len(sources)) / len(sources)  # 源分布权重
    b = np.ones(len(sources_2)) / len(sources_2)  # 目标分布权重
    # 计算 2D 瓦瑟斯坦距离
    distance = ot.emd2(a, b, M)
    logger.log(f"computing the max L2 distance between original data points and round-trip:")
    logger.log(f"    Wasserstein distance: {distance}")
    DIS.append(distance)

    # 加载已有 Excel 文件
    file_path = os.path.join(image_subfolder, 'DIS.xlsx')
    wb = load_workbook(file_path)
    ws = wb.active  # 选择默认工作表
    
    # 追加数据
    ws.append(DIS)
    
    # 保存修改
    wb.save(file_path)
    
    dist.barrier()
    logger.log(f"synthetic cycle translation complete: {i}->{j}->{i}\n\n")


def create_argparser():
    defaults = dict(
        num_samples=9000,
        batch_size=3000,
        model_path=""
    )
    defaults.update(model_and_diffusion_defaults_2d())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Source synthetic dataset."
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1,
        help="Target synthetic dataset."
    )
    parser.add_argument(
        "--truncated_step",
        type=int,
        default=4000,
        help="Diffusion truncated steps."
    )
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
