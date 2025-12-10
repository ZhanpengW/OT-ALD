"""
Synthetic domain translation from a source 2D domain to a target.
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
import torch as th

def main():
    args = create_argparser().parse_args()
    logger.log(f"args: {args}")

    dist_util.setup_dist()
    logger.configure()
    logger.log("starting to sample synthetic data.")

    code_folder = os.getcwd()
    image_folder = os.path.join(code_folder, f"experiments/images")
    pathlib.Path(image_folder).mkdir(parents=True, exist_ok=True)

    truncated_step = args.truncated_step
    i = args.source
    j = args.target
    logger.log(f"reading models for synthetic data...")
    image_subfolder = os.path.join(image_folder, f"translation_cycle_{i}_{j}")
    
    #####################################################################

    source_dir = os.path.join(code_folder, f"models/synthetic/log2D{i}")
    source_model, diffusion = read_model_and_diffusion(args, source_dir)

    shapes = list(Synthetic2DType)
    shape_s = shapes[i]

    latents = []
    sources = []
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
        # print(type(noise))
        logger.log(f"obtained latent representation for {source.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")
        sources.append(source.cpu().numpy())
        latents.append(noise.cpu().numpy())
    sources = np.concatenate(sources, axis=0)
    latents = np.concatenate(latents, axis=0)
    latents_save = latents.copy()

    #####################################################################

    target_dir = os.path.join(code_folder, f"models/synthetic/log2D{j}")
    target_model, _ = read_model_and_diffusion(args, target_dir)

    shapes = list(Synthetic2DType)
    shape_s = shapes[j]

    latents_2 = []
    targets = []
    data = load_2d_data(n_samples=9000, batch_size=args.batch_size, shape=shape_s, training=False)

    for k, (target, extra) in enumerate(data):
        logger.log(f"translating {j}->{i}, batch {k}, shape {target.shape}...")
        logger.log(f"device: {dist_util.dev()}")

        target = target.to(dist_util.dev())

        noise_2 = diffusion.ddim_reverse_sample_loop(
            target_model, target,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained latent representation for {target.shape[0]} samples...")
        logger.log(f"latent with mean {noise_2.mean()} and std {noise_2.std()}")
        targets.append(target.cpu().numpy())
        latents_2.append(noise_2.cpu().numpy())
    targets = np.concatenate(targets, axis=0)
    latents_2 = np.concatenate(latents_2, axis=0)
    latents_save_2 = latents_2.copy()

    #####################################################################
    
    # 计算成本矩阵 (欧几里得距离)
    cost_matrix = ot.dist(latents, latents_2, metric='euclidean')
    
    # 定义均匀分布的权重 (可调整)
    a = np.ones(len(latents)) / len(latents)  # X 的权重
    b = np.ones(len(latents_2)) / len(latents_2)  # Y 的权重
    
    # 计算 OT 传输矩阵 (使用 线性规划 方法)
    transport_plan = ot.emd(a, b, cost_matrix)

    latents_2_new=latents.copy()
    # 计算新掩码
    for i in range(len(latents)):
        index=transport_plan[i, :].argmax()
        latents_2_new[i,:]=latents_2[index, :]
    
    #####################################################################

    latents_2_new=th.tensor(latents_2_new)
    latents_2_new=latents_2_new.to(dist_util.dev())
    target = diffusion.ddim_sample_loop(
        target_model, (9000, 2),
        noise=latents_2_new,
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
    
    #####################################################################
    
    noise_2=noise_2.cpu().numpy()
    latents_new=noise_2.copy()
    # 计算新掩码
    for j in range(len(latents_new)):
        index=transport_plan[:, j].argmax()
        latents_new[j,:]=latents[index, :]
    
    #####################################################################

    latents_new=th.tensor(latents)
    latents_new=latents_new.to(dist_util.dev())
    sources_2 = diffusion.ddim_sample_loop(
        source_model, (9000, 2),
        noise=latents_new,
        clip_denoised=False,
        device=dist_util.dev(),
        truncated_step=truncated_step,
    )
    logger.log(f"finished cycle")
    sources_2=sources_2.cpu().numpy()

    #####################################################################

    sources_image_path = os.path.join(image_subfolder, 'scatter_sources.png')
    scatter(sources, sources_image_path)
    latents_image_path = os.path.join(image_subfolder, 'scatter_latents.png')
    scatter(latents, latents_image_path)
    latents_2_new_image_path = os.path.join(image_subfolder, 'scatter_latents_2_new.png')
    scatter(latents_2_new.cpu().numpy(), latents_2_new_image_path)
    latents_2_image_path = os.path.join(image_subfolder, 'scatter_latents_2.png')
    scatter(latents_2, latents_2_image_path)
    targets_image_path = os.path.join(image_subfolder, 'scatter_targets.png')
    scatter(targets, targets_image_path)

    #####################################################################
    
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
    file_path = os.path.join(image_subfolder, 'DIS_OT.xlsx')
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
