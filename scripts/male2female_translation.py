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
import time

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
    
    logger.log("running image translation...")
    data = load_source_data_for_domain_translation_2(
        batch_size=args.batch_size,
        image_size=args.image_size,
        data_dir=args.data_dir,
    )

    for i, (batch, extra) in enumerate(data):
        logger.log(f"translating batch {i}, shape {batch.shape}.")

        logger.log("saving the original, cropped images.")
        images = ((batch + 1) * 127.5).clamp(0, 255).to(th.uint8)
        images = images.permute(0, 2, 3, 1)
        images = images.contiguous()
        images = images.cpu().numpy()
        for index in range(images.shape[0]):
            filepath = extra["filepath"][index]
            filepath = "experiments/male2female/"+filepath[19:]
            image = Image.fromarray(images[index])
            image.save(filepath)
            logger.log(f"    saving: {filepath}")

        batch = batch.to(dist_util.dev())
    
        start_time = time.time()
        # First, use DDIM to encode to latents.
        logger.log("encoding the source images.")
        noise = diffusion_source.ddim_reverse_sample_loop(
            model_source,
            batch,
            clip_denoised=False,
            device=dist_util.dev(),
            truncated_step=truncated_step,
        )
        logger.log(f"obtained latent representation for {batch.shape[0]} samples...")
        logger.log(f"latent with mean {noise.mean()} and std {noise.std()}")

        # Next, decode the latents to the target domain.
        for eta in args.eta:
            sample = diffusion_target.ddim_sample_loop(
                model_target,
                (args.batch_size, 3, args.image_size, args.image_size),
                noise=noise,
                clip_denoised=args.clip_denoised,
                device=dist_util.dev(),
                eta=eta,
                truncated_step=truncated_step+100,
            )
            sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
            sample = sample.permute(0, 2, 3, 1)
            sample = sample.contiguous()
            end_time = time.time()
            logger.log("batch translated time: ",end_time-start_time)
            
            images = []
            gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
            images.extend([sample.cpu().numpy() for sample in gathered_samples])
            logger.log(f"created {len(images) * args.batch_size} samples")
    
            logger.log("saving translated images.")
            images = np.concatenate(images, axis=0)
    
            for index in range(images.shape[0]):
                base_dir, filename = os.path.split(extra["filepath"][index])
                filename, ext = filename.split(".")
                filepath = os.path.join(base_dir, f"{filename}_translated_eta={str(eta)}_truncated_step={str(truncated_step)}.{ext}")
                filepath = "experiments/male2female/"+filepath[19:]
                image = Image.fromarray(images[index])
                image.save(filepath)
                logger.log(f"    saving: {filepath}")
    dist.barrier()
    logger.log(f"domain translation complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        batch_size=1,
        eta=[0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2],
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_model_path",
        type=str,
        default="trained_models/Cat_model2410000.pt",#Wild_model2480000
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--target_model_path",
        type=str,
        default="trained_models/Dog_model4460000.pt",
        help="Path to the diffusion model weights."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="datasets/test",
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
