# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

"""
Samples a large number of images from a pre-trained DiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo: https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import pytorch_lightning as pl
from models import DiT_models as DummyDiT_models
from download import find_model
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from tqdm import tqdm, trange
import os
from PIL import Image
import numpy as np
import math
import argparse
from typing import List, Optional


def int_or_tuple(value):
    """
    A helper function to convert the input number to a tuple if it is a range
    """
    if "," in value:
        return tuple(map(int, value.split(",")))
    return int(value)

def create_npz_from_sample_folder(sample_dir, num=50_000):
    """
    Builds a single .npz file from a folder of .png samples.
    """
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(args, unparsed):
    """
    Run sampling.
    """
    if args.tf32:
        tf32 = True
        torch.backends.cudnn.allow_tf32 = bool(tf32)
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
        torch.set_float32_matmul_precision('high' if tf32 else 'highest')
        print(f"Fast inference mode is enabled🏎️🏎️🏎️. TF32: {tf32}")
    else:
        print("Fast inference mode is disabled🐢🐢🐢, you may enable it by passing the '--fast-inference' flag!")
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU. sample.py supports CPU-only usage"
    torch.set_grad_enabled(False)

    # Setup Single GPU experiments settings
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    seed = args.global_seed
    pl.seed_everything(seed)    

    opts = dict()
    from utils.parser_setter import extract_parser, printopt
    extract_parser(unparsed, opts)
    print('----> Opt printed as follows:')
    printopt(opts)

    '''model selection'''
    if args.model_domain == 'dit':
        from models import DiT_models
    elif args.model_domain == 'udit':
        from udit_models import DiT_models
    else:
        raise NotImplementedError(f'{args.model_domain} not implemented...')

    if args.ckpt is None:
        assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000

    # Load model:
    latent_size = args.image_size // 8
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        **(opts['network_g'] if opts.get('network_g') is not None else dict())
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
    ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model = torch.compile(model)
    model.eval()  # important!
    diffusion = create_diffusion(str(args.num_sampling_steps))
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    assert args.cfg_scale >= 1.0, "In almost all cases, cfg_scale be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Create folder to save samples:
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    folder_name = f"{model_string_name}-{ckpt_string_name}-size-{args.image_size}-vae-{args.vae}-" \
                  f"cfg-{args.cfg_scale}-seed-{args.global_seed}"
    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    os.makedirs(sample_folder_dir, exist_ok=True)
    print(f"Saving .png samples at {sample_folder_dir}")

    # Figure out how many samples we need to generate on a single GPU and how many iterations we need to run:
    batch_size = args.batch_size
    # To make things evenly-divisible, we'll sample a bit more than we need and then discard the extra samples:
    total_steps = int(math.ceil(args.num_samples / batch_size)) 
    print(f"Total Steps for Sampling: {total_steps}")
    if isinstance(args.num_classes, tuple):
        pbar = tqdm(range(args.num_classes[0], args.num_classes[1]), desc="Sampling", disable=False)
    elif isinstance(args.num_classes, str) and ',' in args.num_classes:
        start, end = map(int, args.num_classes.split(','))
        pbar = tqdm(range(start, end), desc="Sampling", disable=False)
    else:
        pbar = trange(args.num_classes, desc="Sampling", disable=False)
    total = 0
    iterator = args.num_samples_per_class // args.batch_size 
    iteration_per_class = iterator + 1 if args.num_samples_per_class % args.batch_size != 0 else iterator
    for class_label in pbar:
        for _ in range(iteration_per_class):
            # Sample inputs:
            z = torch.randn(batch_size, model.in_channels, latent_size, latent_size, device=device)
            y = torch.tensor([class_label] * batch_size, device=device)

            # Setup classifier-free guidance:
            if using_cfg:
                z = torch.cat([z, z], 0)
                y_null = torch.tensor([1000] * batch_size, device=device)
                y = torch.cat([y, y_null], 0)
                model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                sample_fn = model.forward_with_cfg
            else:
                model_kwargs = dict(y=y)
                sample_fn = model.forward

            # Sample images:
            samples = diffusion.p_sample_loop(
                sample_fn, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=False, device=device
            )
            if using_cfg:
                samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

            samples = vae.decode(samples / 0.18215).sample
            samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

            # Save samples to disk as individual .png files
            for i, sample in enumerate(samples):
                index = i + total
                Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")
            total += batch_size

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="U-DiT-S")
    parser.add_argument("--model-domain", type=str, default='udit')
    parser.add_argument("--vae",  type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int_or_tuple, default=1000) # this should either ve a number or a range, for example 1000 or 200-500
    parser.add_argument("--num-samples-per-class", type=int, default=50)
    parser.add_argument("--cfg-scale",  type=float, default=1.5)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--tf32", action="store_true",
                        help="By default, use TF32 matmuls. This massively accelerates sampling on Ampere GPUs.")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args, unparsed = parser.parse_known_args()
    main(args, unparsed)
