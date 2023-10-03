#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
import argparse
import logging
import copy
import math
import os
import random
from pathlib import Path
from typing import Optional
import glob
import shutil
import random
from io import BytesIO

import accelerate
import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate import Accelerator
from accelerate.state import AcceleratorState
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from packaging import version
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.transforms.functional import to_tensor
from tqdm.auto import tqdm
from PIL import Image

import diffusers
from diffusers import DDPMScheduler
from diffusers import UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, randn_tensor

import transformers

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.19.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def make_image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


def center_crop_arr(pil_image, size=512):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - size) // 2
    crop_x = (arr.shape[1] - size) // 2
    return Image.fromarray(arr[crop_y : crop_y + size, crop_x : crop_x + size])


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-model-finetuned",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )
    parser.add_argument(
        "--input_perturbation",
        type=float,
        default=None,
        help="The scale of input perturbation.",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--num_class",
        type=int,
        default=1000,
        help=(
            "The number of classes in the dataset. This is used to create the class embeddings."
        ),
    )
    parser.add_argument(
        "--dropout_prob",
        type=str,
        default=0.1,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--use_ema", action="store_true", help="Whether to use EMA model."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=24,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use."
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer",
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="fp16",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    ),
    parser.add_argument(
        "--amp",
        action="store_true",
    ),
    parser.add_argument(
        "--noise_offset", type=float, default=0, help="The scale of noise offset."
    )
    parser.add_argument(
        "--snr_gamma",
        type=float,
        default=None,
        help="SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0. "
        "More details here: https://arxiv.org/abs/2303.09556.",
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2image-fine-tune",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=1,
        help="Run validation every X epochs.",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


def normalize_to_neg_one_to_one(img):
    return img * 2.0 - 1.0


def unnormalize_to_one_to_one(img):
    return (img + 1.0) / 2.0


def main():
    args = parse_args()

    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )
    if args.amp:
        accelerator.native_amp = True

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler(2000, 0.00085, 0.012, "scaled_linear")

    def deepspeed_zero_init_disabled_context_manager():
        """
        returns either a context list that includes one that will disable zero.Init or an empty context list
        """
        deepspeed_plugin = (
            AcceleratorState().deepspeed_plugin
            if accelerate.state.is_initialized()
            else None
        )
        if deepspeed_plugin is None:
            return []

        return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

    # with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
    #     vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse")
    unet = UNet2DModel(
        args.resolution,
        in_channels=6,
        out_channels=3,
        attention_head_dim=64,
        norm_num_groups=32,
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        up_block_types=(
            "AttnUpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
        block_out_channels=(64, 128, 256, 512, 512, 1024),
    )

    # Create EMA for the unet.
    if args.use_ema:
        ema_unet = copy.deepcopy(unet)

        ema_unet = EMAModel(
            ema_unet.parameters(),
            model_cls=UNet2DModel,
            model_config=ema_unet.config,
        )

    def compute_snr(timesteps):
        """
        Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
        """
        alphas_cumprod = noise_scheduler.alphas_cumprod
        sqrt_alphas_cumprod = alphas_cumprod**0.5
        sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

        # Expand the tensors.
        # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
        sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
            timesteps
        ].float()
        while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
        alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
            device=timesteps.device
        )[timesteps].float()
        while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
        sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

        # Compute SNR.
        snr = (alpha / sigma) ** 2
        return snr

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_unet.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_unet.load_state_dict(load_model.state_dict())
                ema_unet.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate
            * args.gradient_accumulation_steps
            * args.train_batch_size
            * accelerator.num_processes
        )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # data
    class ImageAugment:
        def __init__(self, shrink_size=2, noise_level=1, down_sample_method=None):
            # noise_level (int): 0: no noise; 1: 75-95% quality; 2:50-75%
            if noise_level == 0:
                self.noise_level = [0, 0]
            elif noise_level == 1:
                self.noise_level = [5, 25]
            elif noise_level == 2:
                self.noise_level = [25, 50]
            else:
                raise KeyError("Noise level should be either 0, 1, 2")
            self.shrink_size = shrink_size
            self.down_sample_method = down_sample_method

        def shrink_img(self, hr_img):
            if self.down_sample_method is None:
                resample_method = random.choice(
                    [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
                )
            else:
                resample_method = self.down_sample_method
            img_w, img_h = tuple(map(lambda x: int(x / self.shrink_size), hr_img.size))
            lr_img = hr_img.resize((img_w, img_h), resample_method)
            return lr_img

        def add_jpeg_noise(self, hr_img):
            quality = 100 - round(random.uniform(*self.noise_level))
            lr_img = BytesIO()
            hr_img.save(lr_img, format="JPEG", quality=quality)
            lr_img.seek(0)
            lr_img = Image.open(lr_img)
            return lr_img

        def process(self, hr_patch_pil):
            w, h = hr_patch_pil.size
            lr_patch_pil = self.shrink_img(hr_patch_pil)
            if self.noise_level[1] > 0:
                lr_patch_pil = self.add_jpeg_noise(lr_patch_pil)
            lr_patch_pil = lr_patch_pil.resize((w, h), Image.BICUBIC)
            return lr_patch_pil, hr_patch_pil

        def up_sample(self, img, resample):
            width, height = img.size
            return img.resize(
                (self.shrink_size * width, self.shrink_size * height), resample=resample
            )

    img_augmenter = ImageAugment(4, 1)  # for valid data

    class ImageData(Dataset):
        def __init__(
            self,
            img_folder,
            patch_size=256,
            shrink_size=4,
            noise_level=1,
            down_sample_method=None,
            color_mod="RGB",
            dummy_len=None,
        ):
            self.img_folder = img_folder
            all_img = glob.glob(self.img_folder + "/**", recursive=True)
            self.img = list(
                filter(
                    lambda x: x.endswith("png")
                    or x.endswith("jpg")
                    or x.endswith("jpeg"),
                    all_img,
                )
            )
            self.total_img = len(self.img)
            self.dummy_len = dummy_len if dummy_len is not None else self.total_img
            self.resize = transforms.Resize(patch_size)
            self.hflip = transforms.RandomHorizontalFlip()
            self.color_mod = color_mod
            self.img_augmenter = ImageAugment(
                shrink_size, noise_level, down_sample_method
            )

        def get_img_patches(self, img_file):
            img_pil = Image.open(img_file).convert("RGB")
            img_patch = self.resize(img_pil)
            # img_patch = self.hflip(img_patch)
            lr_hr_patches = self.img_augmenter.process(img_patch)
            return lr_hr_patches

        def __len__(self):
            return self.dummy_len  # len(self.img)

        def __getitem__(self, idx):
            img = self.img[idx]
            patch = self.get_img_patches(img)
            if self.color_mod == "RGB":
                lr_img = patch[0].convert("RGB")
                hr_img = patch[1].convert("RGB")
            elif self.color_mod == "YCbCr":
                lr_img, _, _ = patch[0].convert("YCbCr").split()
                hr_img, _, _ = patch[1].convert("YCbCr").split()
            else:
                raise KeyError("Either RGB or YCbCr")
            return to_tensor(lr_img), to_tensor(hr_img)

    train_dataset = ImageData(
        "data/beans",
        patch_size=args.resolution,
        shrink_size=4,
        noise_level=1,
        down_sample_method=None,
        color_mod="RGB",
        dummy_len=None,
    )
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )
    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    if args.use_ema:
        ema_unet.to(accelerator.device)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae to gpu and cast to weight_dtype
    # vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))
        # tracker_config.pop("validation_prompts")
        accelerator.init_trackers(args.tracker_project_name, tracker_config)

    # Train!
    total_batch_size = (
        args.train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (
                num_update_steps_per_epoch * args.gradient_accumulation_steps
            )

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, args.max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    for epoch in range(first_epoch, args.num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if (
                args.resume_from_checkpoint
                and epoch == first_epoch
                and step < resume_step
            ):
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert images to latent space
                lr_img = batch[0].to(weight_dtype)
                hr_img = batch[1].to(weight_dtype)
                lr_img = normalize_to_neg_one_to_one(lr_img)
                hr_img = normalize_to_neg_one_to_one(hr_img)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(hr_img)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn(
                        (hr_img.shape[0], hr_img.shape[1], 1, 1),
                        device=hr_img.device,
                    )
                if args.input_perturbation:
                    new_noise = noise + args.input_perturbation * torch.randn_like(
                        noise
                    )
                bsz = hr_img.shape[0]
                # Sample a random timestep for each image
                # timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = torch.randint(
                    0,
                    noise_scheduler.config.num_train_timesteps,
                    (bsz,),
                    device=hr_img.device,
                )
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                if args.input_perturbation:
                    noisy_latents = noise_scheduler.add_noise(
                        hr_img, new_noise, timesteps
                    )
                else:
                    noisy_latents = noise_scheduler.add_noise(hr_img, noise, timesteps)

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(
                        f"Unknown prediction type {noise_scheduler.config.prediction_type}"
                    )
                # Predict the noise residual and compute loss
                model_pred = unet(
                    torch.cat([noisy_latents, lr_img], dim=1), timesteps
                ).sample
                if args.snr_gamma is None:
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="mean"
                    )
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(timesteps)
                    mse_loss_weights = (
                        torch.stack(
                            [snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1
                        ).min(dim=1)[0]
                        / snr
                    )
                    # We first calculate the original loss. Then we mean over the non-batch dimensions and
                    # rebalance the sample-wise losses with their respective loss weights.
                    # Finally, we take the mean of the rebalanced loss.
                    loss = F.mse_loss(
                        model_pred.float(), target.float(), reduction="none"
                    )
                    loss = (
                        loss.mean(dim=list(range(1, len(loss.shape))))
                        * mse_loss_weights
                    )
                    loss = loss.mean()

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean()
                train_loss += avg_loss.item() / args.gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_unet.step(unet.parameters())
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [
                                d for d in checkpoints if d.startswith("checkpoint")
                            ]
                            checkpoints = sorted(
                                checkpoints, key=lambda x: int(x.split("-")[1])
                            )

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = (
                                    len(checkpoints) - args.checkpoints_total_limit + 1
                                )
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(
                                    f"removing checkpoints: {', '.join(removing_checkpoints)}"
                                )

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(
                                        args.output_dir, removing_checkpoint
                                    )
                                    shutil.rmtree(removing_checkpoint)
                        save_path = os.path.join(
                            args.output_dir, f"checkpoint-{global_step}"
                        )
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if global_step >= args.max_train_steps:
                break
            # if global_step == 2:
            #     break

        if accelerator.is_main_process:
            if epoch % args.validation_epochs == 0:
                model = accelerator.unwrap_model(unet)
                if args.use_ema:
                    # Store the UNet parameters temporarily and load the EMA parameters to perform inference.
                    ema_unet.store(model.parameters())
                    ema_unet.copy_to(model.parameters())

                generator = torch.Generator(device=unet.device).manual_seed(0)
                num_inference_steps = 2000
                # output_type = "tensor"
                # guidance_scale = 1.0
                noisy_latents = randn_tensor(
                    (3, 3, 256, 256),
                    generator=generator,
                    device=lr_img.device,
                    dtype=lr_img.dtype,
                )
                lr_images = glob.glob("data/valid/*.jpg")
                processd_images = []
                hr_imgs = []
                for lr_image in lr_images:
                    img = Image.open(lr_image)
                    img = img.resize((256, 256))
                    hr_imgs.append(img)
                    img = img_augmenter.process(img)[0]
                    processd_images.append(img)
                lr_img_save = torch.stack([to_tensor(img) for img in processd_images])
                hr_imgs = torch.stack([to_tensor(img) for img in hr_imgs])

                lr_img = normalize_to_neg_one_to_one(lr_img_save)
                lr_img = lr_img.to(
                    device=noisy_latents.device, dtype=noisy_latents.dtype
                )

                noise_scheduler.set_timesteps(num_inference_steps)
                with torch.no_grad():
                    for t in tqdm(noise_scheduler.timesteps):
                        model_output = unet(
                            torch.cat([noisy_latents, lr_img], dim=1),
                            timestep=t.unsqueeze(0).to(lr_img.device),
                        ).sample
                        noisy_latents = noise_scheduler.step(
                            model_output, t, noisy_latents, generator=generator
                        ).prev_sample
                    #     image = 1 / vae.config.scaling_factor * image
                    #     image = image.to(dtype=latents.dtype)
                    #     image = vae.decode(image).sample
                    image = unnormalize_to_one_to_one(noisy_latents).clamp(0, 1).cpu()
                    image = image.to(torch.float32)
                # pipe = MyPipeline(
                #     vae=vae,
                #     unet=unwrap_unet,
                #     scheduler=noise_scheduler,
                # )
                # image_tensor = pipe(
                #     class_labels=class_label,
                #     generator=generator,
                #     num_inference_steps=num_inference_steps,
                #     output_type=output_type,
                #     guidance_scale=guidance_scale,
                # ).images
                output_name = os.path.join(
                    args.output_dir, "images", f"images-{global_step}.jpg"
                )
                if not os.path.exists(os.path.join(args.output_dir, "images")):
                    os.makedirs(os.path.join(args.output_dir, "images"))

                save_image(
                    torch.cat((hr_imgs, lr_img_save, image), dim=0), output_name, nrow=3
                )
                if args.use_ema:
                    # Switch back to the original UNet parameters.
                    ema_unet.restore(unet.parameters())
    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        if args.use_ema:
            ema_unet.copy_to(unet.parameters())
        # save unet
        accelerator.save(unet.state_dict(), os.path.join(args.output_dir, "unet.pt"))

    accelerator.end_training()


if __name__ == "__main__":
    main()
