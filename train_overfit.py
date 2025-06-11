# Copyright (c) Meta Platforms, Inc. and affiliates.
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
# limitations under the License.

"""
# train with a single gpu
python train.py

# train with 8 gpus
torchrun --standalone --nproc_per_node=8 train.py

# train with multi-node multi-gpu, run
sbatch sbatch_run.sh
"""
from calendar import c
import gc
import re
import math
import os
import random
import shutil
import time
from datetime import datetime
from tkinter.tix import MAX

import hydra
import omegaconf

import torch
import torch.distributed as dist
import tqdm
import webdataset as wds
import yaml
from efm3d.aria.tensor_wrapper import custom_collate_fn
from efm3d.dataset.augmentation import ColorJitter, PointDropSimple, PointJitter
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm_train
from efm3d.dataset.vrs_dataset import preprocess
from efm3d.dataset.wds_dataset import get_tar_sample_num
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from itertools import islice, cycle
import wandb
import argparse


# ---------------------------- Configurations ------------------------------
DATA_PATH = "/home/stud/bbo/projects/EFM3D/data/ase_train_10_seq/0"
MAX_LR = 1e-5
MIN_LR = MAX_LR * 0.1
BATCH_SIZE = 2
MAX_EPOCHS = 600
MAX_SAMPLES_PER_EPOCH = 99999
SAVE_EVERY_EPOCHS = 200  # save the model every
LOG_STEP = 5  # print error every

# Wandb configuration
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "643ef7ca035930247e77758b065aef99348b7e22")
ENTITY_NAME = os.environ.get("WANDB_ENTITY", "jie-hu-technical-university-of-munich")

torch.cuda.empty_cache()
if torch.cuda.is_available():
    free_memory = torch.cuda.mem_get_info()[0] / 1024**3  # Free memory in GB
    print(f"Free GPU memory before loading model: {free_memory:.2f} GB")

# ---------------------------- Helpers ------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="EFM3D Training")
    parser.add_argument("--resume_from", type=str, default=None, 
                        help="Path to checkpoint to resume training from")
    return parser.parse_args()

def get_lr(it, warmup_its, max_its, max_lr, min_lr):
    """
    cosine learning rate scheduler, `it` can be either step or epoch.
    """
    # learning rate scheduler
    # linear warmup for warmup_epochs
    if it < warmup_its:
        return max_lr * (it + 1) / warmup_its

    # return min_lr if epoch > max_epochs
    if it > max_its:
        return min_lr

    # cosine annealing
    decay_ratio = (it - warmup_its) / (max_its - warmup_its)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1.0 -> 0.0
    return min_lr + coeff * (max_lr - min_lr)


def get_dataloader(
    data_path,
    batch_size,
    world_size,
    max_samples_per_epoch,
    epoch_sample_ratio=1.0,
    tar_yaml="train_tars.yaml",
):
    assert (
        epoch_sample_ratio > 0 and epoch_sample_ratio <= 1.0
    ), f"{epoch_sample_ratio} is the ratio ([0, 1]) of samples used in each epoch"

    # # Directly use the 4 shards from the sequence 0
    tar_list = ["shards-0000.tar", "shards-0001.tar", "shards-0002.tar", "shards-0003.tar"]
    # tar_list = ["shards-0000.tar"]
    tar_list = [os.path.join(data_path, tar_name) for tar_name in tar_list]
    
    # check existence
    for tar in tar_list:
        assert os.path.exists(tar), f"{tar} not exists"
    random.shuffle(tar_list)
    dataset = load_atek_wds_dataset_as_efm_train(
        urls=tar_list,
        atek_to_efm_taxonomy_mapping_file=f"{os.path.dirname(__file__)}/efm3d/config/taxonomy/atek_to_efm.csv",
        batch_size=batch_size,
        collation_fn=custom_collate_fn,
    )
    ####
    samples_per_tar = get_tar_sample_num(tar_list[0])
    dataset_size = len(tar_list) * samples_per_tar
    dataset_size = min(dataset_size, max_samples_per_epoch)
    dataset_size = int(dataset_size * epoch_sample_ratio)

    batches_per_epoch = int(dataset_size // (batch_size * world_size))
    dataloader = wds.WebLoader(
        dataset,
        num_workers=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        batch_size=None,
        shuffle=False,
    )
    # total_samples = int(max_samples_per_epoch * epoch_sample_ratio)
    # num_batches = total_samples // (batch_size * world_size)
    dataloader = dataloader.with_epoch(batches_per_epoch).with_length(batches_per_epoch)
    print(f"Overfitting setup: forcing {batches_per_epoch} batches per epoch, batch_size={batch_size}")

    return dataloader


ddp = int(os.environ.get("RANK", -1)) != -1
if ddp:
    assert torch.cuda.is_available()
    init_process_group("nccl")
    DDP_RANK = int(os.environ["RANK"])
    DDP_LOCAL_RANK = int(os.environ["LOCAL_RANK"])
    DDP_WORLD_SIZE = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{DDP_LOCAL_RANK}"
    print(f"==> setting device to {device}")
    torch.cuda.set_device(device)
    master_process = DDP_RANK == 0
else:
    DDP_RANK = 0
    DDP_LOCAL_RANK = 0
    DDP_WORLD_SIZE = 1
    master_process = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
# ---------------------------- Load Model & Optimizer ------------------------------
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    
args = parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"
model_config = omegaconf.OmegaConf.load("efm3d/config/evl_train.yaml")
model = hydra.utils.instantiate(model_config)
model = model
model.to(device)
if ddp:
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
raw_model = model.module if ddp else model
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

starting_epoch = 0
step = 0
val_step = 0
# Checkpoint loading
if args.resume_from and os.path.exists(args.resume_from):
    print(f"Loading checkpoint from {args.resume_from}")
    checkpoint = torch.load(args.resume_from, map_location=device)
    resume_successful = False
    # 1. Prefer explicit fields stored in the ckpt dict
    if {'epoch', 'step', 'val_step'}.issubset(checkpoint.keys()):
        starting_epoch = checkpoint['epoch'] + 1
        step          = checkpoint['step']
        val_step      = checkpoint['val_step']
        resume_successful = True
    # 2. Otherwise fall back to regex parsing of the filename
    else:
        # Accept patterns like “…e34s6160.pth” or “…e34s6160vs4500.pth”
        m = re.search(r"e(\d+)s(\d+)(?:vs(\d+))?", os.path.basename(args.resume_from))
        if m:
            starting_epoch = int(m.group(1)) + 1
            step          = int(m.group(2))
            val_step      = int(m.group(3) or 0)
            resume_successful = True
    # 3. Load the weights / optimizer if we know what to do
    if resume_successful:
        print(f"---Resuming from epoch {starting_epoch}, step {step}, val_step {val_step}---")
        # handle both “state_dict” wrapper and raw pytorch state-dicts
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        raw_model.load_state_dict(state_dict, strict=False)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
    else:
        print("Could not determine resume point - starting fresh from epoch 0")

# ---------------------------- Dataloaders & Augmentation ------------------------------
train_dataloader = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=MAX_SAMPLES_PER_EPOCH,
    tar_yaml="train_tars.yaml",
)
val_dataloader = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=4,
    # tar_yaml="val_tars.yaml",
    tar_yaml="train_tars.yaml",  # Use the same tar files for validation
)

color_jitter = ColorJitter(
    brightness=0.5,
    contrast=0.3,
    saturation=0.3,
    hue=0.05,
    sharpness=2.0,
    snippet_jitter=True,
)
point_drop = PointDropSimple(max_dropout_rate=0.8)
point_jitter = PointJitter(depth_std_scale_min=1.0, depth_std_scale_max=6.0)
# augmentations = [color_jitter, point_drop, point_jitter]
augmentations = [] # No augmentations

# ---------------------------- Wandb Setup ------------------------------
if master_process:
    exp_name = f"efm3d_train_seq0_overfit_b{BATCH_SIZE}g{DDP_WORLD_SIZE}e{MAX_EPOCHS}lr{str(MAX_LR)}_{datetime.fromtimestamp(time.time()).strftime('%y-%m-%d-%H-%M-%S')}"
    # log_dir = os.path.join("wandb_logs", exp_name)
    log_dir = "wandb_logs/overfit_seq0"
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize wandb
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="efm3d-training-overfit",
        entity=ENTITY_NAME,
        name=exp_name,
        config={
            "batch_size": BATCH_SIZE,
            "world_size": DDP_WORLD_SIZE,
            "epochs": MAX_EPOCHS,
            "max_lr": MAX_LR,
            # "min_lr": MIN_LR,
            "data_path": DATA_PATH,
            "max_samples_per_epoch": MAX_SAMPLES_PER_EPOCH,
            "save_every_epochs": SAVE_EVERY_EPOCHS,
        },
        dir=log_dir
    )

# # print the sample id in the first 3 batches
# if master_process:
#     print("Sample IDs in the first 3 batches:")
#     for i, batch in enumerate(islice(train_dataloader, 3)):
#         if "rgb/frame_id_in_sequence" in batch:
#             try:
#                 print("rgb/frame_id_in_sequence", batch["rgb/frame_id_in_sequence"])
#             except Exception as e:
#                 print(f"Error printing frame_id_in_sequence: {e}")
#         else:
#             print("⚠️ No rgb/frame_id_in_sequence in batch")
#         if "rgb/img/time_ns" in batch:
#             try:
#                 print("rgb/img/time_ns", batch["rgb/img/time_ns"])
#             except Exception as e:
#                 print(f"Error printing time_ns: {e}")
            

# ---------------------------- Training Loop ------------------------------
# main loop
for epoch in range(starting_epoch, MAX_EPOCHS):
    # Log GPU memory usage
    if master_process:
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"Epoch {epoch} | GPU Memory: Allocated {allocated:.2f}GB, Reserved {reserved:.2f}GB")
    # Reset epoch losses
    train_loss_sum, train_count = 0.0, 0
    val_loss_sum, val_count = 0.0, 0
    # train ================================================
    model.train()
    for batch in tqdm.tqdm(train_dataloader):
        start = time.time()
        optimizer.zero_grad()
        batch = preprocess(batch, device, aug_funcs=augmentations)
        output = model(batch)
        losses, total_loss = raw_model.compute_losses(output, batch)
        total_loss.backward()

        # epoch-based lr scheduler
        # lr = get_lr(
        #     epoch, warmup_its=5, max_its=MAX_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR
        # )
        lr = MAX_LR  # constant learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        train_loss_sum += total_loss.item()
        train_count += 1
        max_norm = 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        time_per_it = time.time() - start

        if master_process and step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{step} | loss {total_loss.item():.03f} | lr {lr:.06f} | norm {norm} | time {time_per_it:.02f}s/it"
            )

            # log training
            log_dict = {
                "train/loss": total_loss.item(),
                "train/lr": lr,
                "train/iter_sec": time_per_it,
            }
            
            # Add all individual loss components
            for stream in losses:
                for loss_name in losses[stream]:
                    log_dict[f"train/loss/{stream}/{loss_name}"] = losses[stream][loss_name].item()
            
            wandb.log(log_dict, step=step)

            # log images (log every `10xlog_step` since writing video is slow)
            if step % (480 * LOG_STEP) == 0:
                try:
                    # Try using the original visualization but catch any OpenGL errors
                    imgs = raw_model.log_single(batch, output, batch_idx=0)
                    for k, v in imgs.items():
                        # Convert numpy array to format suitable for wandb
                        if v.ndim == 4:  # video
                            wandb.log({f"train/video/{k}": wandb.Video(
                                v.transpose((0, 3, 1, 2)), 
                                fps=10, 
                                format="mp4"
                            )}, step=step)
                        elif v.ndim == 3:  # image
                            wandb.log({f"train/image/{k}": wandb.Image(v)}, step=step)
                except Exception as e:
                    print(f"OpenGL visualization failed: {e}")
        step += 1

    # # val =========================================
    model.eval()
    raw_model.reset_metrics()  # reset metrics for each epoch
    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            start = time.time()
            batch = preprocess(batch, device, aug_funcs=augmentations)
            output = model(batch)
            losses, total_loss = raw_model.compute_losses(output, batch)
            # update metrics
            raw_model.update_metrics(output, batch)
            if ddp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            val_loss_sum += total_loss.item()
            val_count += 1
            time_per_it = time.time() - start

        if master_process and val_step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{val_step} | loss {total_loss.item():.03f} | time {time_per_it:.02f}s/it"
            )

            # log val
            if val_step % LOG_STEP == 0:
                log_dict = {
                    "val/loss": total_loss.item(),
                    "val/iter_sec": time_per_it,
                    "epoch": epoch
                }
                
                # Add all individual loss components
                for stream in losses:
                    for loss_name in losses[stream]:
                        log_dict[f"val/loss/{stream}/{loss_name}"] = losses[stream][loss_name].item()
                
                wandb.log(log_dict, step=step)

            # log images
            if val_step % (160 * LOG_STEP) == 0:
                try:
                    # Try using the original visualization but catch any OpenGL errors
                    imgs = raw_model.log_single(batch, output, batch_idx=0)
                    for k, v in imgs.items():
                        # Convert numpy array to format suitable for wandb
                        if v.ndim == 4:  # video
                            wandb.log({f"val/video/{k}": wandb.Video(
                                v.transpose((0, 3, 1, 2)),
                                fps=10,
                                format="mp4"
                            )}, step=step)
                        elif v.ndim == 3:  # image
                            wandb.log({f"val/image/{k}": wandb.Image(v)}, step=step)
                except Exception as e:
                    print(f"OpenGL visualization failed: {e}")
        step += 1
        val_step += 1
    # record epoch loss at end of epoch
    if master_process:
        epoch_summary = {
            "epoch": epoch,
            "epoch_train_loss": train_loss_sum / train_count if train_count > 0 else 0.0,
            "epoch_val_loss": val_loss_sum / val_count if val_count > 0 else 0.0
        }
        # pull higer-level metrics from the model
        epoch_summary.update({f"val/{k}": v for k, v in raw_model.compute_metrics()["rgb"]["metrics"].items()})
        wandb.log(epoch_summary, step=step)

    # save model
    if master_process and (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
        ckpt_path = os.path.join(
            log_dir, f"model_overfit_e{epoch}s{step}vs{val_step}_l{total_loss.item():.02f}.pth"
        )
        last_ckpt_path = os.path.join(log_dir, "last.pth")
        torch.save(
            {
             "state_dict": raw_model.state_dict(), 
             "optimizer": optimizer.state_dict(),
             "epoch": epoch,
             "step": step,
             "val_step": val_step
             },
            ckpt_path,
        )
        shutil.copy(ckpt_path, last_ckpt_path)
        
        # Log model checkpoint to wandb
        wandb.save(ckpt_path)
        wandb.run.summary.update({
            "last_epoch": epoch,
            "last_loss": total_loss.item(),
            "last_step": step
        })
    # cleanup
    if master_process:
        gc.collect()
        torch.cuda.empty_cache()
        

if master_process:
    wandb.finish()
if ddp:
    destroy_process_group()