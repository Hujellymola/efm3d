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

import json
import math
import os
import random
import shutil
import time
from collections import Counter
from datetime import datetime

import hydra
import omegaconf
import torch
import torch.distributed as dist
import tqdm
import wandb
import webdataset as wds
import yaml
from efm3d.aria.tensor_wrapper import custom_collate_fn
from efm3d.dataset.augmentation import ColorJitter, PointDropSimple, PointJitter
from efm3d.dataset.efm_model_adaptor import load_atek_wds_dataset_as_efm_train
from efm3d.dataset.vrs_dataset import preprocess
from efm3d.dataset.wds_dataset import get_tar_sample_num
from efm3d.inference.fuse import VolumetricFusion
from efm3d.inference.model import EfmInference
from efm3d.inference.pipeline import create_streamer
from efm3d.inference.viz import generate_video
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


DATA_PATH = "./data/ase_train"
MAX_LR = 2e-4
MIN_LR = MAX_LR * 0.1
BATCH_SIZE = 2
MAX_EPOCHS = 20
MAX_SAMPLES_PER_EPOCH = None  # Set to an int for smoke tests.
SAVE_EVERY_EPOCHS = 5  # save the model every
LOG_STEP = 5  # print error every

USE_WANDB = True
WANDB_PROJECT = "efm3d"

SUBSET_TRAIN_SEQ_IDS = ["0", "90", "91", "92", "94", "95", "96", "97", "98"]
SUBSET_EVAL_SEQ_IDS = ["9"]

MINI_EVAL_EVERY_EPOCHS = 1
MINI_EVAL_MAX_SNIPS = 20
MINI_EVAL_SNIP_STRIDE = 0.5
MINI_EVAL_VOXEL_RES = 0.08


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


def get_sequence_id(tar_name):
    return tar_name.split("/")[0]


def load_filtered_tar_list(data_path, tar_yaml, seq_ids, verbose=False):
    tar_yaml_path = os.path.join(data_path, tar_yaml)
    with open(tar_yaml_path, "r") as f:
        tar_names = yaml.safe_load(f)["tars"]

    seq_ids = [str(seq_id) for seq_id in seq_ids]
    seq_id_set = set(seq_ids)
    tar_names = [
        tar_name for tar_name in tar_names if get_sequence_id(tar_name) in seq_id_set
    ]
    shard_counts = Counter(get_sequence_id(tar_name) for tar_name in tar_names)
    missing = [seq_id for seq_id in seq_ids if shard_counts[seq_id] == 0]
    assert len(missing) == 0, (
        f"No tars found for sequence ids {missing} in {tar_yaml_path}"
    )

    ordered_shard_counts = {seq_id: shard_counts[seq_id] for seq_id in seq_ids}
    tar_list = [os.path.join(data_path, tar_name) for tar_name in tar_names]
    for tar in tar_list:
        assert os.path.exists(tar), f"{tar} not exists"

    samples_per_tar = get_tar_sample_num(tar_list[0])
    split_info = {
        "tar_yaml": tar_yaml,
        "seq_ids": seq_ids,
        "num_tars": len(tar_list),
        "samples_per_tar": samples_per_tar,
        "estimated_num_samples": len(tar_list) * samples_per_tar,
        "sequence_shard_counts": ordered_shard_counts,
    }
    if verbose:
        print(
            f"==> {tar_yaml} selected seq ids {seq_ids}; "
            f"shards {ordered_shard_counts}; "
            f"estimated samples {split_info['estimated_num_samples']}"
        )
    return tar_list, split_info


def get_dataloader(
    data_path,
    batch_size,
    world_size,
    max_samples_per_epoch,
    epoch_sample_ratio=1.0,
    tar_yaml="all_tars.yaml",
    seq_ids=None,
    verbose=False,
):
    assert epoch_sample_ratio > 0 and epoch_sample_ratio <= 1.0, (
        f"{epoch_sample_ratio} is the ratio ([0, 1]) of samples used in each epoch"
    )

    assert seq_ids is not None and len(seq_ids) > 0, "seq_ids must be specified"
    tar_list, split_info = load_filtered_tar_list(
        data_path,
        tar_yaml,
        seq_ids,
        verbose=verbose,
    )
    random.shuffle(tar_list)
    dataset = load_atek_wds_dataset_as_efm_train(
        urls=tar_list,
        atek_to_efm_taxonomy_mapping_file=f"{os.path.dirname(__file__)}/efm3d/config/taxonomy/atek_to_efm.csv",
        batch_size=batch_size,
        collation_fn=custom_collate_fn,
    )

    dataset_size = split_info["estimated_num_samples"]
    if max_samples_per_epoch is not None:
        dataset_size = min(dataset_size, max_samples_per_epoch)
    dataset_size = int(dataset_size * epoch_sample_ratio)

    batches_per_epoch = max(1, int(dataset_size // (batch_size * world_size)))
    split_info["samples_per_epoch"] = dataset_size
    split_info["batches_per_epoch"] = batches_per_epoch
    dataloader = wds.WebLoader(
        dataset,
        num_workers=batch_size,
        pin_memory=True,
        prefetch_factor=2,
        batch_size=None,
        shuffle=False,
    )
    dataloader = dataloader.with_epoch(batches_per_epoch)
    dataloader = dataloader.with_length(batches_per_epoch)

    return dataloader, split_info


def tensor_to_float(value):
    if isinstance(value, torch.Tensor):
        return value.detach().float().item()
    return float(value)


def flatten_losses(prefix, losses):
    flat = {}
    for stream in losses:
        for loss_name in losses[stream]:
            flat[f"{prefix}/loss/{stream}/{loss_name}"] = tensor_to_float(
                losses[stream][loss_name]
            )
    return flat


def wandb_video(frames, fps=10):
    return wandb.Video(frames.astype("uint8"), fps=fps, format="mp4")


def log_wandb_scalars(prefix, losses, total_loss, step_value, epoch, extra=None):
    if not USE_WANDB:
        return
    payload = {
        f"{prefix}/loss": tensor_to_float(total_loss),
        f"{prefix}/step": step_value,
        "epoch": epoch,
    }
    if extra is not None:
        payload.update(extra)
    payload.update(flatten_losses(prefix, losses))
    wandb.log(payload)


def log_wandb_videos(prefix, imgs, step_value, epoch):
    if not USE_WANDB:
        return
    payload = {
        f"{prefix}/step": step_value,
        "epoch": epoch,
    }
    for key, frames in imgs.items():
        payload[f"{prefix}/video/{key}"] = wandb_video(frames, fps=10)
    wandb.log(payload)


def init_wandb(exp_name, model_config, train_info, val_info):
    if not USE_WANDB:
        return None
    run = wandb.init(
        project=WANDB_PROJECT,
        name=exp_name,
        config={
            "data_path": DATA_PATH,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "max_lr": MAX_LR,
            "min_lr": MIN_LR,
            "max_samples_per_epoch": MAX_SAMPLES_PER_EPOCH,
            "save_every_epochs": SAVE_EVERY_EPOCHS,
            "log_step": LOG_STEP,
            "subset_train_seq_ids": SUBSET_TRAIN_SEQ_IDS,
            "subset_eval_seq_ids": SUBSET_EVAL_SEQ_IDS,
            "mini_eval_every_epochs": MINI_EVAL_EVERY_EPOCHS,
            "mini_eval_max_snips": MINI_EVAL_MAX_SNIPS,
            "mini_eval_snip_stride": MINI_EVAL_SNIP_STRIDE,
            "mini_eval_voxel_res": MINI_EVAL_VOXEL_RES,
            "train_split": train_info,
            "val_split": val_info,
            "model": omegaconf.OmegaConf.to_container(model_config, resolve=True),
        },
    )
    wandb.define_metric("epoch")
    wandb.define_metric("train/step")
    wandb.define_metric("val/step")
    wandb.define_metric("mini_eval/epoch")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("val/*", step_metric="val/step")
    wandb.define_metric("mini_eval/*", step_metric="mini_eval/epoch")
    return run


def maybe_run_mini_eval(raw_model, log_dir, epoch, device, wandb_run=None):
    if MINI_EVAL_EVERY_EPOCHS <= 0:
        return
    if (epoch + 1) % MINI_EVAL_EVERY_EPOCHS != 0:
        return

    was_training = raw_model.training
    raw_model.eval()
    eval_root = os.path.join(log_dir, "mini_eval", f"epoch_{epoch:04d}")

    for seq_id in SUBSET_EVAL_SEQ_IDS:
        seq_path = os.path.join(DATA_PATH, seq_id)
        output_dir = os.path.join(eval_root, seq_id)
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        print(
            f"==> mini-eval epoch {epoch} seq {seq_id}: "
            f"max_snips={MINI_EVAL_MAX_SNIPS}, stride={MINI_EVAL_SNIP_STRIDE}, "
            f"voxel_res={MINI_EVAL_VOXEL_RES}"
        )

        payload = {
            "epoch": epoch,
            "mini_eval/epoch": epoch,
            f"mini_eval/{seq_id}/failed": 0,
        }
        try:
            streamer = create_streamer(
                seq_path,
                snippet_length_s=1.0,
                stride_length_s=MINI_EVAL_SNIP_STRIDE,
                max_snip=MINI_EVAL_MAX_SNIPS,
            )
            efm_inf = EfmInference(
                streamer,
                raw_model,
                output_dir,
                device=device,
                zip=False,
                obb_only=False,
            )
            efm_inf.run()
            del efm_inf

            try:
                from efm3d.inference.track import track_obbs

                track_obbs(output_dir)
            except Exception as err:
                print(f"Skip mini-eval tracking due to error: {err}")

            metrics = {}
            pred_csv = os.path.join(output_dir, "tracked_scene_obbs.csv")
            gt_csv = os.path.join(output_dir, "gt_scene_obbs.csv")
            if os.path.exists(pred_csv) and os.path.exists(gt_csv):
                try:
                    from efm3d.inference.eval import evaluate_obb_csv

                    metrics.update(
                        evaluate_obb_csv(pred_csv=pred_csv, gt_csv=gt_csv, iou=0.2)
                    )
                except Exception as err:
                    print(f"Skip mini-eval OBB metrics due to error: {err}")

            vol_fusion = VolumetricFusion(
                output_dir,
                voxel_res=MINI_EVAL_VOXEL_RES,
                device=device,
            )
            vol_fusion.run()
            fused_mesh = vol_fusion.get_trimesh()
            fused_mesh_path = os.path.join(output_dir, "fused_mesh.ply")
            has_fused_mesh = (
                fused_mesh.vertices.shape[0] > 0 and fused_mesh.faces.shape[0] > 0
            )
            if has_fused_mesh:
                fused_mesh.export(fused_mesh_path)
                print(
                    "==> mini-eval fused mesh saved to "
                    f"{os.path.abspath(fused_mesh_path)}"
                )
            else:
                print("==> mini-eval fusion produced an empty mesh")

            metrics_path = os.path.join(output_dir, "metrics.json")
            if len(metrics) > 0:
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=2, sort_keys=True)
                payload.update(
                    {
                        f"mini_eval/{seq_id}/{key}": value
                        for key, value in metrics.items()
                    }
                )

            viz_snips = max(
                1, math.ceil((MINI_EVAL_MAX_SNIPS - 1) * MINI_EVAL_SNIP_STRIDE)
            )
            viz_streamer = create_streamer(
                seq_path,
                snippet_length_s=1.0,
                stride_length_s=1.0,
                max_snip=viz_snips,
            )
            vol_fusion.reinit()
            video_path = generate_video(
                viz_streamer,
                output_dir=output_dir,
                vol_fusion=vol_fusion,
                stride_s=MINI_EVAL_SNIP_STRIDE,
            )

            payload.update(
                {
                    f"mini_eval/{seq_id}/has_fused_mesh": int(has_fused_mesh),
                    f"mini_eval/{seq_id}/mesh_vertices": int(
                        fused_mesh.vertices.shape[0]
                    ),
                    f"mini_eval/{seq_id}/mesh_faces": int(fused_mesh.faces.shape[0]),
                }
            )
            if wandb_run is not None and os.path.exists(video_path):
                payload[f"mini_eval/{seq_id}/video"] = wandb.Video(
                    video_path,
                    format="mp4",
                )
            if wandb_run is not None:
                wandb.log(payload)
        except Exception as err:
            print(f"Mini-eval failed for seq {seq_id}: {err}")
            if wandb_run is not None:
                payload[f"mini_eval/{seq_id}/failed"] = 1
                payload[f"mini_eval/{seq_id}/error"] = str(err)
                wandb.log(payload)
            raise
        finally:
            shutil.rmtree(os.path.join(output_dir, "per_snip"), ignore_errors=True)

    if was_training:
        raw_model.train()


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

torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_config = omegaconf.OmegaConf.load("efm3d/config/evl_train.yaml")
model = hydra.utils.instantiate(model_config)
model = model
model.to(device)
if ddp:
    model = DDP(model, device_ids=[DDP_LOCAL_RANK])
raw_model = model.module if ddp else model

train_dataloader, train_info = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=MAX_SAMPLES_PER_EPOCH,
    tar_yaml="all_tars.yaml",
    seq_ids=SUBSET_TRAIN_SEQ_IDS,
    verbose=master_process,
)
val_dataloader, val_info = get_dataloader(
    DATA_PATH,
    BATCH_SIZE,
    DDP_WORLD_SIZE,
    max_samples_per_epoch=MAX_SAMPLES_PER_EPOCH,
    tar_yaml="all_tars.yaml",
    seq_ids=SUBSET_EVAL_SEQ_IDS,
    verbose=master_process,
)
optimizer = torch.optim.AdamW(model.parameters(), lr=MAX_LR)

wandb_run = None
if master_process:
    exp_name = f"efm3d_train_b{BATCH_SIZE}g{DDP_WORLD_SIZE}e{MAX_EPOCHS}lr{str(MAX_LR)}_{datetime.fromtimestamp(time.time()).strftime('%y-%m-%d-%H-%M-%S')}"
    log_dir = os.path.join("tb_logs", exp_name)
    writer = SummaryWriter(log_dir=log_dir)
    wandb_run = init_wandb(exp_name, model_config, train_info, val_info)

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
augmentations = [color_jitter, point_drop, point_jitter]

step = 0
val_step = 0
# main loop
for epoch in range(MAX_EPOCHS):
    # train
    model.train()
    for batch in tqdm.tqdm(train_dataloader):
        start = time.time()
        optimizer.zero_grad()

        batch = preprocess(batch, device, aug_funcs=augmentations)
        output = model(batch)
        losses, total_loss = raw_model.compute_losses(output, batch)

        total_loss.backward()

        # epoch-based lr scheduler
        lr = get_lr(
            epoch, warmup_its=5, max_its=MAX_EPOCHS, max_lr=MAX_LR, min_lr=MIN_LR
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if ddp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
        max_norm = 1.0
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

        optimizer.step()
        time_per_it = time.time() - start

        if master_process and step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{step} | loss {total_loss.item():.03f} | lr {lr:.06f} | norm {norm} | time {time_per_it:.02f}s/it"
            )

            # log training
            writer.add_scalar("train/loss", total_loss.item(), step)
            for stream in losses:
                for loss_name in losses[stream]:
                    writer.add_scalar(
                        f"train/loss/{stream}/{loss_name}",
                        losses[stream][loss_name].item(),
                        step,
                    )
            writer.add_scalar("train/lr", lr, step)
            writer.add_scalar("train/iter_sec", time_per_it, step)
            log_wandb_scalars(
                "train",
                losses,
                total_loss,
                step,
                epoch,
                extra={
                    "train/lr": lr,
                    "train/grad_norm": tensor_to_float(norm),
                    "train/iter_sec": time_per_it,
                },
            )

            # log images (log every `10xlog_step` since writing video is slow)
            if step % (10 * LOG_STEP) == 0:
                imgs = raw_model.log_single(batch, output, batch_idx=0)
                for k, v in imgs.items():
                    vid = torch.tensor(v.transpose((0, 3, 1, 2))).unsqueeze(0)
                    writer.add_video(f"train/{k}", vid, global_step=step, fps=10)
                log_wandb_videos("train", imgs, step, epoch)
        step += 1

    # val
    model.eval()
    for batch in tqdm.tqdm(val_dataloader):
        with torch.no_grad():
            start = time.time()
            batch = preprocess(batch, device, aug_funcs=augmentations)
            output = model(batch)
            losses, total_loss = raw_model.compute_losses(output, batch)
            if ddp:
                dist.all_reduce(total_loss, op=dist.ReduceOp.AVG)
            time_per_it = time.time() - start

        if master_process and val_step % LOG_STEP == 0:
            print(
                f"E:s-{epoch}:{val_step} | loss {total_loss.item():.03f} | time {time_per_it:.02f}s/it"
            )

            # log val
            if val_step % LOG_STEP == 0:
                writer.add_scalar("val/loss", total_loss.item(), val_step)
                for stream in losses:
                    for loss_name in losses[stream]:
                        writer.add_scalar(
                            f"val/loss/{stream}/{loss_name}",
                            losses[stream][loss_name].item(),
                            val_step,
                        )
                writer.add_scalar("val/iter_sec", time_per_it, val_step)
                log_wandb_scalars(
                    "val",
                    losses,
                    total_loss,
                    val_step,
                    epoch,
                    extra={"val/iter_sec": time_per_it},
                )

            # log images
            if val_step % (10 * LOG_STEP) == 0:
                imgs = raw_model.log_single(batch, output, batch_idx=0)
                for k, v in imgs.items():
                    vid = torch.tensor(v.transpose((0, 3, 1, 2))).unsqueeze(0)
                    writer.add_video(f"val/{k}", vid, global_step=val_step, fps=10)
                log_wandb_videos("val", imgs, val_step, epoch)
        val_step += 1

    # save model
    if master_process and (epoch + 1) % SAVE_EVERY_EPOCHS == 0:
        ckpt_path = os.path.join(
            log_dir, f"model_e{epoch}s{step}_l{total_loss.item():.02f}.pth"
        )
        last_ckpt_path = os.path.join(log_dir, "last.pth")
        torch.save(
            {"state_dict": raw_model.state_dict(), "optimizer": optimizer.state_dict()},
            ckpt_path,
        )
        shutil.copy(ckpt_path, last_ckpt_path)

    if master_process:
        maybe_run_mini_eval(raw_model, log_dir, epoch, device, wandb_run)
    if ddp:
        dist.barrier()

if master_process:
    writer.close()
    if wandb_run is not None:
        wandb.finish()
if ddp:
    destroy_process_group()
