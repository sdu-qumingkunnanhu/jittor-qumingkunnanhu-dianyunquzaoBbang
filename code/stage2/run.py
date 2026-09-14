import argparse
import os
import random
import sys
import time
from typing import Dict


STAGE_DIR = os.path.dirname(os.path.abspath(__file__))
RUN_ROOT = os.path.dirname(STAGE_DIR)
os.chdir(RUN_ROOT)
if STAGE_DIR not in sys.path:
    sys.path.insert(0, STAGE_DIR)

import jittor as jt

jt.flags.use_cuda = 1

from jittor import optim
import numpy as np
from omegaconf import OmegaConf

from src.data.dataset import DatasetConfig, PCDatasetModule
from src.model.infocd import calc_cd_like_InfoV2
from src.model.ops import from_normal_frame, to_normal_frame
from src.model.parse import get_model
from src.system.spec import resolve_project_output
from src.utils.pointops import nearest_neighbor_distance


def load_config(kind: str, path: str) -> Dict:
    if path.endswith(".yaml"):
        path = path.removesuffix(".yaml")
    path = f"{path}.yaml"
    print(f"\033[92mload {kind} config: {path}\033[0m", flush=True)
    return OmegaConf.to_container(OmegaConf.load(path))  # type: ignore


def resolve_input_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(RUN_ROOT, path))


def seed_all(seed: int):
    jt.set_global_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def hard_tail_loss(pred, clean, ratio):
    if ratio <= 0:
        return jt.array(0.0)
    k = max(1, min(pred.shape[1], int(np.ceil(float(ratio) * pred.shape[1]))))
    pred_to_gt_dist, _ = nearest_neighbor_distance(pred, clean)
    hard_dist2, _ = jt.topk(
        pred_to_gt_dist * pred_to_gt_dist,
        k,
        dim=1,
        largest=True,
        sorted=False,
    )
    return hard_dist2.mean()


def build_model(model_config, transform_config, checkpoint: str):
    model = get_model(model_config=model_config, transform_config=transform_config)
    model.load(checkpoint)
    return model


def build_stage1_and_dataloader(task, model_config, transform_config, stage1_ckpt: str, seed: int):
    data_config = load_config("data", task["components"]["data"])
    train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])

    stage1 = build_model(model_config, transform_config, stage1_ckpt)
    stage1.eval()
    for param in stage1.parameters():
        param.stop_grad()

    dataset_module = PCDatasetModule(
        process_fn=stage1._process_fn,
        train_dataset_config=train_dataset_config,
        train_transform=stage1.get_train_transform(),
        debug=task.get("debug", False),
        seed=seed,
    )
    return stage1, dataset_module


def train_freeze40(task):
    seed = int(task.get("seed", 2025))
    seed_all(seed)

    stage1_ckpt = resolve_input_path(task["stage1_ckpt"])
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"stage1 checkpoint not found: {stage1_ckpt}")

    transform_config = load_config("transform", task["components"]["transform"])
    model_config = load_config("model", task["components"]["model"])
    system_config = load_config("system", task["components"]["system"])
    exp_dir = resolve_project_output(system_config["ckpt_save_dir"])
    ckpt_name = system_config["ckpt_save_name"]
    os.makedirs(exp_dir, exist_ok=True)

    print("[debug] build frozen PGD1 and live dataloader start", flush=True)
    stage1, dataset_module = build_stage1_and_dataloader(
        task=task,
        model_config=model_config,
        transform_config=transform_config,
        stage1_ckpt=stage1_ckpt,
        seed=seed,
    )
    print("[debug] build frozen PGD1 and live dataloader done", flush=True)

    print("[debug] build trainable PGD2 start", flush=True)
    stage2 = build_model(model_config, transform_config, stage1_ckpt)
    stage2.train()
    print("[debug] build trainable PGD2 done", flush=True)

    epochs = int(task.get("epochs", 40))
    lr = float(task.get("lr", 5e-5))
    lambda_hard = float(task.get("lambda_hard", 0.002))
    hard_tail_ratio = float(task.get("hard_tail_ratio", 0.1))
    grad_clip = float(task.get("grad_clip", 1.0))
    log_every = int(task.get("log_every", 50))
    optimizer = optim.Adam(stage2.parameters(), lr=lr)

    print(
        f"[train] no-cache freeze40 epochs={epochs} lr={lr:g} "
        f"lambda_hard={lambda_hard:g} hard_tail_ratio={hard_tail_ratio:g} "
        f"grad_clip={grad_clip:g}",
        flush=True,
    )
    print(
        "[train] PGD1 is frozen; X1 = PGD1(noisy) is recomputed online for every batch",
        flush=True,
    )

    for epoch in range(epochs):
        train_loader = dataset_module.train_dataloader()
        assert train_loader is not None, "train dataloader is None"
        losses = []
        main_losses = []
        hard_losses = []
        started = time.time()
        batch_count = 0

        for batch_idx, batch in enumerate(train_loader):
            if batch_idx == 0:
                print(f"[debug] epoch {epoch + 1} first batch start", flush=True)
            elif log_every > 0 and (batch_idx + 1) % log_every == 0:
                print(
                    f"[debug] epoch {epoch + 1} batch {batch_idx + 1} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )

            noisy = batch["pc_noisy"].reshape(-1, stage1.patch_size, 3)
            clean = batch["pc_clean"].reshape(-1, stage1.patch_size, 3)
            with jt.no_grad():
                x1 = stage1.denoise_langevin_dynamics(noisy)

            x1_aligned, normal_frames = to_normal_frame(x1)
            clean_aligned, _ = to_normal_frame(clean, normal_frames)
            displacement, commitment = stage2.feature_nets(
                x1_aligned,
                calculate_commitment_losses=True,
            )
            prediction_aligned = x1_aligned + displacement
            prediction = from_normal_frame(prediction_aligned, normal_frames)
            main = calc_cd_like_InfoV2(prediction_aligned, clean_aligned)
            hard = hard_tail_loss(prediction, clean, hard_tail_ratio)
            loss = main + commitment + lambda_hard * hard

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(grad_clip)
            optimizer.step()

            losses.append(float(loss.item()))
            main_losses.append(float(main.item()))
            hard_losses.append(float(hard.item()))
            batch_count += 1

        checkpoint = os.path.join(exp_dir, f"{ckpt_name}_{epoch}.pkl")
        stage2.save(checkpoint)
        print(
            f"[train] epoch={epoch + 1}/{epochs} batches={batch_count} "
            f"loss={np.mean(losses):.9f} main={np.mean(main_losses):.9f} "
            f"hard={np.mean(hard_losses):.9f} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        print(f"[checkpoint] saved {checkpoint}", flush=True)

    return checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stage2/configs/task/freeze40")
    args = parser.parse_args()

    task = load_config("task", args.task)
    train_freeze40(task)


if __name__ == "__main__":
    main()
