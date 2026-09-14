import argparse
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple


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
from src.model.ops import from_normal_frame, patch_based_denoise, to_normal_frame
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


def hard_tail_loss(pred, clean, ratio: float):
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


def save_stage_pair(stage1, stage2, out_path: str):
    merged = {f"pgd1.{key}": value for key, value in stage1.state_dict().items()}
    merged.update({f"pgd2.{key}": value for key, value in stage2.state_dict().items()})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    jt.save(merged, out_path)


def split_stage_pair(path: str) -> Tuple[Dict, Dict]:
    params = jt.load(path)
    pgd1 = {}
    pgd2 = {}
    for key, value in params.items():
        key = str(key)
        if key.startswith("pgd1."):
            pgd1[key[len("pgd1.") :]] = value
        elif key.startswith("pgd2."):
            pgd2[key[len("pgd2.") :]] = value
    if not pgd1 or not pgd2:
        raise ValueError(f"joint checkpoint must contain pgd1.* and pgd2.* keys: {path}")
    return pgd1, pgd2


def load_stage_pair(stage1, stage2, checkpoint: str):
    pgd1, pgd2 = split_stage_pair(checkpoint)
    stage1.load_parameters(pgd1)
    stage2.load_parameters(pgd2)


class CascadePredictor:
    def __init__(self, stage1, stage2):
        self.stage1 = stage1
        self.stage2 = stage2
        self.patch_size = stage1.patch_size
        self.seed_k = stage1.seed_k
        self.seed_k_alpha = stage1.seed_k_alpha
        self.niters = stage1.niters

    def eval(self):
        self.stage1.eval()
        self.stage2.eval()
        return self

    def set_predict(self, is_predict: bool):
        self.stage1.set_predict(is_predict)
        self.stage2.set_predict(is_predict)

    def denoise_langevin_dynamics(self, pc_noisy):
        with jt.no_grad():
            disp1, _ = self.stage1.feature_nets(pc_noisy, calculate_commitment_losses=False)
            x1 = pc_noisy + disp1
            x1_aligned, normal_frames = to_normal_frame(x1)
            disp2, _ = self.stage2.feature_nets(x1_aligned, calculate_commitment_losses=False)
            x2_aligned = x1_aligned + disp2
        return from_normal_frame(x2_aligned, normal_frames)

    @jt.no_grad()
    def predict_step(self, batch: Dict):
        results = []
        for pc_noisy in batch["pc_noisy"]:
            pc_next = pc_noisy
            for _ in range(self.niters):
                pc_next = patch_based_denoise(
                    model=self,
                    pcl_noisy=pc_next,
                    patch_size=self.patch_size,
                    seed_k=self.seed_k,
                    seed_k_alpha=self.seed_k_alpha,
                )
            results.append({"pc_denoised": pc_next.detach().numpy().astype(np.float32)})
        return results


def latest_joint_checkpoint(exp_dir: str, ckpt_name: str) -> Tuple[Optional[str], int]:
    if not os.path.isdir(exp_dir):
        return None, 0
    prefix = f"{ckpt_name}_joint_"
    best_epoch = 0
    best_path = None
    for name in os.listdir(exp_dir):
        if not name.startswith(prefix) or not name.endswith(".pkl"):
            continue
        text = name[len(prefix) : -len(".pkl")]
        if not text.isdigit():
            continue
        epoch = int(text)
        if epoch > best_epoch:
            best_epoch = epoch
            best_path = os.path.join(exp_dir, name)
    return best_path, best_epoch


def build_train_dataset_module(task, data_config, stage1):
    train_dataset_config = DatasetConfig.parse(**data_config["train_dataset"])
    train_dataset_config.batch_size = int(task.get("joint_batch_size", train_dataset_config.batch_size))
    train_dataset_config.rebuild_each_epoch = True
    print(
        f"[data] batch_size={train_dataset_config.batch_size} "
        f"num_workers={train_dataset_config.num_workers} rebuild_each_epoch=True",
        flush=True,
    )
    return PCDatasetModule(
        process_fn=stage1._process_fn,
        train_dataset_config=train_dataset_config,
        train_transform=stage1.get_train_transform(),
        debug=task.get("debug", False),
        seed=int(task.get("seed", 2025)),
    )


def asset_rel_dir(asset, dataset_root: str) -> str:
    assert asset.path is not None, "asset path is None"
    path = Path(asset.path)
    if not path.is_absolute():
        path = Path(RUN_ROOT) / path
    path = path.resolve()
    root = Path(dataset_root).resolve()
    try:
        rel_file = path.relative_to(root)
        return str(rel_file.parent).replace("\\", "/")
    except ValueError:
        relpath = None
        if asset.meta is not None:
            relpath = asset.meta.get("relpath")
        if relpath is not None:
            return str(Path(relpath)).replace("\\", "/")
        cls = asset.cls or "predict"
        return f"{cls}/{path.parent.name}"


def write_predictions(batch, prediction, output_root: str, dataset_root: str, pred_filename: str):
    for i, asset in enumerate(batch["asset"]):
        out_dir = os.path.join(output_root, asset_rel_dir(asset, dataset_root))
        os.makedirs(out_dir, exist_ok=True)
        denoised = prediction[i]["pc_denoised"]
        if not isinstance(denoised, np.ndarray):
            denoised = denoised.numpy()
        np.save(os.path.join(out_dir, pred_filename), denoised.astype(np.float32))


def count_predictions(output_root: str, pred_filename: str) -> int:
    return sum(1 for path in Path(output_root).rglob(pred_filename) if path.is_file())


def predict(task):
    seed = int(task.get("seed", 2025))
    seed_all(seed)

    checkpoint = resolve_input_path(task["load_ckpt"])
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint}")

    data_config = load_config("data", task["components"]["data"])
    transform_config = load_config("transform", task["components"]["transform"])
    model_config = load_config("model", task["components"]["model"])

    save_dir = resolve_project_output(task.get("save_dir", "stage3/result"))
    pred_filename = str(task.get("pred_filename", "denoised.npy"))
    if bool(task.get("overwrite", False)) and jt.rank == 0:
        shutil.rmtree(save_dir, ignore_errors=True)
    if jt.rank == 0:
        os.makedirs(save_dir, exist_ok=True)

    predict_dataset_config = DatasetConfig.parse(**data_config["predict_dataset"]).split_by_cls()
    dataset_root = data_config["predict_dataset"]["datapath"]["input_dataset_dir"]

    stage1 = get_model(model_config=model_config, transform_config=transform_config)
    stage2 = get_model(model_config=model_config, transform_config=transform_config)
    load_stage_pair(stage1, stage2, checkpoint)
    model = CascadePredictor(stage1, stage2)
    model.set_predict(True)
    model.eval()

    dataset_module = PCDatasetModule(
        process_fn=stage1._process_fn,
        predict_dataset_config=predict_dataset_config,
        predict_transform=stage1.get_predict_transform(),
        debug=task.get("debug", False),
        seed=seed,
    )
    dataloaders = dataset_module.predict_dataloader()
    assert dataloaders is not None, "predict dataloader is None"
    if not isinstance(dataloaders, dict):
        dataloaders = {"predict": dataloaders}

    print(f"[predict] checkpoint={checkpoint}", flush=True)
    print(f"[predict] output={save_dir}", flush=True)
    with jt.no_grad():
        for name, dataloader in dataloaders.items():
            print(f"[predict] dataloader={name}", flush=True)
            for batch_idx, batch in enumerate(dataloader):
                prediction = model.predict_step(batch)
                write_predictions(batch, prediction, save_dir, dataset_root, pred_filename)
                if batch_idx % 20 == 0:
                    print(f"[predict] {name} batch={batch_idx}", flush=True)

    model.set_predict(False)
    total = count_predictions(save_dir, pred_filename)
    print(f"[predict] saved {total} files named {pred_filename}", flush=True)
    return save_dir


def train_joint(task, resume: bool = False):
    seed = int(task.get("seed", 2025))
    seed_all(seed)

    stage1_ckpt = resolve_input_path(task["stage1_ckpt"])
    stage2_ckpt = resolve_input_path(task["stage2_ckpt"])
    if not os.path.isfile(stage1_ckpt):
        raise FileNotFoundError(f"stage1 checkpoint not found: {stage1_ckpt}")
    if not os.path.isfile(stage2_ckpt):
        raise FileNotFoundError(f"stage2 checkpoint not found: {stage2_ckpt}")

    data_config = load_config("data", task["components"]["data"])
    transform_config = load_config("transform", task["components"]["transform"])
    model_config = load_config("model", task["components"]["model"])
    system_config = load_config("system", task["components"]["system"])

    exp_dir = resolve_project_output(system_config["ckpt_save_dir"])
    ckpt_name = system_config["ckpt_save_name"]
    os.makedirs(exp_dir, exist_ok=True)

    print(f"[checkpoint] load PGD1 from {stage1_ckpt}", flush=True)
    stage1 = build_model(model_config, transform_config, stage1_ckpt)
    print(f"[checkpoint] load PGD2 from {stage2_ckpt}", flush=True)
    stage2 = build_model(model_config, transform_config, stage2_ckpt)

    start_epoch = int(task.get("start_epoch", 1))
    init_joint_ckpt = task.get("init_joint_ckpt")
    if init_joint_ckpt is not None:
        init_joint_ckpt = resolve_input_path(init_joint_ckpt)
        if not os.path.isfile(init_joint_ckpt):
            raise FileNotFoundError(f"initial joint checkpoint not found: {init_joint_ckpt}")
        load_stage_pair(stage1, stage2, init_joint_ckpt)
        print(f"[checkpoint] initialize joint weights from {init_joint_ckpt}", flush=True)

    if resume:
        latest_path, latest_epoch = latest_joint_checkpoint(exp_dir, ckpt_name)
        if latest_path is not None:
            load_stage_pair(stage1, stage2, latest_path)
            start_epoch = latest_epoch + 1
            print(f"[resume] loaded {latest_path}; next joint epoch={start_epoch}", flush=True)

    stage1.set_predict(False)
    stage2.set_predict(False)
    stage1.train()
    stage2.train()

    stage2_lr = float(task.get("stage2_lr", 5e-5))
    stage1_lr = stage2_lr * float(task.get("stage1_lr_mult", 0.1))
    stage1_loss_weight = float(task.get("stage1_loss_weight", 0.1))
    lambda_hard = float(task.get("lambda_hard", 0.002))
    hard_tail_ratio = float(task.get("hard_tail_ratio", 0.1))
    grad_clip = float(task.get("grad_clip", 1.0))
    log_every = int(task.get("log_every", 50))
    joint_epochs = int(task.get("joint_epochs", 200))

    optimizer = optim.Adam(
        [
            {"params": stage1.parameters(), "lr": stage1_lr},
            {"params": stage2.parameters(), "lr": stage2_lr},
        ],
        lr=stage2_lr,
    )
    print(
        f"[joint] epochs={joint_epochs} batch_size={task.get('joint_batch_size', 16)} "
        f"stage2_lr={stage2_lr:g} stage1_lr={stage1_lr:g} "
        f"stage1_loss_weight={stage1_loss_weight:g} "
        f"lambda_hard={lambda_hard:g} hard_tail_ratio={hard_tail_ratio:g} "
        f"grad_clip={grad_clip:g}",
        flush=True,
    )

    dataset_module = build_train_dataset_module(task, data_config, stage1)
    last_checkpoint = None

    for epoch in range(start_epoch, joint_epochs + 1):
        started = time.time()
        losses = []
        loss1_values = []
        loss2_values = []
        hard_values = []
        batch_count = 0
        train_loader = dataset_module.train_dataloader()
        assert train_loader is not None, "train dataloader is None"

        for batch_idx, batch in enumerate(train_loader):
            noisy = batch["pc_noisy"].reshape(-1, stage1.patch_size, 3)
            clean = batch["pc_clean"].reshape(-1, stage1.patch_size, 3)

            disp1, commitment1 = stage1.feature_nets(noisy, calculate_commitment_losses=True)
            x1 = noisy + disp1
            x1_aligned, normal_frames = to_normal_frame(x1)
            clean_aligned, _ = to_normal_frame(clean, normal_frames)
            disp2, commitment2 = stage2.feature_nets(x1_aligned, calculate_commitment_losses=True)
            x2_aligned = x1_aligned + disp2
            x2 = from_normal_frame(x2_aligned, normal_frames)

            loss1 = calc_cd_like_InfoV2(x1, clean)
            loss2 = calc_cd_like_InfoV2(x2_aligned, clean_aligned)
            hard = hard_tail_loss(x2, clean, hard_tail_ratio)
            loss = loss2 + stage1_loss_weight * loss1 + commitment1 + commitment2 + lambda_hard * hard

            optimizer.zero_grad()
            optimizer.backward(loss)
            optimizer.clip_grad_norm(grad_clip)
            optimizer.step()

            if log_every > 0 and batch_idx % log_every == 0:
                losses.append(float(loss.item()))
                loss1_values.append(float(loss1.item()))
                loss2_values.append(float(loss2.item()))
                hard_values.append(float(hard.item()))
                print(
                    f"[debug] epoch {epoch} batch {batch_idx} "
                    f"elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
            batch_count += 1

        last_checkpoint = os.path.join(exp_dir, f"{ckpt_name}_joint_{epoch}.pkl")
        save_stage_pair(stage1, stage2, last_checkpoint)
        print(
            f"[joint] epoch={epoch}/{joint_epochs} batches={batch_count} "
            f"loss={np.mean(losses):.9f} "
            f"loss1={np.mean(loss1_values):.9f} "
            f"loss2={np.mean(loss2_values):.9f} "
            f"hard={np.mean(hard_values):.9f} "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
        print(f"[checkpoint] saved {last_checkpoint}", flush=True)

    return last_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="stage3/configs/task/joint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    task = load_config("task", args.task)
    mode = task.get("mode", "train")
    if mode == "predict":
        predict(task)
    elif mode == "train":
        train_joint(task, resume=args.resume)
    else:
        raise ValueError(f"unsupported mode: {mode}")


if __name__ == "__main__":
    main()
