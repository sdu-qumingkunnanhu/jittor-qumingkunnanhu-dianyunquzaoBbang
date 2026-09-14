from collections import defaultdict
from typing import Any, Dict, List, Optional

import jittor as jt
from jittor import optim
import os
from tqdm import tqdm

from ..data.asset import Asset
from ..data.dataset import PCDatasetModule


PROJECT_ROOT = os.path.abspath(os.getcwd())


def _get_item(x):
    if isinstance(x, jt.Var):
        return x.item()
    return x


def resolve_project_output(path: str) -> str:
    resolved = path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)
    resolved = os.path.abspath(resolved)
    if os.path.commonpath([PROJECT_ROOT, resolved]) != PROJECT_ROOT:
        raise ValueError(f"output path must stay inside the run directory: {path}")
    return resolved


def distributed_average(values: List[float]):
    local_sum = float(sum(values))
    local_count = float(len(values))
    if jt.in_mpi:
        stat = jt.array([local_sum, local_count]).mpi_all_reduce("sum")
        stat_np = stat.numpy()
        total_sum, total_count = float(stat_np[0]), float(stat_np[1])
    else:
        total_sum, total_count = local_sum, local_count
    if total_count == 0:
        return None
    return total_sum / total_count


def get_optimizer(optimizer_config, model):
    cfg = dict(optimizer_config)
    target = cfg.pop("__target__")
    mapping = {
        "sgd": optim.SGD,
        "adam": optim.Adam,
    }
    if target not in mapping:
        raise ValueError(f"unsupported optimizer: {target}")
    return mapping[target](model.parameters(), **cfg)


class DummyWriter:
    def write(self, batch, prediction: List[Dict], dataset_module: Optional[PCDatasetModule] = None):
        pass


class DummySystem:
    def __init__(
        self,
        dataset_module: PCDatasetModule,
        model: Any,
        loss_config=None,
        optimizer_config=None,
        trainer_config=None,
        writer: Optional[DummyWriter] = None,
        ckpt_save_dir: str = "experiments",
        ckpt_save_name: str = "checkpoint",
    ):
        self.dataset_module = dataset_module
        self.model = model
        self.loss_config = {"loss": 1.0} if loss_config is None else dict(loss_config)
        self.optimizer = get_optimizer(optimizer_config, model) if optimizer_config is not None else None
        self.writer = writer
        self.ckpt_save_dir = resolve_project_output(ckpt_save_dir)
        self.ckpt_save_name = ckpt_save_name
        self.trainer_config = {} if trainer_config is None else dict(trainer_config)
        self.epochs = int(self.trainer_config.get("epochs", 1))
        self.validate_every = int(self.trainer_config.get("validate_every", 1))
        self.save_every = int(self.trainer_config.get("save_every", 1))
        self.log_every = int(self.trainer_config.get("log_every", 10))
        self.scheduler_config = self.trainer_config.get("scheduler", None)
        self._best_scheduler_metric = None
        self._scheduler_bad_epochs = 0
        self._validation_loss = defaultdict(list)
        self._train_loss = []

    def _weighted_loss(self, loss_dict: Dict[str, jt.Var]):
        assert isinstance(loss_dict, dict), "loss_dict must be a dict"
        loss_sum = 0.0
        for name, value in loss_dict.items():
            assert name in self.loss_config, f"unspecified loss name: `{name}`"
            weight = self.loss_config[name]
            if weight != 0:
                loss_sum = loss_sum + weight * value
        if not isinstance(loss_sum, jt.Var):
            loss_sum = jt.array(loss_sum)
        return loss_sum

    def training_step(self, batch):
        loss_dict = self.model.training_step(batch)
        return self._weighted_loss(loss_dict)

    def record_train_loss(self, loss):
        value = _get_item(loss)
        self._train_loss.append(value)
        return value

    def validation_step(self, batch):
        loss_dict = self.model.validation_step(batch)
        loss = self._weighted_loss(loss_dict)
        cls = "all"
        assets: Optional[List[Asset]] = batch.get("asset", None)
        if assets:
            cls = assets[0].cls or "unknown"
        self._validation_loss[f"val/{cls}_loss"].append(_get_item(loss))
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        return self.model.predict_step(batch)

    def on_train_epoch_start(self, epoch):
        self._train_loss = []

    def on_train_epoch_end(self, epoch):
        avg = distributed_average(self._train_loss)
        if avg is not None:
            self._step_scheduler(epoch, avg)
        if jt.rank == 0 and avg is not None:
            print(f"[train] epoch={epoch} loss={avg:.6f}")

    def _step_scheduler(self, epoch, metric):
        if self.scheduler_config is None:
            return
        target = self.scheduler_config.get("__target__", "plateau")
        if target != "plateau":
            raise ValueError(f"unsupported scheduler: {target}")
        frequency = int(self.scheduler_config.get("frequency", 1))
        if frequency <= 0 or (epoch + 1) % frequency != 0:
            return
        threshold = float(self.scheduler_config.get("threshold", 1e-12))
        if self._best_scheduler_metric is None or metric < self._best_scheduler_metric - threshold:
            self._best_scheduler_metric = metric
            self._scheduler_bad_epochs = 0
            return
        self._scheduler_bad_epochs += 1
        patience = int(self.scheduler_config.get("patience", 2))
        if self._scheduler_bad_epochs <= patience:
            return
        factor = float(self.scheduler_config.get("factor", 0.5))
        min_lr = float(self.scheduler_config.get("min_lr", 1e-9))
        old_lr = float(self.optimizer.lr)
        new_lr = max(old_lr * factor, min_lr)
        self.optimizer.lr = new_lr
        self._scheduler_bad_epochs = 0
        if jt.rank == 0:
            print(f"[scheduler] epoch={epoch} lr {old_lr:.6g} -> {new_lr:.6g}")

    def on_validation_epoch_start(self, epoch):
        self._validation_loss = defaultdict(list)

    def on_validation_epoch_end(self, epoch):
        for name, values in sorted(self._validation_loss.items()):
            avg = distributed_average(values)
            if jt.rank == 0 and avg is not None:
                print(f"[{name}] epoch={epoch} loss={avg:.6f}")

    def on_before_optimizer_step(self, optimizer):
        pass

    def _progress(self, dataloader, desc):
        if jt.rank != 0:
            return dataloader
        total = max(1, len(dataloader) // dataloader.batch_size)
        return tqdm(dataloader, total=total, desc=desc)

    def _run_validation(self, epoch):
        validate_dataloader = self.dataset_module.validate_dataloader()
        if validate_dataloader is None:
            return
        self.model.eval()
        self.on_validation_epoch_start(epoch)
        if not isinstance(validate_dataloader, dict):
            validate_dataloader = {"validate": validate_dataloader}
        with jt.no_grad():
            for name, dataloader in validate_dataloader.items():
                for batch in self._progress(dataloader, f"Validate {name}"):
                    self.validation_step(batch)
        self.on_validation_epoch_end(epoch)

    def _save_checkpoint(self, epoch):
        if jt.rank != 0:
            return
        if self.save_every <= 0 or (epoch + 1) % self.save_every != 0:
            return
        os.makedirs(self.ckpt_save_dir, exist_ok=True)
        checkpoint_path = os.path.join(self.ckpt_save_dir, f"{self.ckpt_save_name}_{epoch}.pkl")
        self.model.save(checkpoint_path)
        print(f"[checkpoint] saved {checkpoint_path}")

    def train(self):
        assert self.optimizer is not None, "optimizer is None, cannot train"
        self.model.set_predict(False)
        for epoch in range(self.epochs):
            self.model.train()
            self.on_train_epoch_start(epoch)
            train_dataloader = self.dataset_module.train_dataloader()
            assert train_dataloader is not None, "train_dataloader is None"
            pbar = self._progress(train_dataloader, f"Train epoch {epoch}")
            for batch_idx, batch in enumerate(pbar):
                loss = self.training_step(batch)
                self.optimizer.zero_grad()
                self.optimizer.backward(loss)
                self.on_before_optimizer_step(self.optimizer)
                self.optimizer.step()
                should_log = self.log_every > 0 and batch_idx % self.log_every == 0
                if should_log:
                    loss_value = self.record_train_loss(loss)
                    if jt.rank == 0 and hasattr(pbar, "set_description"):
                        pbar.set_description(f"Epoch {epoch}, Loss: {loss_value:.6f}")
            self.on_train_epoch_end(epoch)
            if self.validate_every > 0 and (epoch + 1) % self.validate_every == 0:
                self._run_validation(epoch)
            self._save_checkpoint(epoch)

    def predict(self):
        self.model.set_predict(True)
        self.model.eval()
        predict_dataloader = self.dataset_module.predict_dataloader()
        assert predict_dataloader is not None, "predict_dataloader is None"
        if not isinstance(predict_dataloader, dict):
            predict_dataloader = {"predict": predict_dataloader}
        with jt.no_grad():
            for name, dataloader in predict_dataloader.items():
                pbar = self._progress(dataloader, f"Predict {name}")
                for batch_idx, batch in enumerate(pbar):
                    prediction = self.predict_step(batch, batch_idx)
                    if self.writer is not None and jt.rank == 0:
                        self.writer.write(batch, prediction, dataset_module=self.dataset_module)
                    if jt.rank == 0 and hasattr(pbar, "set_description"):
                        pbar.set_description(f"Predicting {name}, Batch {batch_idx}")


BaseWriter = DummyWriter
BaseSystem = DummySystem
