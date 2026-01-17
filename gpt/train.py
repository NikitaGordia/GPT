from typing import List

import hydra
import lightning.pytorch as L
from lightning.pytorch.loggers import TensorBoardLogger
from loguru import logger
from omegaconf import DictConfig
import tiktoken
import torch

from gpt.data import FineWebDataModule
from gpt.hellaswag.evaluate import evaluate as hs_validation
from gpt.hellaswag.evaluate import render_results
from gpt.learning_rate import CosineDecayWarmupScheduler
from gpt.model import GPT
from gpt.utils import RuntimeEnvironment, Timeit, count_parameters, setup_env


class HellaswagCallback(L.callbacks.Callback):
    """Runs Hellaswag validation at specified intervals."""

    def __init__(self, env: RuntimeEnvironment, cfg_hellaswag: DictConfig, encoding: str):
        super().__init__()
        self.tokenizer = tiktoken.get_encoding(encoding)
        self.env = env
        self.hs_cfg = cfg_hellaswag

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not trainer.is_global_zero:
            return

        logger.info("Running Hellaswag validation...")

        num_total, num_correct, num_correct_norm = hs_validation(
            self.hs_cfg,
            self.env,
            pl_module.model,
            self.tokenizer,
        )

        if num_total > 0:
            acc = num_correct / num_total * 100
            norm_acc = num_correct_norm / num_total * 100
            pl_module.log("hs_val_acc", acc, rank_zero_only=True)
            pl_module.log("hs_val_norm_acc", norm_acc, rank_zero_only=True)
            logger.info(render_results(num_total, num_correct, num_correct_norm))
        else:
            logger.warning("Hellaswag validation ran but found 0 examples.")


class GPTLightningModule(L.LightningModule):
    def __init__(self, model: GPT, train_cfg: DictConfig) -> None:
        super().__init__()

        self.train_cfg = train_cfg
        self.model = model
        logger.info(f"Model size: {count_parameters(self.model):,}")

    def forward(self, X, Y):
        return self.model(X, Y)

    def training_step(self, batch, _):
        X, Y = batch
        _, loss = self(X, Y)

        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, sync_dist=True)

        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate", lr, on_step=True, on_epoch=False, prog_bar=False, sync_dist=False
        )
        print("train", loss.item(), "learning_rate", lr)
        return loss

    def validation_step(self, batch, _):
        X, Y = batch
        _, loss = self(X, Y)
        print("val", loss.item())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.model.configure_optimizers(
            weight_decay=self.train_cfg.optimizer.weight_decay,
            learning_rate=self.train_cfg.lr.max,
            device_type=self.device.type,
        )
        scheduler = CosineDecayWarmupScheduler(
            optimizer=optimizer,
            warmup_steps=self.train_cfg.lr.warmup_steps,
            max_steps=self.train_cfg.max_steps,
            min_lr=self.train_cfg.lr.max * self.train_cfg.lr.min_factor,
            max_lr=self.train_cfg.lr.max,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


def configure_callbacks(env: RuntimeEnvironment, cfg: DictConfig) -> List[L.Callback]:
    callbacks = [
        L.callbacks.LearningRateMonitor(logging_interval="step"),
    ]
    if cfg.train.validation.hellaswag:
        callbacks.append(HellaswagCallback(env, cfg.data.hellaswag, cfg.tokens.encoding))
    return callbacks


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def train(cfg: DictConfig) -> None:
    env = setup_env(cfg.env)

    model = GPTLightningModule(GPT(cfg.model), cfg.train)

    if cfg.train.compile_model:
        with Timeit() as timeit:
            logger.info("Compiling the model...")
            model = torch.compile(model)
            logger.info(f"Model is compiled within {timeit.now():.3f} seconds.")

    datamodule = FineWebDataModule(
        cfg.data.fineweb.path,
        cfg.train.batch_size,
        cfg.train.seq_length,
        num_workers=cfg.data.num_workers_per_device,
    )

    tokens_per_micro_batch = cfg.train.micro_batch_size * cfg.train.seq_length * env.ddp_world_size
    assert cfg.train.batch_size % tokens_per_micro_batch == 0, (
        "make sure batch_size is divisible by micro_batch_size * seq_length"
    )
    grad_accum_steps = cfg.train.batch_size // tokens_per_micro_batch
    logger.info(f"gradient accumulation steps: {grad_accum_steps}")

    trainer = L.Trainer(
        max_steps=cfg.train.max_steps,
        val_check_interval=2,
        accelerator=env.device_type,
        devices="auto",  # auto-detects number of devices
        strategy=cfg.train.strategy,
        precision="bf16-mixed" if env.use_autocast else "32-true",
        gradient_clip_val=cfg.train.optimizer.grad_clip,
        accumulate_grad_batches=grad_accum_steps,
        logger=TensorBoardLogger("lightning_logs", name="gpt_fineweb"),
        callbacks=configure_callbacks(env, cfg),
    )
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()
