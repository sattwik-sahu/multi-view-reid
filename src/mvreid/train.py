import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from mvreid.core.data.base import MultiViewReidDataModule


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    torch.set_float32_matmul_precision("high")

    if cfg.trainer.benchmark:
        torch.backends.cudnn.benchmark = True

    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        log_model="all",
    )

    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.weights_dir,
            monitor="val/recall_at_1",
            mode="max",
            save_top_k=3,
            filename="best-checkpoint-{epoch:02d}-{val/recall_at_1:.3f}",
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    model = hydra.utils.instantiate(cfg.model)

    if cfg.model_compile:
        model._backbone = torch.compile(model._backbone, mode="reduce-overhead")

    datamodule: MultiViewReidDataModule = hydra.utils.instantiate(cfg.dataset)

    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy if cfg.trainer.devices != 1 else "auto",
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
