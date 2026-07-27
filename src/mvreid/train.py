import hydra
import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

# Package imports
from mvreid.core.data import ReidDataModule


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Print resolved parameters
    print(OmegaConf.to_yaml(cfg))

    # Optimizes matrix multiplications on Ampere and newer architectures
    torch.set_float32_matmul_precision("high")

    if cfg.trainer.benchmark:
        torch.backends.cudnn.benchmark = True

    # 1. Initialize Logger
    wandb_logger = WandbLogger(
        project=cfg.logger.project,
        name=cfg.logger.name,
        log_model="all",
    )

    # 2. Configure Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=cfg.weights_dir,
            monitor="val/recall_at_1",
            mode="max",
            save_top_k=3,
            filename="best-checkpoint-{epoch:02d}-{val/recall_at_1:.3f}",
        ),
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(
            monitor="val/recall_at_1",
            patience=8,
            mode="max",
            verbose=True,
        ),
    ]

    # 3. Instantiate Backbone & Lightning Model
    model = hydra.utils.instantiate(cfg.model)

    if cfg.model_compile:
        model._backbone = torch.compile(model._backbone, mode="reduce-overhead")  # type: ignore

    # 4. Instantiate the nested dataset factory
    dataset_factory = hydra.utils.instantiate(cfg.dataset.dataset_factory)

    # 5. Instantiate Datamodule using parameters from the dataset config block
    datamodule = ReidDataModule(
        dataset_factory=dataset_factory,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
    )

    # 6. Initialize Trainer
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

    # Log variables
    wandb_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    # 7. Start Training
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
