import lightning as L
import torch
from einops import rearrange

from mvreid.core._typing import ReidSample
from mvreid.utils.metrics import MultiViewEuclideanMetric
from mvreid.utils.visreg import VISReg


class ImageEncoder(L.LightningModule):
    def __init__(
        self, backbone: torch.nn.Module, visreg_lambda: float, lr: float
    ) -> None:
        super().__init__()

        self._backbone = backbone
        self._visreg = VISReg()
        self._visreg_lambda: float = visreg_lambda
        self._lr: float = lr

        # Instantiate the custom TorchMetrics object
        self.val_metrics = MultiViewEuclideanMetric()

    def training_step(self, batch: ReidSample, batch_idx):
        images: torch.Tensor = batch.images

        if images.ndim == 3:
            images = images.expand(1, 1, *images.shape)
        elif images.ndim == 4:
            images = images.expand(1, *images.shape)

        nv = images.shape[1]

        # Re-arrange images to pass into encoder
        images_batched: torch.Tensor = rearrange(images, "b nv c h w -> (b nv) c h w")

        # Pass through encoder
        embeddings: torch.Tensor = self._backbone(images_batched).last_hidden_state

        # Calculate loss
        cls_embeddings: torch.Tensor = rearrange(
            embeddings[:, 0], "(b nv) d -> b nv d", nv=nv
        )
        target_embeddings = cls_embeddings.mean(1, keepdim=True).repeat(1, nv, 1)
        loss_emb = torch.nn.functional.mse_loss(cls_embeddings, target_embeddings)
        loss_visreg = self._visreg(embeddings.transpose(0, 1))
        loss = (1 - self._visreg_lambda) * loss_emb + self._visreg_lambda * loss_visreg

        # Log the losses
        self.log_dict(
            {
                "train/loss": loss,
                "train/loss_emb": loss_emb,
                "train/loss_visreg": loss_visreg,
            },
        )
        return loss

    def validation_step(self, batch: ReidSample, batch_idx):
        images: torch.Tensor = batch.images

        if images.ndim == 3:
            images = images.expand(1, 1, *images.shape)
        elif images.ndim == 4:
            images = images.expand(1, *images.shape)

        nv = images.shape[1]

        # Re-arrange images to pass into encoder
        images_batched: torch.Tensor = rearrange(images, "b nv c h w -> (b nv) c h w")

        # Pass through encoder
        embeddings = self._backbone(images_batched).last_hidden_state

        # Calculate loss
        cls_embeddings: torch.Tensor = rearrange(
            embeddings[:, 0], "(b nv) d -> b nv d", nv=nv
        )
        target_embeddings = cls_embeddings.mean(1, keepdim=True).repeat(1, nv, 1)
        loss_emb = torch.nn.functional.mse_loss(cls_embeddings, target_embeddings)
        loss_visreg = self._visreg(embeddings.transpose(0, 1))
        loss = (1 - self._visreg_lambda) * loss_emb + self._visreg_lambda * loss_visreg

        # Accumulate validation step embeddings into the TorchMetrics state
        self.val_metrics.update(embeddings=cls_embeddings)  # type: ignore

        # Log step-level validation losses (corrected log keys to "val/")
        self.log_dict(
            {
                "val/loss": loss,
                "val/loss_emb": loss_emb,
                "val/loss_visreg": loss_visreg,
            },
            on_step=False,
            on_epoch=True,
        )

    def on_validation_epoch_end(self):
        # Compute multi-view metrics using the collected global validation pool
        metrics = self.val_metrics.compute()

        # Log the aggregated global metrics
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            sync_dist=True,
        )

        # Reset metric accumulated states for the next epoch
        self.val_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters(), lr=self._lr)
        return optimizer
