from typing import override

import lightning as L
import torch
from einops import rearrange

from mvreid.core._typing import ReidSample
from mvreid.utils.metrics import MultiViewEuclideanMetric
from mvreid.utils.visreg import VISReg


class ImageEncoder(L.LightningModule):
    """Vision Transformer encoder for multi-view re-identification.

    Processes multiple views of the same entity through a shared ViT backbone,
    producing CLS and patch embeddings. Supports a VIS regularization loss that
    encourages view-invariant representations.
    """

    def __init__(
        self, backbone: torch.nn.Module, visreg_lambda: float, lr: float
    ) -> None:
        super().__init__()

        self._backbone = backbone
        self._visreg = VISReg()
        self._visreg_lambda: float = visreg_lambda
        self._lr: float = lr

        self.val_metrics = MultiViewEuclideanMetric()
        self.test_metrics = MultiViewEuclideanMetric()

    @staticmethod
    def _preprocess(images: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Ensure a batch dimension exists and merge views into the batch axis.

        Handles three input shapes:
            - ``(c, h, w)``           → ``(1, 1, c, h, w)``
            - ``(nv, c, h, w)``       → ``(1, nv, c, h, w)``
            - ``(b, nv, c, h, w)``    → passed through

        Returns:
            Batched images of shape ``(b * nv, c, h, w)`` and the view count ``nv``.
        """
        if images.ndim == 3:
            images = images.expand(1, 1, *images.shape)
        elif images.ndim == 4:
            images = images.expand(1, *images.shape)

        nv = images.shape[1]
        images_batched = rearrange(images, "b nv c h w -> (b nv) c h w")
        return images_batched, nv

    @override
    def forward(self, sample: ReidSample) -> dict[str, torch.Tensor]:
        """Produce all embedding representations for a multi-view sample.

        The backbone is called once on the view-merged batch, then embeddings
        are split into CLS and patch tokens.

        Returns:
            A dict with:
            - **full** — raw backbone output, shape ``(b * nv, nt, d)``
            - **cls**  — CLS token per view, shape ``(b, nv, d)``
            - **patch** — patch tokens per view, shape ``(b, nv, nt - 1, d)``
        """
        images_batched, nv = self._preprocess(sample.images)
        embeddings = self._backbone(images_batched).last_hidden_state

        embeddings_r = rearrange(embeddings, "(b nv) nt d -> b nv nt d", nv=nv)
        return {
            "full": embeddings,
            "cls": embeddings_r[:, :, 0, :],
            "patch": embeddings_r[:, :, 1:, :],
        }

    def _compute_loss(
        self,
        cls_embeddings: torch.Tensor,
        visreg_input: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Embedding-matching loss with optional VIS regularisation.

        The primary loss encourages CLS embeddings of different views of the
        same entity to be similar (mean of ``nv`` views vs. each view).  The
        VIS regularisation shapes the embedding distribution toward a standard
        normal via a projection-based uniformity term.

        Args:
            cls_embeddings: CLS tokens, shape ``(b, nv, d)``.
            visreg_input:   Tensor passed to VISReg (full token sequence by
                default).  Shape ``(b * nv, nt, d)`` or ``(b, nv, d)``.

        Returns:
            ``{"loss": …, "loss_emb": …, "loss_visreg": …}``.
        """
        target = cls_embeddings.mean(dim=1, keepdim=True).expand_as(cls_embeddings)
        loss_emb = torch.nn.functional.mse_loss(cls_embeddings, target)

        if visreg_input is None:
            visreg_input = cls_embeddings
        loss_visreg = self._visreg(visreg_input.transpose(0, 1))

        loss = (1 - self._visreg_lambda) * loss_emb + self._visreg_lambda * loss_visreg

        return {"loss": loss, "loss_emb": loss_emb, "loss_visreg": loss_visreg}

    def training_step(self, batch: ReidSample, batch_idx: int) -> torch.Tensor:
        """Training step: embedding-matching loss + VIS regularisation on CLS."""
        out = self(sample=batch)
        losses = self._compute_loss(out["cls"])

        self.log_dict({f"train/{k}": v for k, v in losses.items()})
        return losses["loss"]

    def _shared_step(
        self,
        batch: ReidSample,
        metrics: MultiViewEuclideanMetric,
        prefix: str,
    ) -> dict[str, torch.Tensor]:
        """Reusable evaluation step shared by validation and test.

        Computes loss, logs it under ``{prefix}/…``, and updates retrieval
        metrics with CLS embeddings.

        Note:
            VISReg receives the **full** token sequence (including patches)
            during evaluation, which provides richer structural signal than
            CLS-only.
        """
        out = self(sample=batch)
        losses = self._compute_loss(out["cls"], visreg_input=out["full"])

        metrics.update(embeddings=out["cls"])

        self.log_dict(
            {f"{prefix}/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
        )
        return losses

    def validation_step(self, batch: ReidSample, batch_idx: int) -> None:
        """Validation step — delegates to :meth:`_shared_step`."""
        self._shared_step(batch, self.val_metrics, "val")

    @override
    def on_validation_epoch_end(self) -> None:
        """Log and reset validation retrieval metrics at epoch end."""
        metrics = self.val_metrics.compute()

        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items()},
            sync_dist=True,
        )

        self.val_metrics.reset()

    def test_step(self, batch: ReidSample, batch_idx: int) -> None:
        """Test step — delegates to :meth:`_shared_step`."""
        self._shared_step(batch, self.test_metrics, "test")

    @override
    def on_test_epoch_end(self) -> None:
        """Log and reset test retrieval metrics at epoch end."""
        metrics = self.test_metrics.compute()

        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items()},
            sync_dist=True,
        )

        self.test_metrics.reset()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure Adam optimiser."""
        return torch.optim.Adam(self.parameters(), lr=self._lr)
