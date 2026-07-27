from typing import override

import torch
from einops import rearrange, repeat
from torchmetrics import Metric


class MultiViewEuclideanMetric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Gathers and concatenates embeddings across steps and DDP ranks
        self.add_state("embeddings", default=[], dist_reduce_fx="cat")

    @override
    def update(self, embeddings: torch.Tensor):
        """
        Args:
            embeddings (torch.Tensor): Embedding tensor of shape (batch_size, n_views, d)
        """
        self.embeddings.append(embeddings)

    def compute(self):
        # Concatenate steps/batches into a single tensor
        if isinstance(self.embeddings, list):
            if len(self.embeddings) == 0:
                return self._empty_result()
            embeddings_tensor = torch.cat(self.embeddings, dim=0)
        else:
            embeddings_tensor = self.embeddings

        B, V, D = embeddings_tensor.shape
        if B < 2 or V < 2:
            return self._empty_result()

        # 1. Flatten view dimension into the batch dimension using einops
        flat_embeddings = rearrange(embeddings_tensor, "b v d -> (b v) d")

        # 2. Compute pairwise Euclidean distances
        dist_matrix = torch.cdist(flat_embeddings, flat_embeddings, p=2)

        # 3. Map views to object identities using einops repeat
        # This replaces repeat_interleave; 'b -> (b v)' behaves like consecutive copies (e.g., 0,0,0,1,1,1)
        labels_base = torch.arange(B, device=self.device)
        labels = repeat(labels_base, "b -> (b v)", v=V)

        # 4. Compute Recall@1 (Minimum Distance Search)
        dist_matrix_retrieval = dist_matrix.clone()
        dist_matrix_retrieval.fill_diagonal_(
            float("inf")
        )  # Exclude matching a view with itself

        nearest_neighbor_idx = torch.argmin(dist_matrix_retrieval, dim=-1)
        nearest_neighbor_labels = labels[nearest_neighbor_idx]
        recall_at_1 = (nearest_neighbor_labels == labels).float().mean()

        # 5. Compute Distance Statistics using explicit einops-like column/row broadcasting
        labels_col = rearrange(labels, "n -> n 1")
        labels_row = rearrange(labels, "n -> 1 n")
        same_object_mask = labels_col == labels_row

        # Exclude self-distance (diagonal is False)
        intra_mask = same_object_mask & ~torch.eye(
            B * V, dtype=torch.bool, device=self.device
        )
        inter_mask = ~same_object_mask

        mean_intra_dist = dist_matrix[intra_mask].mean()
        mean_inter_dist = dist_matrix[inter_mask].mean()
        distance_gap = mean_inter_dist - mean_intra_dist

        return {
            "recall_at_1": recall_at_1,
            "mean_intra_dist": mean_intra_dist,
            "mean_inter_dist": mean_inter_dist,
            "distance_gap": distance_gap,
        }

    def _empty_result(self):
        return {
            "recall_at_1": torch.tensor(0.0, device=self.device),
            "mean_intra_dist": torch.tensor(0.0, device=self.device),
            "mean_inter_dist": torch.tensor(0.0, device=self.device),
            "distance_gap": torch.tensor(0.0, device=self.device),
        }
