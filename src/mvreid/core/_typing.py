from typing import Any

import torch
from tensordict import TensorClass


class ReidSample(TensorClass, frozen=True):
    """A sample from the reidentification dataset."""

    images: torch.Tensor
    """
    Images corresponding to the same entity, from possibly different views/cameras.
    Shape: (b, 3, h, w)
    """

    camera_ids: list[Any]
    """The list of camera IDs from which the images were captured."""

    entity_id: Any
    """The ID of the entity in the images."""
