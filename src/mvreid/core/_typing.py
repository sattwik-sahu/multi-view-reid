import torch
from tensordict import TensorClass


class ReidSample[TEntityId: int | str, TCameraId: int | str](TensorClass):
    """A sample from the reidentification dataset."""

    images: torch.Tensor
    """
    Images corresponding to the same entity, from possibly different views/cameras.
    Shape: (b, 3, h, w)
    """

    camera_ids: list[TCameraId]
    """The list of camera IDs from which the images were captured."""

    entity_id: TEntityId
    """The ID of the entity in the images."""
