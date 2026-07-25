import os

import pytest
import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.dummy import DummyDataset

RUN_DATASETS = os.getenv("RUN_DATASETS") == "1"

DATASETS_TO_TEST = [(DummyDataset, lambda _: {})]


def test_reid_sample_creation_and_manipulation():
    # -------------------------------------------------------------------------
    # 1. ARRANGE: Define dummy data inputs for a single multi-view sample
    # -------------------------------------------------------------------------
    n_views = 3
    h, w = 256, 128
    images = torch.randn(n_views, 3, h, w)  # Shape: [n_views, channels, H, W]
    camera_ids = [10, 11, 12]  # List of length n_views
    entity_id = "person_001"  # Single entity string ID

    # -------------------------------------------------------------------------
    # 2. ACT: Instantiate the ReidSample (batch_size is empty [] for a single sample)
    # -------------------------------------------------------------------------
    sample = ReidSample(
        images=images,
        camera_ids=camera_ids,
        entity_id=entity_id,
        batch_size=[],  # <--- CORRECT: An unbatched sample is a scalar/empty batch
    )

    # -------------------------------------------------------------------------
    # 3. ASSERT: Verify a single sample's properties
    # -------------------------------------------------------------------------
    # A single sample has no batch dimensions
    assert sample.batch_size == torch.Size([])

    # Verify that shapes and values are preserved exactly as loaded
    assert sample.images.shape == (n_views, 3, h, w)
    assert torch.equal(sample.camera_ids, torch.as_tensor(camera_ids))  # type: ignore
    assert sample.entity_id == entity_id

    # -------------------------------------------------------------------------
    # 4. ASSERT: Casting (.half(), .to())
    # -------------------------------------------------------------------------
    # Converting the sample to half-precision propagates to the internal tensors
    half_precision_sample = sample.half()
    assert half_precision_sample.images.dtype == torch.float16
    assert half_precision_sample.entity_id == "person_001"  # Non-tensors are unaffected

    # -------------------------------------------------------------------------
    # 5. ASSERT: Stacking (Simulating DataLoader Collation)
    # -------------------------------------------------------------------------
    # Let's stack 2 samples to simulate a DataLoader batching process (Batch Size = 2)
    batch_size = 2
    stacked: ReidSample = torch.stack([sample, sample])  # type: ignore

    # The collated batch now correctly has a batch_size of [2]
    assert stacked.batch_size == torch.Size([batch_size])

    # The image tensor is now properly batched: [batch_size, n_views, channels, H, W]
    assert stacked.images.shape == (batch_size, n_views, 3, h, w)

    # Non-tensor camera IDs are collated into a nested list structure: [batch_size, n_views]
    assert len(stacked.camera_ids) == batch_size
    assert torch.equal(
        stacked.camera_ids[0], torch.as_tensor(camera_ids)
    )  # First item's camera IDs

    # Entity IDs are collated into a list of length [batch_size]
    assert len(stacked.entity_id) == batch_size
    assert stacked.entity_id == ["person_001", "person_001"]


@pytest.mark.skipif(not RUN_DATASETS, reason="Set RUN_DATASETS=1 to run dataset tests")
@pytest.mark.parametrize("dataset_class, setup_fn", DATASETS_TO_TEST)
def test_dataset_contract(dataset_class, setup_fn, tmp_path):
    """Loops through all datasets, runs their custom setups, and validates their contract."""
    n_views: int = 4
    batch_size: int = 8

    # Run the dataset's specific setup function to get its custom arguments
    custom_kwargs = setup_fn(tmp_path)

    # Instantiate the class using n_views and unpack the custom kwargs
    dataset = dataset_class(n_views=n_views, **custom_kwargs)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, collate_fn=torch.stack
    )
    batch = next(iter(loader))

    # Run your standard assertions
    assert isinstance(len(dataset), int)
    assert len(dataset) >= 0

    if len(dataset) > 0:
        assert isinstance(batch, ReidSample)
        assert batch.batch_size == torch.Size([batch_size])
        assert batch.images.shape[:2] == torch.Size([batch_size, n_views])
        assert batch.camera_ids.shape == torch.Size([batch_size, n_views])
