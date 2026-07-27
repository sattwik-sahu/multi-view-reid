import os

import pytest
import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data import DummyDataModule, MultiViewReidDataModule

RUN_DATASETS = os.getenv("RUN_DATASETS") == "1"

DATAMODULES_TO_TEST = [
    (
        DummyDataModule,
        lambda: {
            "n_views": 4,
            "batch_size": 8,
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
        },
    ),
]


def test_reid_sample_creation_and_manipulation():
    n_views = 3
    h, w = 256, 128
    images = torch.randn(n_views, 3, h, w)
    camera_ids = [10, 11, 12]
    entity_id = "person_001"

    sample = ReidSample(
        images=images,
        camera_ids=camera_ids,
        entity_id=entity_id,
        batch_size=[],
    )

    assert sample.batch_size == torch.Size([])
    assert sample.images.shape == (n_views, 3, h, w)
    assert torch.equal(sample.camera_ids, torch.as_tensor(camera_ids))
    assert sample.entity_id == entity_id

    half_precision_sample = sample.half()
    assert half_precision_sample.images.dtype == torch.float16
    assert half_precision_sample.entity_id == "person_001"

    batch_size = 2
    stacked: ReidSample = torch.stack([sample, sample])

    assert stacked.batch_size == torch.Size([batch_size])
    assert stacked.images.shape == (batch_size, n_views, 3, h, w)
    assert len(stacked.camera_ids) == batch_size
    assert torch.equal(stacked.camera_ids[0], torch.as_tensor(camera_ids))
    assert len(stacked.entity_id) == batch_size
    assert stacked.entity_id == ["person_001", "person_001"]


@pytest.mark.skipif(not RUN_DATASETS, reason="Set RUN_DATASETS=1 to run dataset tests")
@pytest.mark.parametrize("dm_class, setup_fn", DATAMODULES_TO_TEST)
def test_datamodule_contract(dm_class, setup_fn):
    kwargs = setup_fn()
    datamodule: MultiViewReidDataModule = dm_class(**kwargs)
    datamodule.setup()
    loader = datamodule.train_dataloader()
    batch = next(iter(loader))

    assert isinstance(batch, ReidSample)
    assert batch.batch_size == torch.Size([kwargs["batch_size"]])
    assert batch.images.shape[:2] == torch.Size(
        [kwargs["batch_size"], kwargs["n_views"]]
    )
    assert batch.camera_ids.shape == torch.Size(
        [kwargs["batch_size"], kwargs["n_views"]]
    )
