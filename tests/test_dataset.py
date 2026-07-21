import pytest
import torch

from mvreid.core._typing import ReidSample
from mvreid.core.data.dummy import DummyDataset

DATASETS_TO_TEST = [(DummyDataset, lambda _: {})]


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
