import os
from pathlib import Path
from typing import Any

import pytest
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

from mvreid.core._typing import ReidSample
from mvreid.core.data.dummy import DummyDataset
from mvreid.core.data.veri776 import Veri776Dataset

# --- CONFIGURATION ---
RUN_DATASETS = os.getenv("RUN_DATASETS") == "1"
LOCAL_VERI_ROOT = "/home/jataayu/multi-view-reid/VeRi"

# --- 1. REID SAMPLE UNIT TESTS (Missing Logic Added) ---


def test_reid_sample_contract_logic():
    """Checks missing ReidSample logic: creation, casting, and stacking."""
    n_views, h, w = 3, 128, 128
    images = torch.randn(n_views, 3, h, w)
    camera_ids = [10, 11, 12]
    entity_id = 100

    # Act
    sample = ReidSample(
        images=images, camera_ids=camera_ids, entity_id=entity_id, batch_size=[]
    )

    # Assert Properties
    assert sample.batch_size == torch.Size([])
    assert sample.images.shape == (n_views, 3, h, w)

    # Missing Logic: Casting (.half() test)
    half_sample = sample.half()
    assert half_sample.images.dtype == torch.float16
    assert half_sample.entity_id == 100

    # Missing Logic: Stacking (Batch size check)
    batch_size = 2
    stacked: ReidSample = torch.stack([sample, sample])
    assert stacked.batch_size == torch.Size([batch_size])
    assert stacked.images.shape == (batch_size, n_views, 3, h, w)
    assert len(stacked.entity_id) == batch_size  # Entity IDs become a list in batch
    print("\n✅ ReidSample property logic passed.")


# --- 2. THE LOCAL VISUAL TEST (Your script integrated) ---


def test_veri_locally_with_full_logic():
    """
    Runs the full logic on your local path: /home/jataayu/multi-view-reid/VeRi
    Includes: Dataloader check, Shape check, Resolution check, and Visualization.
    """
    ROOT = Path(LOCAL_VERI_ROOT)
    if not ROOT.exists():
        pytest.skip(f"Local VeRi not found at {ROOT}")

    N_VIEWS = 10
    BATCH_SIZE = 4

    print(f"\n🔍 Testing VeRi-776 Dataset at: {ROOT}")

    # 1. Instantiate Dataset
    dataset = Veri776Dataset(root=ROOT, n_views=N_VIEWS, split="train")
    print(f"✅ Dataset initialized. Entities: {len(dataset)}")

    # 2. Test Single Item & Missing logic checks
    sample = dataset[0]
    assert isinstance(sample, ReidSample)
    assert sample.batch_size == torch.Size([])
    assert sample.images.shape == (N_VIEWS, 3, 128, 128)
    print(f"✅ Single sample check passed (Resolution: 128x128).")

    # 3. Test DataLoader Stacking & Batch Size logic
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=torch.stack)
    batch = next(iter(loader))

    print(f"✅ Batch check: Stacking successful.")
    assert batch.batch_size == torch.Size([BATCH_SIZE])
    assert batch.images.shape == (BATCH_SIZE, N_VIEWS, 3, 128, 128)
    assert len(batch.entity_id) == BATCH_SIZE

    # 4. Visualization
    print("🖼️ Generating visualization for 10 views...")
    fig, axes = plt.subplots(1, N_VIEWS, figsize=(15, 4))
    fig.suptitle(
        f"Local VeRi Check | Entity: {sample.entity_id} | 10 Views", fontsize=14
    )

    for i in range(N_VIEWS):
        img = sample.images[i].permute(1, 2, 0).numpy()
        axes[i].imshow(img)
        axes[i].set_title(f"C:{sample.camera_ids[i]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


# --- 3. DUMMY SETUP FOR CI ---


def setup_veri_dummy(tmp_path: Path):
    root = tmp_path / "veri_dummy"
    train_dir = root / "image_train"
    train_dir.mkdir(parents=True)
    for i in range(5):
        img = Image.new("RGB", (128, 128))
        img.save(train_dir / f"0001_c00{i}_000{i}.jpg")
    return {"root": root, "split": "train"}


DATASETS_TO_TEST = [(DummyDataset, lambda _: {}), (Veri776Dataset, setup_veri_dummy)]


@pytest.mark.skipif(
    not RUN_DATASETS, reason="Set RUN_DATASETS=1 to run automated tests"
)
@pytest.mark.parametrize("dataset_class, setup_fn", DATASETS_TO_TEST)
def test_dataset_contract(dataset_class, setup_fn, tmp_path):
    """Standard automated contract test."""
    n_views, batch_size = 4, 2
    custom_kwargs = setup_fn(tmp_path)
    dataset = dataset_class(n_views=n_views, **custom_kwargs)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=torch.stack)
    batch = next(iter(loader))

    assert isinstance(batch, ReidSample)
    assert batch.batch_size == torch.Size([batch_size])
    assert batch.images.shape[-2:] == torch.Size([128, 128])


if __name__ == "__main__":
    # To run manually like your old script:
    test_reid_sample_contract_logic()
    test_veri_locally_with_full_logic()
