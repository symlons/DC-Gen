from data import CTVolumeDataset
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import Resize, Compose

class Resize3D:
    def __init__(self, size):
        self.resize = Resize(size)

    def __call__(self, vol):
        vol = vol.squeeze()

        vol = torch.stack([
            self.resize(slice.unsqueeze(0)).squeeze(0)
            for slice in vol
        ])

        vol = vol.unsqueeze(0)
        return vol

transform = Compose([
    Resize3D((512, 512)),
])

dataset = CTVolumeDataset(
    nifti_dir="/mnt/Volume-eV4BofCN/storage_processed/dataset/train",
    dims="3d",
    n_slices=32,
    fraction=1.0,
    transform=transform
)

print("Dataset length:", len(dataset))

loader = DataLoader(
    dataset,
    batch_size=8,
    num_workers=0,
    persistent_workers=True
)

for i, vol in enumerate(loader):
    print(f"Sample {i}: shape = {tuple(vol.shape)}")

    if i >= 9:
        break