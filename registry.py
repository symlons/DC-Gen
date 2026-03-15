from data import CTVolumeDataset
import torch.nn.functional as F

dataset_registry = {
    "CTVolume": CTVolumeDataset,
}

loss_registry = {
    "l1": F.l1_loss,
    "mse": F.mse_loss,
}
