import os
import numpy as np
from viz import diff_visualization

artifact_dir = "test_artifacts"
os.makedirs(artifact_dir, exist_ok=True)

gt_single = np.random.rand(64, 64).astype(np.float32)
recon_single = gt_single + (np.random.rand(64, 64).astype(np.float32) - 0.5) * 0.2
save_path_single = os.path.join(artifact_dir, "diff_single.png")
diff_visualization(gt_single, recon_single, save_path_single, title_suffix="Single Image")
print(f"Saved single-image diff to {save_path_single}")

batch_size = 4
gt_batch = np.random.rand(batch_size, 64, 64).astype(np.float32)
recon_batch = gt_batch + (np.random.rand(batch_size, 64, 64).astype(np.float32) - 0.5) * 0.2
save_path_batch = os.path.join(artifact_dir, "diff_batch.png")
diff_visualization(gt_batch, recon_batch, save_path_batch, title_suffix="Batch Test")
print(f"Saved batch diff to {save_path_batch}")