import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

from abc import ABC, abstractmethod
from typing import Any
import os
from basics import ensure_batch, to_numpy, ensure_numpy, compute_map

class Saver(ABC):
    @abstractmethod
    def save(self, gt, recon, save_dir, global_step):
        pass

class Visualize(Saver):
    available_savers: dict[str, Any] = {}
    attrs: dict[str, Any]

    def __init_subclass__(cls, viz_type, **kwargs):
        super().__init_subclass__(**kwargs)
        if not viz_type:
            raise ValueError("Invalid viz_type.")
        cls.available_savers[viz_type] = cls
        cls.viz_type = viz_type

    def __new__(cls, viz_type: str, **attrs):
        try:
            subclass = cls.available_savers[viz_type]
        except KeyError:
            raise NotImplementedError(f'{viz_type} is not implemented.')
        obj = super().__new__(subclass)
        obj.attrs = attrs
        return obj

    def __init__(self, **attributes: Any):
        for attr, value in attributes.items():
            setattr(self, attr, value)

    def load(self, data):
        raise NotImplementedError(
            f'The viz type "{self.__class__.__name__}" does not have a method implemented.'
        )

class VisualizeBase(Visualize, viz_type="base"):
    def _prepare_and_diff(self, gt, recon):
        gt = ensure_batch(ensure_numpy(gt).squeeze())
        recon = ensure_batch(ensure_numpy(recon).squeeze())
        diff = compute_map(gt, recon)
        return gt, recon, diff

    def _get_plot_ranges(self, gt_list, recon_list, diff_list):
        vmin, vmax = get_plot_range(np.concatenate([gt_list, recon_list]))
        diff_vmin, diff_vmax = get_plot_range(diff_list, symmetric=True)
        return vmin, vmax, diff_vmin, diff_vmax

    def _plot_and_save(self, gt_list, recon_list, diff_list, save_path, titles=None):
        B = len(gt_list)
        vmin, vmax, diff_vmin, diff_vmax = self._get_plot_ranges(gt_list, recon_list, diff_list)

        fig, axes = plot_batch(
            [[gt_list[i], recon_list[i], diff_list[i]] for i in range(B)],
            titles=titles if titles else [[f"GT {i}", f"Recon {i}", f"Diff {i}"] for i in range(B)],
            cmaps=['gray', 'gray', 'bwr'],
            vmin_vmax=[
                {'vmin': vmin, 'vmax': vmax},
                {'vmin': vmin, 'vmax': vmax},
                {'vmin': -diff_vmax, 'vmax': diff_vmax}
            ]
        )
        save_figure(fig, save_path, im_for_colorbar=axes[0,2].images[0], label="Difference Intensity")

class Visualize2D(VisualizeBase, viz_type="2d"):
    def save(self, gt, recon, save_dir, global_step, title_suffix=""):
            gt_batch, recon_batch, diff_batch = self._prepare_and_diff(gt, recon)

            for i in range(gt_batch.shape[0]):
                slice_save_path = os.path.join(save_dir, f"sample_step{global_step}_{i}_{title_suffix}.png")
                titles = [[f"GT {title_suffix} (Sample {i})",
                           f"Recon {title_suffix} (Sample {i})",
                           f"Diff {title_suffix} (Sample {i})"]]
                self._plot_and_save(gt_batch[i:i+1], recon_batch[i:i+1], diff_batch[i:i+1],
                                    slice_save_path, titles=titles)

class Visualize3D(VisualizeBase, viz_type="3d"):
    def save(self, gt, recon, save_dir, global_step, slice_indices=None):
            batch_size = gt.shape[0]
            depth_center = gt.shape[2] // 2
            slice_indices = slice_indices or [depth_center]

            for b in range(batch_size):
                recon_volume, gt_volume, diff_volume = self._prepare_and_diff(gt[b, 0], recon[b, 0])

                nib.save(nib.Nifti1Image(recon_volume, affine=np.eye(4)), os.path.join(save_dir, f"recon_step{global_step}_b{b}.nii.gz"))
                nib.save(nib.Nifti1Image(gt_volume, affine=np.eye(4)), os.path.join(save_dir, f"gt_step{global_step}_b{b}.nii.gz"))
                nib.save(nib.Nifti1Image(diff_volume, affine=np.eye(4)), os.path.join(save_dir, f"diff_step{global_step}_b{b}.nii.gz"))

                for idx in slice_indices:
                    recon_slice, gt_slice, diff_slice = self._prepare_and_diff(recon_volume[idx], gt_volume[idx])
                    slice_save_path = os.path.join(save_dir, f"slice_step{global_step}_{idx}_b{b}.png")
                    titles = [[f"GT B{b} Slice{idx}", f"Recon B{b} Slice{idx}", f"Diff B{b} Slice{idx}"]]
                    self._plot_and_save(gt_slice, recon_slice, diff_slice, slice_save_path, titles=titles)

def get_plot_range(data, symmetric=False):
    if symmetric:
        abs_max = max(np.max(np.abs(data)), 1e-8)
        return -abs_max, abs_max
    return data.min(), data.max()

def plot_batch(batch_data, titles=None, cmaps=None, vmin_vmax=None):
    B = len(batch_data)
    cols = len(batch_data[0])
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(B, cols, figsize=(5*cols, 5*B), constrained_layout=True)
    if B == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(B):
        for j in range(cols):
            axes[i, j].imshow(batch_data[i][j], cmap=cmaps[j], **vmin_vmax[j])
            axes[i, j].axis('off')
            if titles:
                axes[i, j].set_title(titles[i][j])
    return fig, axes

def save_figure(fig, save_path, im_for_colorbar=None, label=None):
    if im_for_colorbar is not None:
        fig.colorbar(im_for_colorbar, ax=fig.axes, location='right', shrink=0.85, pad=0.02, label=label)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
