"""LPIPS- and local-contrast-based pseudo-mask generation."""

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from skimage.morphology import skeletonize


__all__ = ["LPIPSPseudoLabeler", "cal_detection_mask"]


def _gaussian_kernel(kernel_size, sigma, device, dtype):
    coordinates = torch.arange(
        kernel_size,
        device=device,
        dtype=dtype,
    ) - kernel_size // 2
    gaussian = torch.exp(-(coordinates.square()) / (2 * sigma**2))
    gaussian = gaussian / gaussian.sum()
    return torch.outer(gaussian, gaussian)


def cal_detection_mask(first_image, second_image, window_size=11, sigma=1.5):
    """Return a per-pixel local contrast-similarity map in BCHW format."""
    if first_image.shape != second_image.shape:
        raise ValueError(
            "The two input tensors must have identical shapes, "
            f"got {tuple(first_image.shape)} and {tuple(second_image.shape)}."
        )
    if first_image.ndim == 3:
        first_image = first_image.unsqueeze(0)
        second_image = second_image.unsqueeze(0)
    elif first_image.ndim != 4:
        raise ValueError("Input tensors must have CHW or BCHW format.")
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd integer.")

    channels = first_image.shape[1]
    kernel = _gaussian_kernel(
        window_size,
        sigma,
        first_image.device,
        first_image.dtype,
    )
    window = kernel.view(1, 1, window_size, window_size).repeat(
        channels,
        1,
        1,
        1,
    )
    padding = window_size // 2

    first_mean = F.conv2d(
        first_image,
        window,
        padding=padding,
        groups=channels,
    )
    second_mean = F.conv2d(
        second_image,
        window,
        padding=padding,
        groups=channels,
    )
    first_variance = (
        F.conv2d(
            first_image.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - first_mean.square()
    ).clamp_min(0)
    second_variance = (
        F.conv2d(
            second_image.square(),
            window,
            padding=padding,
            groups=channels,
        )
        - second_mean.square()
    ).clamp_min(0)

    stability = 0.03**2
    contrast_similarity = (
        2
        * torch.sqrt(first_variance + 1e-8)
        * torch.sqrt(second_variance + 1e-8)
        + stability
    ) / (first_variance + second_variance + stability)
    return contrast_similarity.mean(dim=1, keepdim=True)


class LPIPSPseudoLabeler:
    """Create binary artifact masks from LPIPS and contrast similarity."""

    def __init__(
        self,
        backbone="vgg",
        thr=0.30,
        min_area=2000,
        dilate_iter=2,
        erode_iter=1,
        thin=False,
        device="cuda",
    ):
        if not 0 <= thr <= 1:
            raise ValueError("thr must be in [0, 1].")
        if min_area < 0 or dilate_iter < 0 or erode_iter < 0:
            raise ValueError("Morphology parameters cannot be negative.")

        self.threshold = thr
        self.min_area = min_area
        self.dilate_iter = dilate_iter
        self.erode_iter = erode_iter
        self.thin = thin
        self.device = torch.device(device)
        self.lpips_model = lpips.LPIPS(
            net=backbone,
            spatial=True,
        ).to(self.device).eval()
        self.morphology_kernel = cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (5, 5),
        )

    @torch.no_grad()
    def compute_heat_map(self, ground_truth, super_resolved):
        if ground_truth.shape != super_resolved.shape:
            raise ValueError(
                "ground_truth and super_resolved must have identical shapes."
            )
        if ground_truth.ndim != 4:
            raise ValueError("Images must have BCHW format.")

        lpips_map = self.lpips_model(
            super_resolved,
            ground_truth,
            normalize=True,
        ).clamp(0, 1)
        contrast_map = cal_detection_mask(
            ground_truth,
            super_resolved,
        )
        return contrast_map * (1 - lpips_map)

    @torch.no_grad()
    def __call__(self, ground_truth, super_resolved, thr=None):
        threshold = self.threshold if thr is None else thr
        if not 0 <= threshold <= 1:
            raise ValueError("thr must be in [0, 1].")

        heat_map = self.compute_heat_map(ground_truth, super_resolved)
        mask = (heat_map < threshold).float()
        return heat_map, self._postprocess_mask(mask)

    def _postprocess_mask(self, mask):
        mask_array = mask.squeeze(1).byte().cpu().numpy()
        processed_masks = []

        for single_mask in mask_array:
            if self.erode_iter:
                single_mask = cv2.erode(
                    single_mask,
                    self.morphology_kernel,
                    iterations=self.erode_iter,
                )
            if self.dilate_iter:
                single_mask = cv2.dilate(
                    single_mask,
                    self.morphology_kernel,
                    iterations=self.dilate_iter,
                )

            single_mask = ndimage.binary_fill_holes(
                single_mask > 0,
                structure=np.ones((3, 3)),
            ).astype(np.uint8)
            if self.min_area:
                component_count, labels, stats, _ = (
                    cv2.connectedComponentsWithStats(
                        single_mask,
                        connectivity=8,
                    )
                )
                cleaned_mask = np.zeros_like(single_mask)
                for component_index in range(1, component_count):
                    if (
                        stats[component_index, cv2.CC_STAT_AREA]
                        >= self.min_area
                    ):
                        cleaned_mask[labels == component_index] = 1
                single_mask = cleaned_mask

            if self.thin:
                single_mask = skeletonize(
                    single_mask.astype(bool)
                ).astype(np.uint8)
            processed_masks.append(single_mask)

        processed_array = np.stack(processed_masks)[:, None]
        return torch.from_numpy(processed_array).to(
            device=mask.device,
            dtype=mask.dtype,
        )
