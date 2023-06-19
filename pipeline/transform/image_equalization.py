import numpy as np
from pathlib import Path
from typing import Optional
from skimage import exposure

from . import ImageTransform


# Equalizes Image Intensities of each image to roughly a normal distribution with mean 1/2.
# Input:    adapt                   If True, a more complex method that takes local information into account is used
#           min_percentile          ignore everything below / above the q , (1-q) quantile
#           min / max_intensity     min / max intensity in the resulting image
class ImageIntensityEqualization(ImageTransform):

    def __init__(self, 
                out_dir: Optional[Path] = None, adaptive: bool = False, 
                min_quantile: float = 0.02, min_intensity: float = 0, max_intensity: float = 1
    ) -> None:
        super().__init__(out_dir=out_dir)
        self._adaptive = adaptive
        self._min_quantile = min_quantile
        self._min_intensity = min_intensity
        self._max_intensity = max_intensity

    def transform(self, img: np.ndarray) -> np.ndarray:
        adapted_image = np.empty_like(img).astype(np.float32)
        for n_channel in range(img.shape[0]):
            if self._adaptive:
                adapted_image[n_channel] = exposure.equalize_adapthist(img[n_channel, :, :],
                                                                       clip_limit=self._min_quantile)
            else:
                adapted_image[n_channel] = exposure.equalize_hist(img[n_channel, :, :])

        return adapted_image * (self._max_intensity - self._min_intensity) + self._min_intensity
