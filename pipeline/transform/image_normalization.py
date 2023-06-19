import itertools
import gc
import numpy as np
from pathlib import Path
import pickle
from typing import Optional, Dict, Tuple, List
from scipy.interpolate import interp1d

from . import ImageTransform
from .. import ImageBatch


# Normalizes Image Intensities throughout a Batch using Nyul's Method.
# This is done by equalizing all histograms to a "standard histogram" computed from the data. It therefore takes batch
#   statistics into account instead of equalizing to some independent histogram
# Method adapted from source below to fit our datatypes.
# https://github.com/sergivalverde/MRI_intensity_normalization/blob/master/nyul.py
# Input:    min / max_percentile    ignore everything below / above these threshold quantile histograms
#           min / max_intensity     min / max intensity in the resulting image
#           l_quantile:step:u_quantile  Other landmarks taken into consideration
#           hist_dir                During Training: standard hist / quantiles will be saved there.
#                                   During Usage: standard hist / quantiles used for training will be loaded from there.
#                                   Technically: It will be loaded if it exists in this dict with the naming convention
class ImageIntensityNormalization(ImageTransform):

    def __init__(
        self, out_dir: Optional[Path] = None, hist_dir: Optional[Path] = None,
        min_quantile: float = 0.02, max_quantile: float = 0.98, 
        min_intensity: float = 0, max_intensity: float = 1,
        l_quantile: float = 0.1, u_quantile: float = 0.9, step: float = 0.1
    ) -> None:
        super().__init__(out_dir=out_dir)
        self.hist_dir = hist_dir if hist_dir is not None else Path(".")
        self.min_quantile = min_quantile
        self.max_quantile = max_quantile
        self.min_intensity = min_intensity
        self.max_intensity = max_intensity
        self.l_quantile = l_quantile
        self.u_quantile = u_quantile
        self.step = step

    # We allign each image to the batch standard histogram depending on its number of channels.
    def global_pre_transform(self, batch: ImageBatch) -> dict:

        # Compute two kwargs: standard_scales and quantiles
        stats = {"standard_scales": {}, "quantiles": {}}

        # Group images after number of channels
        for num_channels, same_channel_count in itertools.groupby(batch, lambda img: img.img.shape[0]):

            # Get statistcs for each group
            batch_same_count = ImageBatch(*(img.img_path for img in same_channel_count))
            standard_scales, quantiles = self.get_standard_histogram(batch_same_count, num_channels)

            # Append them to overall statistics
            stats["standard_scales"][num_channels] = standard_scales
            stats["quantiles"][num_channels] = quantiles

        return stats

    def transform(
        self, img: np.ndarray,
        standard_scales: List,
        quantiles: List
    ) -> np.ndarray:

        # Set up output image
        tot_channels = img.shape[0]
        transformed_image = np.empty_like(img).astype(np.float32)

        # Get info from global pretransform arguments
        quantiles = quantiles[tot_channels]
        standard_scales = standard_scales[tot_channels]

        # Transforming each Channel on its own
        for n_channel in range(tot_channels):
            curr_channel = img[n_channel, :, :]
            landmarks = ImageIntensityNormalization.get_landmarks(curr_channel, quantiles)

            f = interp1d(
                landmarks, standard_scales[n_channel], kind="linear",
                fill_value=(standard_scales[n_channel][0], standard_scales[n_channel][-1]), 
                bounds_error=False
            )

            # Apply transformation to input image
            transformed_image[n_channel, :, :] = f(curr_channel)

        return transformed_image

    def get_standard_histogram(self, batch: ImageBatch, num_channels: int) -> Tuple[Dict, np.ndarray]:
        """
          Get standard histogram
        """

        hist_path = self.get_histpath(num_channels)

        # Calculate batch standard histogram for each channel or loads the previous one if it exists
        # If a hist_dir is given, save the file there.
        if not hist_path.exists():

            # Compute statistics
            standard_scales, quantiles = self.compute_standard_histogram(batch, num_channels)

            # Create directory and save stats
            self.hist_dir.mkdir(exist_ok=True, parents=True)
            with hist_path.open('wb') as hist_file:
                pickle.dump((standard_scales, quantiles), hist_file)
        
        else:

            # Load stats as already computed
            with hist_path.open('rb') as hist_file:
                standard_scales, quantiles = pickle.load(hist_file)

        return standard_scales, quantiles

    def compute_standard_histogram(self, batch: ImageBatch, num_channels: int) -> Tuple[Dict, np.ndarray]:
        """
          Calculates a standard histogram based on the i_min, l_percentile : u_percentile : step, i_max quantiles
          The standard histogram takes values in [min_intensity, max_intensity]
        """

        quantiles = np.concatenate((
            [self.min_quantile], 
            np.arange(self.l_quantile, self.u_quantile + self.step / 10, self.step), 
            [self.max_quantile]
        ))
        standard_scales = {}

        # We calculate the standard hist for each channel separately
        for n_channel in range(num_channels):
            standard_scale = np.zeros(len(quantiles))

            # Process each image in order to build the standard scale
            for i, img in enumerate(batch, start=1):
                img_data = img.img[n_channel, :, :]

                landmarks = ImageIntensityNormalization.get_landmarks(img_data, quantiles)
                min_p = np.quantile(img_data, self.min_quantile)
                max_p = np.quantile(img_data, self.max_quantile)

                # Interpolating function
                f = interp1d([min_p, max_p], [self.min_intensity, self.max_intensity])

                # Interpolate landmarks
                landmarks = np.array(f(landmarks))

                # Iteratively calculate the mean
                standard_scale = (standard_scale * (i - 1) + landmarks) / i

                # Fixes some RAM issues
                del img_data, f, img
                gc.collect()

            standard_scales[n_channel] = standard_scale

        return standard_scales, quantiles

    def get_histpath(self, num_channels: int):
        return self.hist_dir / ("standard_hist_" + str(num_channels) + "channels.pkl")

    # Gets the Landmarks for Nyul's Method. Extra method to guarantee the usage of same procedure
    @staticmethod
    def get_landmarks(img: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
        landmarks = np.quantile(img, quantiles)
        return landmarks
