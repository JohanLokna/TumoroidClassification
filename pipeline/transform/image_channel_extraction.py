import numpy as np
from pathlib import Path
from typing import Optional, List

from . import ImageTransform
from .. import Image, ImageBatch


class ImageChannelExtraction(ImageTransform):

    def __init__(self, channels: List[str], out_dir: Optional[Path] = None) -> None:
        super().__init__(out_dir=out_dir)
        self.keep_channels = channels

    def transform(self, img: np.ndarray, **kwargs) -> np.ndarray:
        '''
        Removes all channels besides those specified in the instance of the Extractor.
        If keep_channels is None, all channels are kept

        :param img: Current image being transformed
        :param kwargs: Arguments including all meta information (calculated in local_pre_transform)
        :return: A (len(self.keep_channels, x, y) dimensional ndarray with only specified dimensions left.
        '''
        if self.keep_channels is None:
            return img
        else:
            keep_indices = [kwargs['channelNames'].index(channel) for channel in self.keep_channels]
            return img[keep_indices, :, :]

    def local_pre_transform(self, img: Image) -> dict:
        '''
        A way to access meta information of a single image

        :param img: The current image being processed
        :return: The meta information of the image.
        '''
        return img.meta

    def post_processing(self, img_path: Path, meta: dict) -> None:
        '''
        Updates the meta information of the image after removing the channels to the new channels

        :param img_path: Path of the transformed image
        :param meta: Old meta dict, obtained by local_pre_transform
        '''
        if self.keep_channels is not None:
            meta['channelNames'] = self.keep_channels
            meta['channelCount'] = len(self.keep_channels)
            img = Image(img_path)
            img.meta = meta
