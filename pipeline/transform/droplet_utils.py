import torch
from typing import Tuple

from . import DropletTransform


class DropletSequence(DropletTransform):

    def __init__(self, *transforms : DropletTransform) -> None:
        super().__init__()
        self.transforms = transforms

    def transform(self, drop : torch.FloatTensor) -> torch.FloatTensor:
        for t in self.transforms:
            drop = t(drop)
        return drop


class DropletIdentity(DropletSequence):

    def __init__(self) -> None:
        super().__init__()


class DropletPadding(DropletTransform):

    def __init__(self, padded_size: Tuple[int]) -> None:
        super().__init__()
        self.padded_size = padded_size

    def transform(self, drop: torch.FloatTensor) -> torch.FloatTensor:
        '''
        Puts the droplet in the middle of a black canvas of size self.padded_size
        If padded_size is None, no padding is performed.
        Drops that exceed the canvas size throw an error.

        :param drop: The input droplet
        :return: The padded droplet
        '''
        if self.padded_size is None:
            return drop

        padded_drop = torch.zeros(self.padded_size, dtype=torch.float)
        pad_x = self.padded_size[0]
        img_x = drop.shape[0]
        left_x = pad_x - img_x
        start_x = left_x // 2
        end_x = start_x + img_x

        pad_y = self.padded_size[1]
        img_y = drop.shape[1]
        left_y = pad_y - img_y
        start_y = left_y // 2
        end_y = start_y + img_y
        try:
            padded_drop[start_x:end_x, start_y:end_y] = drop
        except Exception as e:
            raise RuntimeError("Droplet larger than padding size.")

        return padded_drop
