import numpy as np
from pathlib import Path
from typing import Optional

from . import ImageTransform
from .. import ImageBatch


class ImageSequence:
    '''
    Class that applies multiple ImageTransformations in a row.
    '''
    def __init__(self, *transforms: ImageTransform, out_dir: Optional[Path] = None) -> None:
        self.transforms = transforms

    def __call__(self, batch: ImageBatch) -> ImageBatch:
        for t in self.transforms:
            batch = t(batch)
        return batch


class ImageIdentity(ImageSequence):

    def __init__(self, out_dir : Optional[Path] = None) -> None:
        super().__init__(out_dir=out_dir)
