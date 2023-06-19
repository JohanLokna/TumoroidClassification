from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
from typing import Optional, Dict

from .. import ImageBatch, Image


class ImageTransform(ABC):

    def __init__(self, out_dir: Optional[Path] = None ) -> None:
        super().__init__()
        self.out_dir = out_dir

    def __call__(self, batch: ImageBatch) -> ImageBatch:
        
        new_batch_paths = []

        # Get global arguments over batch - e.g. statistics
        gloabl_kwargs = self.global_pre_transform(batch)

        for img in batch:

            local_kwargs = self.local_pre_transform(img)    # Gets information from the Image, e.g. Meta
            out_path = self.get_outpath(img.img_path)       # Calculates where the image should be saved

            # Save transformed image to the path
            Image.make_image(
                self.transform(img.img, **gloabl_kwargs, **local_kwargs),
                out_path
            )

            self.post_processing(out_path, local_kwargs)    # Post Processing with access to out_path and local args

            # Append image to new batch
            new_batch_paths.append(out_path)

        return ImageBatch(*new_batch_paths)

    def get_outpath(self, img_path: Path) -> Path:
        """
          Given an Image, calculate the path where it should be saved and creates the directory if necessary
        """

        if self.out_dir is None:
            out_path = img_path.with_suffix(".npy")

        else:
            self.out_dir.mkdir(exist_ok=True, parents=True)
            out_path = self.out_dir / (img_path.stem + ".npy")

        return out_path

    def global_pre_transform(self, batch: ImageBatch) -> Dict:
        return {}

    def local_pre_transform(self, img: Image) -> dict:
        return {}

    def post_processing(self, img_path: Path, local_kwargs: dict) -> None:
        return None

    @abstractmethod
    def transform(self, img: np.ndarray, **kwargs) -> np.ndarray:
        pass
