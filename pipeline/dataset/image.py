from pathlib import Path
import nd2
import json
import numpy as np

from .. import StandardLogger, MetaExtractor


class Image:

    def __init__(self, img_path : Path) -> None:
        self.img_path = img_path

    @property
    def img(self) -> np.ndarray:
        if self.img_path.suffix == ".nd2":
            return nd2.imread(self.img_path)
        elif self.img_path.suffix == ".npy":
            return np.load(self.img_path)
        else:
            raise RuntimeError("Unknown image format")

    @property
    def meta(self) -> dict:
        '''
        Returns the meta data of the given image.

        :return: Meta information as a dictionary {num_channels: int, channels: list[str]}
        '''
        if self.img_path.suffix == ".nd2":
            return MetaExtractor.extract_meta(self.img_path)
        elif self.img_path.suffix == ".npy":
            meta_path = MetaExtractor.meta_path_of_npy(self.img_path)
            with meta_path.open('r') as meta_file:                # If an error occurs here, the meta was not calculated
                meta = json.load(meta_file)
            return meta
        else:
            raise RuntimeError("Unknown image format")

    @meta.setter
    def meta(self, data: dict) -> None:
        '''
        Sets the meta data of the given image.
        Only works for numpy array images
        '''
        if self.img_path.suffix == ".nd2":
            raise RuntimeError("Cannot change META of .nd2 file")
        elif self.img_path.suffix == ".npy":
            meta_path = MetaExtractor.meta_path_of_npy(self.img_path)
            with meta_path.open('w+') as meta_file:
                json.dump(data, meta_file)
        else:
            raise RuntimeError("Unknown image format")

    @staticmethod
    def make_image(img, img_path):
        np.save(open(img_path, "wb+"), img)

        if img_path.suffix != ".npy":
            StandardLogger()(f"Recieved filepath with suffix not matching npy-file format: {img_path}")

        return Image(img_path)


class ImageBatch():

    def __init__(self, *img_paths : str) -> None:
        self.img_list = [Image(p) for p in img_paths]

    def __iter__(self):
        return self.img_list.__iter__()

    def __len__(self):
        return len(self.img_list)

    def union(self, other: "ImageBatch") -> None:
        self.img_list += other.img_list

    @staticmethod
    def from_dir(img_dir: Path):
        return ImageBatch(*img_dir.glob("**/*.nd2"))
