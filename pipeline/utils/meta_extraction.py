import nd2
import json
from pathlib import Path
from typing import Optional


class MetaExtractor:
    suffix = "_meta.json"

    def __init__(self, out_dir: Optional[Path] = None) -> None:
        '''
        :param out_dir: Directory in which the meta information is saved. Strongly recommended to use the same as for Trafos.
        '''
        self.out_dir = out_dir

    def __call__(self, img_path: Path) -> None:
        '''
        Extracts the meta information from the img at img_path and saves the information in self.out_dir

        :param img_path: Path to the image in question
        :return: No Return Value. The Meta information is saved in out_dir / (img_path.stem + suffix).
        '''
        if img_path.suffix != ".nd2":                           # We only extract from nd2, npy is already calced
            return

        meta = MetaExtractor.extract_meta(img_path)             # Extracts the data from the file

        meta_path = self.meta_path(img_path)                    # Save meta data to the path
        with meta_path.open('w+') as meta_file:
            json.dump(meta, meta_file)

    def extract_batch(self, batch) -> None:
        '''
        Extracts the meta from a whole batch by calling the __call__ method on each image in the batch
        :param batch: ImageBatch from which the meta data should be extracted.
        :return: Nothing, meta files are saved in the directory out_dir.
        '''
        for img in batch:
            self(img.img_path)

    def meta_path(self, img_path: Path) -> Path:
        """
          Given an Image, calculate the path where its meta data should be saved and creates the directory if necessary
          If no out_dir is given, the directory of the image is used.
        """
        if self.out_dir is None:
            out_path = MetaExtractor.meta_path_of_npy(img_path)

        else:
            self.out_dir.mkdir(exist_ok=True, parents=True)
            out_path = self.out_dir / (img_path.stem + MetaExtractor.suffix)

        return out_path

    @staticmethod
    def meta_path_of_npy(img_path: Path) -> Path:
        '''
        This assumes the out_dir of the Metaextractor is the same as ImageTrafos.

        :param img_path: Path to the img.npy file
        :return: The meta file path assuming it is in the same dictionary.
        '''
        return img_path.with_name(img_path.stem.__str__() + MetaExtractor.suffix)

    @staticmethod
    def extract_meta(img_path: Path) -> dict:
        '''
        Extracts all necessary available information from the image at img_path.

        :param img_path:
        :return: meta information as dict {attr: val}
        '''
        name_meta = MetaExtractor.meta_from_name(img_path)                      # Meta manually parsed from filename

        with nd2.ND2File(img_path) as file:                                     # Meta saved in the nd2 file
            whole_meta = file.metadata                                          # Huge Metafile from image
            channelCount = whole_meta.contents.channelCount
            channelNames = [''] * channelCount
            for channel in whole_meta.channels:
                channelNames[channel.channel.index] = channel.channel.name

            specific_meta = {"channelCount": channelCount, "channelNames": channelNames}

        return dict(specific_meta, **name_meta)

    @staticmethod
    def meta_from_name(img_path: Path) -> dict:
        '''
        Manually extraction of meta data from name. Assumes a certain naiming convention

        :param img_path: Image Path. It assumes that the name contains "_TX_" where X is the day at which the img is taken.
        :return: Return the dict with informations.
        '''
        stem = img_path.stem
        parts = stem.__str__().split('_')
        meta = {}
        meta["manual"] = False
        for part in parts:
            if part.startswith('Split'):
                meta["manual"] = True
            elif part.__len__() == 2 and part.lower().startswith('t'):
                meta["day"] = int(part[1]) - 1

        return meta
