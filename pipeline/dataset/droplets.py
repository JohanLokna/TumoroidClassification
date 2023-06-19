import json

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
from torch.utils.data import IterableDataset
from typing import Optional, Callable

from .image import ImageBatch
from . import DropletExtractor
from .. import BaseLogger, StandardLogger


class DropletDataset(IterableDataset):

    def __init__(
        self, data_dir : Path, 
        transform : Optional[Callable] = None,
        load_drug = False,
        log : Optional[BaseLogger] = StandardLogger()
    ):
        # Set up transform
        self.transform = transform

        # Load drugs
        self.load_drug = load_drug

        # Set up logger
        self.log = log

        # Set up data directory
        self.data_dir = data_dir
        self.meta_data = json.load(open(data_dir / "meta_data.json", "r"))
        self.indecies = [(i, j) for i, size in enumerate(self.file_sizes) for j in range(size)]

        # Set up iteration structure
        self.cached_idx = None
        self.cached_droplets_ = None
        self.cached_labels_ = None
        self.cached_drugs_ = None

    @property
    def data_paths(self):
        return self.meta_data["data_paths"]

    @property
    def file_sizes(self):
        return self.meta_data["file_sizes"]

    def load_into_cache(self, idx):
        
        # Load data from memory
        batch = torch.load(self.data_paths[idx])
        
        # Save into cache
        self.cached_idx = idx
        self.cached_droplets_ = batch["droplets"]
        self.cached_labels_ = batch["labels"]
        self.cached_drugs_ = batch["drugs"]

    def __len__(self):
        return sum(self.file_sizes)

    def __iter__(self):
        return (
            self[idx] for idx in self.indecies
        )

    @property
    def indices(self):
        return [(i, j) for i, size in enumerate(self.file_sizes) for j in range(size)]

    def __getitem__(self, idx):

        # Tuple specifies should be of shape (file_idx, img_idx)
        if isinstance(idx, tuple):
            file_idx, img_idx = idx
            return self.get_helper(file_idx, img_idx)
        
        elif isinstance(idx, int):
            file_idx, img_idx = self.indecies[idx]
            return self.get_helper(file_idx, img_idx)

        elif isinstance(idx, slice):
            return (self.get_helper(file_idx, img_idx) for file_idx, img_idx in self.indecies[idx])
              

    def get_helper(self, file_idx, img_idx):

        # Load file if not cached
        if (self.cached_idx is None) or file_idx != self.cached_idx:
            self.load_into_cache(file_idx)

        # Get specific data
        img = self.cached_droplets_[img_idx]
        label = self.cached_labels_[img_idx]
        drug = self.cached_drugs_[img_idx]

        # Transform if provided
        if self.transform is not None:
            img = self.transform(img)

        # Return data
        if self.cached_labels_ is not None:
            if self.load_drug:
                return img, label, drug
            else:
                return img, label

        else:
            if self.load_drug:
                return img, drug
            else:
                return img

    def add_metadata(self, meta_data : dict):
        self.meta_data.update(meta_data)
        json.dump(self.meta_data, open(self.data_dir / "meta_data.json", "w+"))


    @staticmethod
    def from_image_batch(
        batch: ImageBatch,
        data_dir: Path,
        scoring_dir: Path,
        extract_labels: bool = True,
        validator: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        outlier_detector: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        log: Optional[BaseLogger] = StandardLogger(),
    ):

        # Create data directory - should be an empty directory
        data_dir.mkdir(exist_ok=True, parents=True)
        if any(data_dir.iterdir()):
            log(f"{data_dir} is a non-empty directory upon dataset creation")

        # Data structures
        data_paths = []
        file_sizes = []

        # Logging information
        skipped_files = 0
        skipped_droplets = 0

        print("tqdm outputs image / number of images")
        for img in tqdm(batch):

            # Find path related to image
            img_path = img.img_path

            # Find related scoring files in self.scoring_dir
            matches = list(scoring_dir.glob(f"**/*{img_path.stem}*.xlsx"))

            # Log if not unique scoring file
            if len(matches) != 1 and (log is not None):
                log(f"{img_path} ignored - invalid number of related scoring files: {len(matches)}")
                continue

            # Extract images
            scoring_path = matches[0]

            # If any error happens on the whole image level, we skip the whole image.
            try:
                # Extract droplets and labels
                # Convert them to torch
                droplets, labels, drugs = DropletExtractor.extract_image(img, scoring_path, extract_labels)
                droplets = [torch.from_numpy(drop.astype(np.float32)) for drop in droplets]

                # Validate droplets
                if validator is not None and extract_labels is not False:
                    skipped_droplets += len(droplets)

                    # Keep drops which are flagged ok
                    flagged_ok = [False]*len(droplets)
                    for i in range(len(flagged_ok)):
                        drop = droplets[i]
                        try:                        # If an error occurs during validation, we remove the drop.
                            r = validator(drop)
                        except Exception as e:
                            r = False
                        flagged_ok[i] = r
                    droplets = [drop for i, drop in enumerate(droplets) if flagged_ok[i]]

                    labels = labels[flagged_ok]
                    drugs = drugs[flagged_ok]

                    skipped_droplets -= len(droplets)

                # Apply pre transform
                if pre_transform is not None:
                    droplets_curr, flagged_outlier = [], []
                    for i, drop in enumerate(droplets):
                        try:
                            droplets_curr.append(pre_transform(drop))
                        except Exception as e:
                            flagged_outlier.append(i)
                    
                    # Remove labels and drugs
                    flagged_ok = [i for i in range(len(labels)) if i not in flagged_outlier]
                    labels = labels[flagged_ok]
                    drugs = drugs[flagged_ok]

                    # Skip droplets where an exception occur
                    skipped_droplets += len(droplets) - len(droplets_curr)
                    droplets = droplets_curr

                # Outlier detection of droplets
                if outlier_detector is not None and extract_labels is not False:
                    skipped_droplets += len(droplets)

                    # Pad droplets
                    max_len = max([torch.numel(drop) for drop in droplets])
                    droplets_pad = [torch.zeros(max_len) for drop in droplets]
                    for i, drop in enumerate(droplets):
                        droplets_pad[i][:torch.numel(drop)] = drop.flatten()

                    # Drop drops which are flagged as outliers
                    flagged_outlier = outlier_detector.detect(droplets_pad, labels)
                    droplets = [drop for i, drop in enumerate(droplets) if i not in flagged_outlier]

                    flagged_ok = [i for i in range(len(labels)) if i not in flagged_outlier]
                    labels = labels[flagged_ok]
                    drugs = drugs[flagged_ok]

                    skipped_droplets -= len(droplets)

                # Save batch
                data_path = data_dir / (img.img_path.stem + ".pt")
                torch.save(
                    {"droplets": droplets, "labels": labels, "drugs": drugs},
                    data_path
                )

                # Append file to meta data
                data_paths.append(str(data_path))
                file_sizes.append(len(droplets))

            except Exception as e:
                print("image skipped")
                print(f"{img_path} ignored - error occured when extracting the droplet: {str(e)}")
                skipped_files += 1
                continue
        
        # Save metadata
        meta_data = {
            "data_paths": data_paths,
            "file_sizes": file_sizes
        }

        json.dump(meta_data, open(data_dir / "meta_data.json", "w+"))

        print(f"Created dataset: Skipped {skipped_files} files and filtered {skipped_droplets}")

        return DropletDataset(data_dir=data_dir, log=log, transform=transform)
