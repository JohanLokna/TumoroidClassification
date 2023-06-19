# import pytorch_lightning as pl
from .droplets import DropletDataset
from torch.utils.data import random_split, DataLoader 
from typing import Optional, Callable
import torch 


# class DropletDataModule(pl.LightningDataModule):

class DropletDataModule(torch.nn.Module):
    def __init__(
        self, 
        collate_fn: Callable, 
        data_dir: str = "path/to/dir", 
        batch_size: int = 32, 
        transform: Optional[Callable]=None, 
        tts: float=.05, 
        seed: Optional[int]=0, 
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transform = transform
        self.collate_fn = collate_fn
        self.seed = seed
        self.tts = tts
        self.setup(tts)

    def setup(self, tts: float=.05):
        full_dataset = DropletDataset(self.data_dir, transform=self.transform)
        train_size = int( len(full_dataset) * (1-tts) )
        val_size = len(full_dataset) - train_size
        self.train_data, self.val_data = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(self.seed))

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_data, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def collate_fn(batch):
        print(batch)
        x = [item[0] for item in batch]
        y = [item[1] for item in batch]
        return x, y
