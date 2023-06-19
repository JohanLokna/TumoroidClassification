import torch 
import torch.nn as nn
from torch.utils.data import DataLoader 
from typing import Callable
from .Metric import Metric
from .Logging import *
from tqdm.auto import tqdm
from typing import Optional, List
from pathlib import Path 

class Trainer(nn.Module):
    def __init__(
        self, 
        model: nn.Module, 
        name: str, 
        data_pipeline_train: nn.Module, 
        data_pipeline_val: nn.Module, 
        criterion: Callable, 
        optimizer: torch.optim.Optimizer, 
        device: str, 
        metrics: List[Metric],
        keep_epochs: bool=False,
        verbose: bool=True, 
        use_wandb: bool=True,
        validate_every: int=-1,
        wandb_args={}, 
        directory: Optional[str]=None,
    ):
        super().__init__()
        self.model = model
        self.data_pipeline_train = data_pipeline_train
        self.data_pipeline_val = data_pipeline_val
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.validate_every = validate_every
        self.verbose = verbose
        self.train_logger = TrainLogger(self, metrics=metrics, use_wandb=use_wandb, keep_epochs=keep_epochs, **wandb_args)
        self.directory = directory
        self.name = name
    
    def to_device(self, d: dict):
        for k, v in d.items():
            if isinstance(v, torch.Tensor):
                d[k] = v.to(self.device)
        return d

    def forward_item(self, d: dict):
        """
        Get logits (model output) and labels from a dictionary d provided by a data loader
        """
        d = self.to_device(d)
        d = self.data_pipeline_train(d)
        x, y = d['x'], d['y'].long()
        logits = self.model(x)
        return logits, y

    def validate(
        self, 
        epoch_logger: EpochLogger, 
        val_loader: DataLoader, 
        i: int
    ):
        """
        Perform validation and log it to the epoch_logger
        """
        
        with torch.no_grad():
            self.model.eval()
            for d_v in val_loader:
                y_v_logits, y_v = self.forward_item(d_v)
                y_v_pred = torch.sigmoid(y_v_logits)
                epoch_logger.register_val(i, y_v_pred, y_v) 
            self.model.train()

        return epoch_logger.val_report(i)

    def step(self, d):
        logits, y = self.forward_item(d)
        y_pred = torch.sigmoid(logits)
        self.optimizer.zero_grad()
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()
        return loss

    def train(
        self, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        epochs: int=1,
    ):
        """
        Train for some epochs using the train and validation loaders
             
        """
        self.train_logger.start_training()
        validate_every = self.validate_every
        best_val_score = torch.inf 

        for epoch in range(epochs):
            old_val_score = best_val_score
            epoch_logger = self.train_logger.start_epoch(epoch)

            for i, d in tqdm(enumerate(train_loader), total=len(train_loader)):

                # optimization step
                loss = self.step(d)
                epoch_logger.train_update(loss)

                # reporting etc
                if i % validate_every == validate_every-1: 
                    epoch_logger.train_report()
                    loss = self.validate(epoch_logger, val_loader, i)
                    best_val_score = min([loss, best_val_score])
                    self.train_logger.wandb_report()

            # Validate at the end of every epoch
            epoch_logger.train_report()
            loss = self.validate(epoch_logger, val_loader, validate_every)
            best_val_score = min([loss, best_val_score])
            self.train_logger.wandb_report()
            
            if best_val_score < old_val_score: # save model if it has the best loss this far
                output_path = Path(self.directory)
                self.train_logger.save_epoch_stats(epoch_logger, f'{self.directory}/stats.json')
                if not output_path.exists(): output_path.mkdir(parents=True, exist_ok=True)
                torch.save(self.model, f'{self.directory}/{self.name}.pth')
            
            epoch_logger.finish_epoch(train_loader) # report final loss for this epoch
            self.train_logger.wandb_report()
            self.train_logger.finish_epoch() # report to wandb etc 


  

