from abc import ABC, abstractmethod
import torch


class DropletTransform(ABC):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, drop : torch.FloatTensor) -> torch.FloatTensor:
        return self.transform(drop)
    
    @abstractmethod
    def transform(self, drop : torch.FloatTensor) -> torch.FloatTensor:
        pass
