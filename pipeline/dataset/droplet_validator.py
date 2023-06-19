from abc import ABC, abstractmethod
import torch
from typing import Callable


class DropletValidator(ABC):

    def __init__(self) -> None:
        super().__init__()
    
    def __call__(self, drop : torch.FloatTensor) -> bool:
        return self.validate(drop)
    
    @abstractmethod
    def validate(self, drop : torch.FloatTensor) -> bool:
        pass


class ValidatorSequence(DropletValidator):

    def __init__(self, *validators : DropletValidator) -> None:
        super().__init__()
        self.validators = validators

    def validate(self, drop: torch.FloatTensor) -> bool:
        return all([v(drop) for v in self.validators])


class AnythingGoesValidator(ValidatorSequence):

    def __init__(self) -> None:
        super().__init__()


class NonEmptyValidator(DropletValidator):

    def validate(self, drop: torch.FloatTensor) -> bool:
        return torch.norm(drop).item() > 0


class TransformValidator(DropletValidator):

    def __init__(self, transform : Callable) -> None:
        super().__init__()
        self.transform = transform
        self.bad_ones = []

    def validate(self, drop: torch.FloatTensor) -> bool:
        
        # Test if transform yields an error
        try:
          self.transform(drop)
        except:
          
          self.bad_ones.append(drop)

          return False
      
        # Otherwise let through
        return True
