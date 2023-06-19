from abc import ABC, abstractmethod
import torch
from typing import Callable, Optional

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np

# abstract classes

class DropletDetector(ABC):
    def __init__(self)->None:
        super().__init__()

    @abstractmethod
    def detect(self):
        pass

# Outlier classes

class DropletOutlierDetector(DropletDetector):
    def __init__(self)->None:
        super().__init__()

    @abstractmethod
    def detect(self, drops: list, labels: Optional[list])->list:
        """
        Returns list of indices of drops considered outliers.
        """
        pass

class DropletOutlierDetectorSequence(DropletOutlierDetector):
    def __init__(self, *detectors : DropletDetector) -> None:
        super().__init__()
        self.detectors = detectors
        self.compstyle = 'union'

    def set_compstyle(self, compstyle: str):
        allowed_styles = ['union', 'intersection']
        if not compstyle in allowed_styles :
            raise RuntimeError(f"compstyle argument to DropletOutlierSequence not in"+str(allowed_styles))
        else: 
            self.compstyle = compstyle

    def detect(self, drops: list, labels: Optional[list]) -> list:
        """
        Returns the union of outliers detected by the detectors in the sequence.
        """
        outliers = []
        for det in self.detectors:
            if self.compstyle == 'union':
                    outliers = list(set(outliers).union(set(det.detect(drops, labels))))
            if self.compstyle == 'intersection':
                    outliers = list(set(outliers).intersection(set(det.detect(drops, labels))))
        return  outliers

class DropletOutlierIsolationForest(DropletOutlierDetector):
    def __init__(
        self, 
        n_estimators: Optional[int] = 100, 
        n_jobs: Optional[int] = None, 
        contamination: Optional[float] = 'auto'
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.sklearnforest = IsolationForest(
            n_estimators=self.n_estimators, 
            n_jobs=self.n_jobs, 
            contamination=self.contamination
        )

    def update_params():
        pass

    def detect(self, drops: list, labels: Optional[list])-> list:
        """
        drops: a list of torch.Tensors containing the training data, from which the outliers are to be detected.
        """
        npdrops = [torch.flatten(drop).numpy() for drop in drops]
        self.sklearnforest.fit(npdrops)
        preds = self.sklearnforest.predict(npdrops)
        outliers = []
        for i in range(len(preds)):
            if preds[i]==-1:
                outliers.append(i)
        return outliers

class DropletOutlierLocalFactor(DropletOutlierDetector):
    def __init__(
        self, 
        n_neighbors: Optional[int] = 20, 
        algorithm: Optional[str] = 'auto',
        metric: Optional[str] = 'minkowski',
        p: Optional[int] = 2,
        n_jobs: Optional[int] = None, 
        contamination: Optional[float] = 'auto'
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.p = p
        self.n_jobs = n_jobs
        self.contamination = contamination
        self.sklearnLOF = LocalOutlierFactor(
            n_neighbors=self.n_neighbors, 
            algorithm=self.algorithm,
            metric = self.metric,
            p = self.p,
            n_jobs=self.n_jobs, 
            contamination=self.contamination
        )

    def update_params():
        pass 

    def detect(self, drops: list, labels: Optional[list])-> list:
        """
        drops: a list of torch.Tensors containing the training data, from which the outliers are to be detected.
        """
        npdrops = [torch.flatten(drop).numpy() for drop in drops]
        preds = self.sklearnLOF.fit_predict(npdrops)
        outliers = []
        for i in range(len(preds)):
            if preds[i]==-1:
                outliers.append(i)
        return outliers


# Anomaly classes

class DropletAnomalyDetector(DropletDetector):
    def __init__(self)->None:
        super().__init__()

    @abstractmethod
    def detect(self, drops: list, newdrop: torch.FloatTensor) -> bool:
        """
        If returns true, newdrop is considered an anomaly.
        """
        pass

    def detect_batch(self, drops: list, newdrops: list) -> list:
        return [self.detect(drops, newdrop) for newdrop in newdrops]


class DropletAnomalyDetectorSequence(DropletAnomalyDetector):
    def __init__(self, *detectors : DropletDetector) -> None:
        super().__init__()
        self.detectors = detectors

    def detect(self, drops: list, newdrop: torch.FloatTensor) -> bool:
        return all([v.detect(drops, newdrop) for v in self.detectors])


class DropletAnomalyIsolationForest(DropletAnomalyDetector):
    def __init__(
        self, 
        n_estimators: Optional[int] = 100, 
        n_jobs: Optional[int] = None, 
    ):
        super().__init__()
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.sklearnforest = IsolationForest(
            n_estimators=self.n_estimators, 
            n_jobs=self.n_jobs
        )

    def update_params():
        pass

    def detect(self, drops: list, newdrop: torch.FloatTensor)-> bool:
        """
        drops: a list of torch.Tensors containing the training data considered normal
        newdrop: a torch.Tensor containing the new drop to be investigated.
        """
        npdrops = [torch.flatten(drop).numpy() for drop in drops] + [torch.flatten(newdrop).numpy()]
        self.sklearnforest.fit(npdrops)
        return (self.sklearnforest.predict(npdrops)[-1] == -1)
