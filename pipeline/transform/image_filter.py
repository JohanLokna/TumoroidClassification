from abc import ABC, abstractmethod
from typing import Optional

from .. import ImageBatch, Image, BaseLogger, PrintLogger, StandardLogger


class ImageFilter(ABC):

    def __init__(self, reason: str = "unknown", log: Optional[BaseLogger] = PrintLogger()) -> None:
        super().__init__()
        self.reason = reason
        self.log = log

    def __call__(self, batch: ImageBatch) -> ImageBatch:

        new_batch_paths = []
        filtered = 0

        for img in batch:

            # Append image to new batch
            if not self.filter(img):
                new_batch_paths.append(img.img_path)
            else:
                filtered += 1

        if filtered != 0 and (self.log is not None):
            self.log(f"{filtered} out of {len(batch)} images filtered from dataset. Reason: {self.reason}")

        return ImageBatch(*new_batch_paths)

    @abstractmethod
    def filter(self, img: Image) -> bool:
        '''
        :param img: The input Image
        :return: Whether to remove the file or not. False -> The image is kept, True -> The image is filtered.
        '''
        pass


class ImageFilterDayZero(ImageFilter):

    def __init__(self, log: Optional[BaseLogger] = PrintLogger()) -> None:
        super().__init__(reason="Images were taken on day Zero and DayZero filtering is activated.", log=log)

    def filter(self, img: Image) -> bool:
        return img.meta["day"] > 0


class ImageFilterManual(ImageFilter):

    def __init__(self, log: Optional[BaseLogger] = PrintLogger()) -> None:
        super().__init__(reason="Manual splits were discovered and ManualSplit filtering is activated.", log=log)

    def filter(self, img: Image) -> bool:
        return img.meta["manual"]


class FilterSequence:
    '''
    Class that applies multiple ImageTransformations in a row.
    '''
    def __init__(self, *filters: ImageFilter) -> None:
        self.filters = filters

    def __call__(self, batch: ImageBatch) -> ImageBatch:
        for fil in self.filters:
            batch = fil(batch)
        return batch
