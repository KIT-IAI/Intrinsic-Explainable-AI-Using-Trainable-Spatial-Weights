import numpy as np


class BaseDataset:
    """
    Base dataset class used for all datasets.
    All dataset should initialize themself
    and should be accessed via array access.
    """

    def __getitem__(self, index) -> np.ndarray:
        raise NotImplementedError("Datasets should implement __getitem__.")

    def __len__(self) -> int:
        raise NotImplementedError("Datasets should implement __len__.")