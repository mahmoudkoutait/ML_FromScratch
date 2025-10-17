# this is a base abstract class for all ml models.
from abc import ABC, abstractmethod
import numpy as np

class BaseModel(ABC):
    '''abstract base class for ml models.'''

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict(self, x: np.ndarray) -> np.ndarray:
        pass
