"""Types and data structures."""

from typing import Callable, TypeAlias

import numpy as np
from numpy.typing import NDArray

Array: TypeAlias = NDArray[np.float32]
MaxTrainIters: TypeAlias = int | Callable[[int], int]
