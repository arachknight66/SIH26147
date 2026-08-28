from __future__ import annotations
from typing import Sequence, TypeVar
import numpy as np

T = TypeVar("T")

def partition_frames(
    items: Sequence[T],
    selection_ratio: float = 0.70,
) -> tuple[list[T], list[T]]:
    """
    Partition sequence into selection (training) and validation (held-out) subsets.

    Parameters
    ----------
    items : Sequence[T]
    selection_ratio : float

    Returns
    -------
    selection_set : list[T]
    validation_set : list[T]
    """
    n = len(items)
    if n <= 1:
        return list(items), list(items)
    split = max(1, int(n * selection_ratio))
    return list(items[:split]), list(items[split:])
