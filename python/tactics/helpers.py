from __future__ import annotations
import functools, operator
from typing import Iterable, Union, TypeVar

T = TypeVar("T")
U = TypeVar("U")
def prod(x: Iterable[T]) -> Union[T, int]:
    return functools.reduce(operator.mul, x, 1)