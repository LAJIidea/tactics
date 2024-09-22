from __future__ import annotations
import dataclasses
from typing import List, Tuple, Callable, Optional, ClassVar, Type, Union, Sequence, Dict
import numpy as np
from tactics._c import Tensor as Itensor
from tactics.shape.node import sint, Variable
from tactics.helpers import prod
import math

ConstType = Union[float, int, bool]

class Tensor:
    """
    A `Tensor` is a multi-dimensional matrix containing elements of a single data type.
    """
    training: ClassVar[bool] = False
    no_grad: ClassVar[bool] = False

    def __init__(self, data: Union[None, ConstType, List, Tuple, np.ndarray, bytes], 
                 shape: List = None, requires_grad: Optional[bool] = None) -> Tensor:
        # tensor can have gradients if you have called .backward
        self.grad: Optional[Tensor] = None

        self.requires_grad: Optional[bool] = requires_grad

        # create a core tensor from the different types of inputs
        if isinstance(data, np.ndarray):
            if shape:
                self.shape = shape
                self._tensor = Itensor(shape, data)
            else:
                self.shape = data.shape
                self._tensor = Itensor(self.shape, data)
        elif isinstance(data, (list, tuple)):
            pass
    
    def __repr__(self) -> str:
        return self._tensor.dump()
    
    def __hash__(self) -> int:
        return id(self)

    def __bool__(self):
        raise TypeError("__bool__ on Tensor is not defined")

    def __len__(self):
        if not self.shape:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]
    
    @property
    def shape(self) -> Tuple[sint, ...]:
        return self.shape
    
    @property
    def ndim(self) -> int:
        return len(self.shape)
    
    def numel(self) -> sint:
        return prod(self.shape)
    
    def assign(self, x) -> Tensor:
        pass

    def detach(self) -> Tensor:
        pass

    def numpy(self) -> np.ndarray:
        pass

    @staticmethod
    def empty(*shape, **kwargs):
        pass

    @staticmethod
    def full(shape: Tuple[sint, ...], fill_value: ConstType, **kwargs) -> Tensor:
        pass

    @staticmethod
    def zeros(shape, dtype=np.float32) -> Tensor:
        data = np.zeros(shape, dtype)
        return Tensor(data=data, shape=shape)

    @staticmethod
    def ones(shape, dtype=np.float32) -> Tensor:
        data = np.ones(shape, dtype)
        return Tensor(data=data, shape=shape)
    
    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        pass

    @staticmethod
    def glorot_uniform(*shape, **kwargs) -> Tensor:
        pass

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs) -> Tensor:
        pass
    
    def reshape(self, shape, *args) -> Tensor:
        pass

    def expand(self, shape, *args) -> Tensor:
        pass

    def add(self, x: Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def sub(self, x: Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def mul(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def div(self, x:Union[Tensor, ConstType], reverse=False, upcast=True) -> Tensor:
        pass

    def xor(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def bitwise_and(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def bitwise_or(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def lshift(self, x: int) -> Tensor:
        assert isinstance(x, int) and x >= 0, f"not supported lshift with {x=}"
        return self.mul(2 ** x)
    
    def rshift(self, x: int) -> Tensor:
        pass
    
    def pow(self, x:Union[Tensor, ConstType], reverse=False) -> Tensor:
        pass

    def sum(self, axis:Optional[Union[int, Sequence[int]]]=None, keepdim=False, acc_dtype=None):
        pass

    def logical_not(self):
        pass

    def neg(self):
        pass

    def not_eq(x):
        return False

    def less_than(x):
        return False

    def greate_than(x):
        return False

    def __neg__(self) -> Tensor: return self.neg()

    def __add__(self, x) -> Tensor: return self.add(x)
    def __sub__(self, x) -> Tensor: return self.sub(x)
    def __mul__(self, x) -> Tensor: return self.mul(x)
    def __pow__(self, x) -> Tensor: return self.pow(x)
    def __truediv__(self, x) -> Tensor: return self.div(x)
    def __floordiv__(self, x) -> Tensor: return self.div(x, upcast=False)
    def __matmul__(self, x) -> Tensor: return self.matmul(x)
    def __and__(self, x) -> Tensor: return self.bitwise_and(x)
    def __or__(self, x) -> Tensor: return self.bitwise_or(x)
    def __xor__(self, x) -> Tensor: return self.xor(x)
    def __lshift__(self, x) -> Tensor: return self.lshift(x)
    def __rshift__(self, x) -> Tensor: return self.rshift(x)

    def __radd__(self, x) -> Tensor: return self.add(x, True)
    def __rsub__(self, x) -> Tensor: return self.sub(x, True)
    def __rmul__(self, x) -> Tensor: return self.mul(x, True)
    def __rpow__(self, x) -> Tensor: return self.pow(x, True)
    def __rtruediv__(self, x) -> Tensor: return self.div(x, True)
    def __rfloordiv__(self, x) -> Tensor: return self.div(x, True, upcast=False)
    def __rmatmul__(self, x) -> Tensor: return self.matmul(x, True)
    def __rand__(self, x) -> Tensor: return self.bitwise_and(x, True)
    def __ror__(self, x) -> Tensor: return self.bitwise_or(x, True)
    def __rxor__(self, x) -> Tensor: return self.xor(x, True)

    def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
    def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
    def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
    def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
    def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
    def __ifloordiv__(self, x) -> Tensor: return self.assign(self.div(x, upcast=False))
    def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))
    def __iand__(self, x) -> Tensor: return self.assign(self.bitwise_and(x))
    def __ior__(self, x) -> Tensor: return self.assign(self.bitwise_or(x))
    def __ixor__(self, x) -> Tensor: return self.assign(self.xor(x))
    def __ilshift__(self, x) -> Tensor: return self.assign(self.lshift(x))
    def __irshift__(self, x) -> Tensor: return self.assign(self.rshift(x))

    def __lt__(self, x) -> Tensor: return self.less_than(x)
    def __gt__(self, x) -> Tensor: return self.greate_than(x)
    def __ge__(self, x) -> Tensor: return (self<x).logical_not()
    def __le__(self, x) -> Tensor: return (self>x).logical_not()
    def __ne__(self, x) -> Tensor: return self.not_eq(x)  
    def __eq__(self, x) -> Tensor: return (self!=x).logical_not() 

    def dot(self, w: Tensor, acc_dtype: None) -> Tensor:
        pass

    def matmul(self, x: Tensor, reverse=False, acc_dtype=None) -> Tensor:
        return x.dot(self, acc_dtype=acc_dtype) if reverse else self.dot(x, acc_dtype=acc_dtype)
    
    def max_pool2d(self, kernel_size=(2,2), stride=None, dilation=1, padding=0):
        pass
    
    def conv2d(self, weight: Tensor, bias: Tensor=None, groups=1, stride=1, dilation=1, padding=0, acc_dtype=None) -> Tensor:
        pass

    def conv_transpose2d(self, weight:Tensor, bias:Optional[Tensor]=None, groups=1, stride=1, dilation=1, padding=0, output_padding=0) -> Tensor:
        pass
    
    def batchnorm(self, weight: Optional[Tensor], bias: Optional[Tensor], mean: Tensor, invstd: Tensor, axis: Union[int, Tuple[int, ...]]=1) -> Tensor:
        pass

    def linear(self, weight: Tensor, bias: Optional[Tensor]=None) -> Tensor:
        x = self.mul(weight) if len(weight.shape) == 1 else self.dot(weight)
        return x.add(bias) if bias is not None else x
    
    @property
    def T(self) -> Tensor:
        return self.transpose()
    
    def transpose(self, dim0=1, dim1=0) -> Tensor:
        order = list(range(self.ndim))
        order[dim0], order[dim0] = order[dim1], order[dim0]
        return self.permute(order)
    
    def permute(self, order, *args) -> Tensor:
        pass

    def dropout(self, p=0.5) -> Tensor:
        pass

    def exp(self) -> Tensor:
        pass

    def exp2(self) -> Tensor:
        pass

    def relu(self) -> Tensor:
        pass

    def sigmoid(self) -> Tensor:
        pass

    def sqrt(self) -> Tensor:
        pass

    def rsqrt(self):
        return self.reciprocal().sqrt()

    def sin(self) -> Tensor:
        pass

    def cos(self) -> Tensor:
        return ((math.pi/2)-self).sin()

    def tan(self) -> Tensor:
        return self.sin() / self.cos()
    
    def trunc(self) -> Tensor:
        pass

    def ceil(self) -> Tensor:
        pass

    def floor(self) -> Tensor:
        pass
  
    def round(self) -> Tensor: 
        pass

    def mean(self, axis: Optional[Union[int, Sequence[int]]]=None, keepdim=False) -> Tensor:
        pass

    def sign(self):
        pass
    
    def abs(self):
        return self * self.sign()
    
    def reciprocal(self) -> Tensor:
        pass
