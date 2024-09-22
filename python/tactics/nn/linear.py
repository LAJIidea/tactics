from tactics.tensor import Tensor
import math

class Linear:
    """
    Applies a linear transformation to the incoming data.
    """
    def __init__(self, in_features, out_features, bias=True) -> None:
        bound = 1 / math.sqrt(in_features)
        self.weight = Tensor.uniform(out_features, in_features, low=-bound, high=bound)
        self.bias = Tensor.uniform(out_features, low=-bound, high=bound) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.linear(self.weight.transpose(), self.bias)