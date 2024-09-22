from tactics.tensor import Tensor
from tactics.helpers import prod
from typing import Optional, Tuple

class BatchNorm:
    """
    Applies Batch Normalization over a 2D or 3D input.
    """
    def __init__(self, sz: int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
        self.eps, self.track_running_stats, self.momentum = eps, track_running_stats, momentum

        self.weight: Optional[Tensor] = Tensor.ones(sz) if affine else None
        self.bias: Optional[Tensor] = Tensor.zeros(sz) if affine else None

        self.num_batches_tracked = Tensor.zeros(1)
        if track_running_stats: 
            self.running_mean, self.running_var = Tensor.zeros(sz), Tensor.ones(sz)

    def calc_stats(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        shape_mask = [1, -1, *([-1]*(x.ndim-2))]
        if self.track_running_stats and not Tensor.training:
            return self.running_mean, self.running_var.reshape(shape=shape_mask).expand(x.shape)

        batch_mean = x.mean(axis=(reduce_axes:=tuple(x for x in range(x.ndim) if x != 1)))
        y = (x - batch_mean.detach().reshape(shape=shape_mask))
        batch_var = (y*y).mean(axis=reduce_axes)
        return batch_mean, batch_var

    def __call__(self, x: Tensor) -> Tensor:
        batch_mean, batch_var = self.calc_stats(x)
        if self.track_running_stats and Tensor.training:
            self.running_mean.assign((1-self.momentum) * self.running_mean + self.momentum * batch_mean.detach())
            self.running_var.assign((1-self.momentum) * self.running_var + self.momentum * prod(x.shape)/(prod(x.shape)-x.shape[1]) * batch_var.detach())
            self.num_batches_tracked += 1
        return x.batchnorm(self.weight, self.bias, batch_mean, batch_var.add(self.eps).rsqrt())
    
BatchNorm2d = BatchNorm3d = BatchNorm