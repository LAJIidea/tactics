from tactics.tensor import Tensor
from tactics.helpers import prod
from typing import Optional, Tuple
import math

def Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    return Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

class Conv2d:
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dialtion=1, groups=1, bias=True):
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        self.stride, self.padding, self.dilation, self.groups = stride, padding, dialtion, groups
        scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
        self.weight = Tensor.uniform(out_channels, in_channels//groups, *self.kernel_size, low=-scale, high=scale)
        self.bias = Tensor.uniform(out_channels, low=-scale, high=scale) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return x.conv2d(self.weight, self.bias, padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)
    
def ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, groups=1, bias=True):
    return ConvTranspose2d(in_channels, out_channels, (kernel_size,), stride, padding, output_padding, dilation, groups, bias)

class ConvTranspose2d(Conv2d):
    """
    Applies a 2D transposed convolution operator over an input image.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dialtion=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dialtion, groups, bias)
        scale = 1 / math.sqrt(in_channels * prod(self.kernel_size))
        self.weight = Tensor.uniform(in_channels, out_channels//groups, *self.kernel_size, low=-scale, high=scale)
        self.output_padding = output_padding

    def __call__(self, x:Tensor):
        return x.conv_transpose2d(self.weight, self.bias, padding=self.padding, output_padding=self.output_padding, stride=self.stride,
                              dilation=self.dilation, groups=self.groups)