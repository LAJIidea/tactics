from tactics.tensor import Tensor
from typing import Dict, Callable, Optional, Any, Mapping

def _forward_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"forward\" function")

class Module:
    """
    Base class for all neural network modules.
    Your models should also subclass this class.
    """

    training: bool
    _parameters: Dict[str, Optional[Tensor]]
    _buffers: Dict[str, Optional[Tensor]]
    _state_dict_hooks: Dict[int, Callable]
    _load_state_dict_pre_hooks: Dict[int, Callable]
    _state_dict_pre_hooks: Dict[int, Callable]
    _load_state_dict_post_hooks: Dict[int, Callable]
    _modules: Dict[str, Optional['Module']]
    call_super_init: bool = False
    _compiled_call_impl: Optional[Callable] = None

    def __init__(self, *args, **kwargs) -> None:
        pass

    forward: Callable[..., Any] = _forward_unimplemented

    def add_module(self, name: str, module: Optional['Module']) -> None:
        self._modules[name] = module

    def get_submodule(self, target: str) -> "Module":
        pass

    def get_parameters(self, target: str) -> Tensor:
        pass

    def get_buffer(self, target: str) -> Tensor:
        pass

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool=True, assign: bool=False):
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
    