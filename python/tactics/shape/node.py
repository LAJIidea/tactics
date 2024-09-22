from __future__ import annotations
from typing import List, Dict, Callable, Tuple, Type, Union, Optional, Any, Set, Mapping
import functools

class Node:
    b: Union[None, int]
    min: int
    max: sint
    def render(self, ops=None, ctx=None) -> Any:
        if ops is None: ops = render_python
        assert self.__class__ == Variable or self.min != self.max
        return ops[type(self)](self, ops, ctx)
    def vars(self) -> Set[Variable]: return set()
    # substitue Varaibles with values in var_vals
    def subsitute(self, var_vlas: Mapping[Variable, Variable]) -> Node: raise RuntimeError(self.__class__.__name__)
    def unbind(self) -> Tuple[Node, Optional[int]]: return self.subsitute({v: v.unbind()[0] for v in self.vars() if v.val is not None}), None
    
    @functools.cached_property
    def key(self) -> str: return self.render(ctx="DEBUG")
    def __repr__(self) -> str:
        return self.render(ctx="REPR")
    def __str__(self) -> str:
        return "<"+self.key+">"
    def __hash__(self) -> int:
        return hash(self.key)
    def __bool__(self): 
        return not (self.max == self.min == 0)
    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Node):
            return NotImplemented
        return self.key == value.key
    def __neg__(self):
        return self*-1
    def __add__(self, b: Union[Node, int]):
        return Node.sum([self, b])
    def _radd__(self, b: int):
        return self+b
    def __sub__(self, b:Union[Node, int]):
        return self+-b
    def __rsub__(self, b: int):
        return -self+b
    def __le__(self, b: Union[Node, int]):
        return self < (b+1)
    def __gt__(self, b: Union[Node, int]):
        return (-self) < (-b)
    def __ge__(self, b: Union[Node, int]):
        return (-self) < (-b+1)
    # def __lt__(self, b: Union[Node, int]):
    #     return create_node()

    @staticmethod
    def sum(nodes:List[Node]) -> Node:
        return None

class Variable(Node):
    def __new__(cls, *args):
        expr, nmin, nmax = args
        assert nmin >= 0 and nmin <= nmax, f"invalid Variable {expr=} {nmin=} {nmax=}"
        if nmin == nmax: return None
        return super().__new__(cls)

    def __getnewargs__(self): return (self.expr, self.min, self.max)  # args passed to __new__ when unpickling

    def __init__(self, expr:str, nmin:int, nmax:sint):
        self.expr, self.min, self.max = expr, nmin, nmax
        self._val: Optional[int] = None
    @property
    def val(self):
        assert self._val is not None, f"Variable isn't bound, can't access val of {self}"
        return self._val
    def bind(self, val):
        assert self._val is None and self.min<=val<=self.max, f"cannot bind {val} to {self}"
        self._val = val
        return self
    def unbind(self) -> Tuple[Variable, int]:
        assert self.val is not None, f"cannot unbind {self}"
        return Variable(self.expr, self.min, self.max), self.val
    def vars(self): return {self}
    def substitute(self, var_vals: Mapping[Variable, Variable]) -> Node: return var_vals.get(self, self)

# symbolic int, these are allowed in a Tensor shape
sint = Union[int, Variable]

render_python: Dict[Type, Callable[..., str]] = {
    Variable: lambda self, ops, ctx: f"{self.expr}[{self.min}-{self.max}{'='+str(self.val) if self._val is not None else ''}]" if ctx == "DEBUG" \
        else (f"Variable('{self.expr}', {self.min}, {self.max})"+(f".bind({self.val})" if self._val is not None else '') if ctx == "REPR" \
        else f"{self.expr}"),
}