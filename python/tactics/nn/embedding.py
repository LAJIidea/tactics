from tactics.tensor import Tensor

class Embedding:
    """
    A simple lookup table that stores embeddings of a fixed dictionary and size.
    """
    def __init__(self, vocab_size: int, embed_size: int):
        self.vocab_sz, self.embed_sz, self.weight = vocab_size, embed_size, Tensor.glorot_uniform(vocab_size, embed_size)

    def __call__(self, idx: Tensor) -> Tensor:
        if idx.numel() == 0:
            return Tensor.empty(idx.shape+(self.embed_sz,))
        arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz)
        if not hasattr(self, 'arange'):
            self.arange = Tensor.arange(self.vocab_sz)
        arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.reshape(weight_shp).expand(big_shp)
        return (arange == idx).mul(vals).sum(2)