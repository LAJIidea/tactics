from tactics import Tensor, nn

class Minist(nn.Model):

    def __init__(self) -> None:
        self.c1 = nn.Conv2d(1, 32, 5)
        self.c2 = nn.Conv2d(32, 32, 5)
        self.bn1 = nn.BatchNorm(32)
        self.c3 = nn.Conv2d(32, 64, 3)
        self.c4 = nn.Conv2d(64, 64, 3)
        self.bn2 = nn.BatchNorm(64)
        self.ln = nn.Linear(576, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c1(x)
        x = x.relu()
        x = self.c2(x).relu()
        x = self.bn1(x).max_pool2d()
        x = self.c3(x).relu()
        x = self.bn2(x).max_pool2d()
        return self.lin(x.flatten(1))
    

if __name__ == "__main__":
    model = Minist()
    state_dict = {
        'weight': Tensor.ones([5, 3, 3, 3]),
        'bias': Tensor.ones([5])
    }
    model.load_state_dict(model, state_dict)
    x = Tensor.empty()
    output = model(x)

