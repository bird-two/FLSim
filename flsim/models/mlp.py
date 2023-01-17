'''Multi-layer perceptron
This code is mainly from [1]. We make some modifications according to [3] (i.e., remove `self.softmax = nn.Softmax(dim=1)`).
As a result, we use `CrossEntropy` instead of `NLLLoss` as the loss function.
Ref [3] is another implementation of MLP (without "dropout").

References:
[1] https://github.com/AshwinRJ/Federated-Learning-PyTorch/blob/master/src/models.py
[2] https://github.com/AshwinRJ/Federated-Learning-PyTorch/issues/30
[2] https://github.com/katsura-jp/fedavg.pytorch/blob/master/src/models/mlp.py
'''

from torch import nn

class MLP(nn.Module):
    def __init__(self, dim_in=28*28, dim_hidden=64, dim_out=10):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        #self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x