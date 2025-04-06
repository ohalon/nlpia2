# from: https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9
import torch
from torch.autograd import Variable
# from torch.nn import functional as F
from torch import nn


x_data = Variable(torch.Tensor([[10.0], [9.0], [3.0], [2.0]]))
y_data = Variable(torch.Tensor([[90.0], [80.0], [50.0], [30.0]]))


class LinearRegression(torch.nn.Module):
    def __init__(self, num_inputs=1, num_neurons=1):
        super().__init__()
        self.linear = nn.Linear(
            in_features=num_inputs,
            out_features=num_neurons)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearRegression()

"""
Slopes and intercepts are initialized to random values between 0 and 1.

>>> torch.manual_seed(3200)
>>> model=LinearRegression()
>>> list(model.parameters())
>>> [Parameter containing:
 tensor([[0.2490]], requires_grad=True),
 Parameter containing:
 tensor([0.7142], requires_grad=True)]

>>> model = LinearRegression()
>>> list(model.parameters())
[Parameter containing:
 tensor([[0.9969]], requires_grad=True),
 Parameter containing:
 tensor([-0.9100], requires_grad=True)]

"""

# lr = learning rate
# SGD(params, lr=<required parameter>, momentum=0, dampening=0, weight_decay=0, nesterov=False)
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

# Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# MSELoss(size_average=None, reduce=None, reduction: str = 'mean')
criterion = torch.nn.MSELoss(size_average=False)


for epoch in range(20):
    model.train()
    optimizer.zero_grad()

    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)

    loss.backward()
    optimizer.step()
