"""
References:

* [May 30 video on manning publications twitch](https://twitch.tv/manningpublications)
* [1d conv blog post](https://controlandlearning.wordpress.com/2020/07/26/pytorch-basics-1d-convolution)
* [python history](https://gitlab.com/tangibleai/nlpia2/-/blob/main/src/nlpia2/ch07/mobprog_cnn_on_parts_of_speech.hist.py)
* https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
* https://gitlab.com/tangibleai/nlpia2/-/blob/main/src/nlpia2/ch07/ch07.ipynb
"""

import torch
import torch.nn as nn
import numpy as np
x_1d = torch.tensor([1, 1, 1, 1, 1], dtype=torch.float)
print(x_1d.shape)
x_1d.size()
x_1d.shape
x_1d = x_1d.unsqueeze(0)
x_1d
x_1d.shape
x_1d.unsqueeze(2)
x_1d.unsqueeze(2).shape
x_1d = x_1d.unsqueeze()
x_1d = x_1d.unsqueeze(0)
x_1d
x_1d.shape
cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1, bias=False)
print(cnn1d_1.weight)
print(cnn1d_1.bias)
cnn1d_1.state_dict()
state = cnn1d_1.state_dict()
state['weight'] = torch.tensor(np.array([[[2]]]))
cnn = cnn1d_1
cnn.load_state_dict(state)
cnn
cnn.weight
cnn.forward(x_1d)
cnn1d_1 = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False)
print(cnn1d_1.weight)
print(cnn1d_1.bias)
state['weight'] = torch.tensor(np.array([[[2, -1]]]))
cnn.load_state_dict(state)
state = cnn1d_1.state_dict()
cnn = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=2, bias=False
                )
cnn.load_state_dict(state)
x = x_1d
cnn.forward(x)
cnn
cnn.weight
state
state['weight'] = torch.tensor(np.array([[[2, -1]]]))
cnn.load_state_dict(state)
cnn.forward(x)
x = torch.tensor([1, 1, 2, 2, 2], dtype=torch.float)
cnn.forward(x)
x = torch.tensor([1, 1, 2, 2, 2], dtype=torch.float).unsqueeze(0).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float).unsqueeze(0).unsqueeze(0)
cnn.forward(x)
2 + -2
x = torch.tensor([1, 1, 2, -1, 1], dtype=torch.float).unsqueeze(0).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[1, 2], [0, 1], [0, 1], [0, 1]], dtype=torch.float).unsqueeze(0).unsqueeze(0)
x
cnn = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=2, bias=False
                )
cnn.forward(x)
x.transpose(2)
x.transpose(2, 3)
x = x.transpose(2, 3)
cnn.forward(x)
x = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]])
print(x.shape)
x = np.array([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]]).unsqueeze(0)
print(x.shape)
x = torch.tensor([[1, 1, 1, 1, 1], [1, 2, 3, 4, 5]]).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[1., 1., 1., 1., 1.], [1., 2., 3., 4., 5.]]).unsqueeze(0)
cnn.forward(x)
cnn.weight
state = cnn.state_dict()
state['weight'] = torch.tensor(np.array([[[2, -1], [0, 0]]]))
cnn.load_state_dict(state)
cnn.forward(x)
state['weight'] = torch.tensor(np.array([[[2, -1], [1, 1]]]))
cnn.load_state_dict(state)
cnn.forward(x)
x = torch.tensor([[1., 1., 1, -.5, 1.], [1., 1., 1., 1., 1.]]).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 1., 0]]).unsqueeze(0)
state['weight'] = torch.tensor(np.array([[[1, 0], [0, 1]]]))
cnn.load_state_dict(state)
cnn.forward(x)
state['weight'] = torch.tensor(np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]))
state['bias']
state
cnn.forward(x)
cnn.load_state_dict(state)
cnn = nn.Conv1d(in_channels=3, out_channels=1, kernel_size=3, bias=False)
cnn
cnn.load_state_dict(state)
cnn.forward(x)
x = torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 1., 0], [0, 0, 0, 0, 1]]).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1., 0], [0, 0, 0, 0, 1]]).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 1]]).unsqueeze(0)
cnn.forward(x)
x = torch.tensor([[0, 0, 1, 0, 0], [0, 0, 0, 1., 0], [0, 0, 0, 0, 1]]).unsqueeze(0)
