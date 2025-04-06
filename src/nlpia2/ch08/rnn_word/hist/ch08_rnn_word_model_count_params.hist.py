%run generate
from main import *
!pip install -U tensorboard
!pip install -U tensorboard
from main import *
model
describe_model(model)
from torch_utils import describe_model
from nlpia2.torch_utils import describe_model
describe_model(model)
for p in model.parameters():
    print(vars(p)
    )
for p in model.parameters():
    print(dir(p))
for p in model.parameters():
    print([k for k in dir(p) if not k.startswith('_') and not k.endswith('_')])
for p in model.parameters():
    for k in dir(p) if not k.startswith('_') and not k.endswith('_'):
        print(k)
for p in model.parameters():
    for k in dir(p):
        if not k.startswith('_') and not k.endswith('_'):
            print(k)
for p in model.parameters():
    print(p)
    for k in dir(p):
        if not k.startswith('_') and not k.endswith('_'):
            print(k)
for p in model.parameters():
    print('-'*100)
    print(p.name)
    for k in dir(p):
        if k.startswith('_') or k.endswith('_'):
            continue
        if k in dir(torch.Tensor()):
            continue
        print(k)
for p in model.parameters():
    print('-'*100)
    print(p.names)
    for k in dir(p):
        if k.startswith('_') or k.endswith('_'):
            continue
        if k in dir(torch.Tensor()):
            continue
        print(k)
for p in model.parameters():
    print('-'*100)
    # print(p.names)
    for k in dir(p):
        if k.startswith('_') or k.endswith('_'):
            continue
        if k in dir(torch.Tensor()):
            continue
        print(k)
model.state_dict()
for k in model.state_dict():
    print k
for k in model.state_dict():
    print(k)
for k, v in model.state_dict().items():
    print(k, v)
for k, v in model.state_dict().items():
    print(k, v.size().product())
for k, v in model.state_dict().items():
    print(k, v.size().prod())
for k, v in model.state_dict().items():
    print(k, v.size().prod())
import math
torch.product
for k, v in model.state_dict().items():
    print(k, torch.prod(v.size()))
s = v.size()
s
s.numel()
for k, v in model.state_dict().items():
    print(k, torch.prod(v.numel()))
for k, v in model.state_dict().items():
    print(k, v.numel())
count_layer_parameters(model)
from nlpia2.torch_utils import count_layer_parameters
count_layer_parameters(model)
hist
%run ../../../torch_utils
%run ../../../torch_utils.py
%run ../../torch_utils.py
count_layer_parameters(model)
describe_model(model)
describe_model(model)
%run ../../torch_utils.py
describe_model(model)
%run ../../torch_utils.py
describe_model(model)
hist -o -p -f hist/ch08_rnn_word_model_count_params.hist.md
hist -f hist/ch08_rnn_word_model_count_params.hist.py
