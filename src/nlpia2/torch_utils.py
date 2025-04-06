# torch_utils.py
import pandas as pd
from torch.utils.tensorboard import SummaryWriter

# default `log_dir` is "runs" - we'll be more specific here


def tensorboard_summary(model):
    writer = SummaryWriter(),
    writer.add_graph(model)


def count_parameters(model, include_constants=False):
    """ Estimate the number of trained parameters in each layer """
    return sum(count_layer_parameters(model=model, include_constants=False))


def count_layer_parameters(model, include_constants=False):
    """ Estimate the number of trained parameters in each layer """
    return [
        p.numel() for p in model.parameters()
        if include_constants or p.requires_grad
    ]


def describe_model(model):
    state = model.state_dict()
    names = state.keys()
    weights = state.values()
    params = model.parameters()
    df = pd.DataFrame([
        dict(
            name=name,
            learned_params=int(p.requires_grad) * p.numel(),
            total_params=p.numel(),
            size=p.size(),
        )
        for name, w, p in zip(names, weights, params)
    ]
    )
    df = df.set_index('name')
    return df
