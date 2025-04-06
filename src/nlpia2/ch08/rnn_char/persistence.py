# save this for chapter 13 - deploying models
import json
import torch
from pathlib import Path


def save_model(filepath, **meta):
    """ Save met to filepath.meta.json & state_dict to filepath.state_dict.pickle """
    filepath = Path(filepath)
    state_dict = meta.pop('state_dict', {})
    model = meta.pop('model', None)
    if model is not None:
        state_dict = model.state_dict()
    # losses = meta.pop('losses', [])  # noqa
    with filepath.with_suffix('.meta.json').open('wt') as fout:
        json.dump(meta, fout)
    with filepath.with_suffix('.state_dict.pickle').open('wb') as fout:
        torch.save(state_dict, fout)
    return filepath


def load_model_meta(filepath, model=None):
    """ Return meta dict from filepath.meta.json & state_dict from filepath.state_dict.pickle """
    filepath = Path(filepath)
    with filepath.with_suffix('.meta.json').open('rt') as fin:
        meta = json.load(fin)
    with filepath.with_suffix('.state_dict.pickle').open('rb') as fin:
        state_dict = torch.load(fin)
    meta['state_dict'] = state_dict
    if model is not None:
        model.load_state_dict(state_dict)
    meta['model'] = model
    return meta
