from itertools import product
import pandas as pd
from pathlib import Path
import json
import jsonlines
import argparse

from main import main, DEFAULT_HYPERPARAMS

DEFAULT_TUNING_PARAMS = dict(
    method='random',  # random
    tune_stop_fract=0.001,
    tune_stop_count=10,
    loss_name='test_loss',
)

DEFAULT_HYPERPARAM_RANGES = dict(
    early_stop_fract=(0.001,),
    early_stop_count=(2,),
    hidden_size=(200,),
    epochs=(32,),
    dropout=(0., .2, .35, .5),
    rnn_type=tuple('RNN_TANH RNN_RELU GRU LSTM'.split()),
    lr=(2.,),
    num_layers=(1, 2, 3, 4, 5)
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Optimize hyperparameters by attempting to run main.main(**hyperparams) with a variety of hyperparams')
    parser.add_argument('--method', type=str, default=DEFAULT_TUNING_PARAMS['method'],
                        help='Hyperparameter search algorithm: "grid" or "random".')
    parser.add_argument('--tune_stop_count', type=int, default=DEFAULT_TUNING_PARAMS['tune_stop_count'],
                        help='Stop tuning if improvement (reduction) in the loss (relative to best_loss) for this many attempts.')
    parser.add_argument('--tune_stop_fract', type=float, default=DEFAULT_HYPERPARAMS['tune_stop_fract'],
                        help='Stop tuning if improvement (reduction) in the loss (relative to best_loss) is less than this fraction of the best_loss for tune_stop_count attempts.')

    args = parser.parse_args()
    return args


def hyperparameter_search(
        # hyperparameter tuning parameters
        method='random',  # random
        tune_stop_fract=0.001,
        tune_stop_count=10,
        loss_name='test_loss',
        # hyperparameter ranges:
        train_stop_fract=(0.001,),
        train_stop_count=(2,),
        hidden_size=(200,),
        epochs=(32,),
        dropout=(0., .2, .35, .5),
        rnn_type=tuple('RNN_TANH RNN_RELU GRU LSTM'.split()),
        lr=(2.,),
        num_layers=(1, 2, 3, 4, 5),
        **kwargs):
    hypernames = 'hidden_size epochs rnn_type dropout lr num_layers'.split()
    hypervalues = [hidden_size, epochs, rnn_type, dropout, lr, num_layers]
    hypervalues += list(kwargs.values())
    hypernames += list(kwargs.keys())
    hyperparam_ranges = dict(list(zip(hypernames, hypervalues)))
    hyperparameter_grid = list(product(*list(hyperparam_ranges.values())))
    json.dump(hyperparameter_grid, open(f'experiment_grid_{len(hyperparameter_grid)}.json', 'w'), indent=4)
    json.dump(hyperparam_ranges, open(f'experiment_plan_{len(hyperparam_ranges)}.json', 'w'), indent=4)
    df = pd.DataFrame(hyperparameter_grid, columns=list(hyperparam_ranges.keys()))
    df = df.sample(len(df))  # shuffle row order while retaining original index
    df.to_csv('experiment_grid.csv')

    best_loss = 1e6
    no_improvement_count = 0

    print(f'Running {len(hyperparameter_grid)} experiments...')
    experiments = []

    for idx, hyperparams in df.iterrows():
        train_kwargs = DEFAULT_HYPERPARAMS.copy()
        train_kwargs['id'] = idx
        train_kwargs.update(hyperparams.to_dict())
        train_kwargs['filename'] = f'model_{idx:04d}.pt'
        print(json.dumps(train_kwargs, indent=4))

        results = main(**train_kwargs)

        experiments.append(results)
        with open('experiments.jsonl', 'at') as fout:
            print(json.dumps(results, indent=4))
            fout.write(json.dumps(results) + '\n')

        improvement = best_loss - results[loss_name]

        if improvement < tune_stop_fract * best_loss:
            no_improvement_count += 1
            print(f'HYPERPARAM improvement ({improvement:10.3f}) is < {tune_stop_fract:7.3f} * {best_loss:7.3f}...')
            print(f'HYPERPARAM no improvement count: {no_improvement_count}')
        else:
            no_improvement_count = 0
            best_loss = results[loss_name]
            print(f'HYPERPARAM new best_loss: {best_loss:7.3f} is {improvement * 100. / best_loss:7.3f}% improvement')

        if no_improvement_count >= tune_stop_count:
            print(f'HYPERPARAM Stopping tuning at best_loss: {best_loss:7.4f}.')
            break

    with open('experiments.json', 'at') as fout:
        json.dump(experiments, fout)
    return experiments


def show_best_experiments(jsonl_path='experiments.jsonl', topk=10):
    jsonl_path = Path(jsonl_path)
    with jsonlines.open(jsonl_path) as fin:
        lines = list(fin)
    df = pd.DataFrame(lines)
    df.to_csv(jsonl_path.with_suffix('.csv'))
    cols = 'id rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
    print(df[[c for c in cols if c in df.columns]].round(2).sort_values('test_loss').head(topk))
    return df


if __name__ == '__main__':
    args = parse_args()
    experiments = hyperparameter_search(
        method=args.method,  # random
        tune_stop_fract=args.tune_stop_fract,
        tune_stop_count=args.tune_stop_count,
        loss_name=args.loss_name,)
    print(experiments)
    df = pd.DataFrame(experiments)
    print(df)
    df.to_csv('experiments.csv')
    cols = 'rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
    print(df[cols].round(2).sort_values('test_loss').head())
