import json
import pandas as pd
from pathlib import Path
from matplotlib import pyplot as plt

pd.options.display.max_columns = 100
pd.options.display.max_rows = 40
pd.options.display.float_format = '{: 5.4f}'.format


def get_sorted_experiments(
        datadir='experiments',
        important_hyperparams=None):
    important_hyperparams = important_hyperparams or str.split(
        'split_random_state torch_random_state filename case_sensitive num_stopwords'
        'kernel_lengths epochs learning_rate seq_len vocab_size dropout_portion train_accuracy test_accuracy'
    )
    datapath = Path(datadir)
    print(important_hyperparams)
    experiments = []
    for filepath in datapath.glob('*.json'):
        print(f'processing {filepath}')
        with filepath.open() as fin:
            d = json.load(fin)
            d['filename'] = filepath.with_suffix('').name.split('_')[-1]
            experiments.append(d)
    exp = experiments[-1]
    print(experiments[-1].keys())
    important_hyperparams = [k for k in exp.keys() if isinstance(exp[k], (float, int, list, str))]
    df = pd.DataFrame([{k: e.get(k, None) for k in important_hyperparams} for e in experiments])
    if 'test_accuracy' in important_hyperparams:
        df = df.sort_values('test_accuracy')
    elif 'test_loss' in important_hyperparams:
        df = df.sort_values('test_accuracy')
    else:
        test_cols = [s for s in df.columns if 'test' in s.lower() or 'val' in s.lower()]
        df = df.sort_values([s for s in test_cols if 'acc' in s.lower() or 'loss' in s.lower()][-1])
    return df


def get_sorted_experiments_from_log(datadir=(Path.home() / '.nlpia2-data' / 'log')):
    datadir = Path(datadir)
    paths = list(datadir.glob('*'))
    df = []
    for p in paths:
        d = json.load(p.open())
        df.append({k: d.get(k) for k in d.keys() if k not in ('learning_curve', 'y_test', 'y_train')})
        df[-1]['filename'] = p.name[-12:-5]
    df = pd.DataFrame(df)
    return df.sort_values('test_accuracy')


def plot_learning_curve(df, experiment):
    title = 'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'.format(
        **experiment)
    df.columns = 'loss training set test set'.split()
    df.columns = ['loss', 'training set', 'test set']
    fig = df[['training set', 'test set']].plot(
        linewidth=3, grid='on', ylabel='accuracy', xlabel='epochs', title=title)
    plt.show(block=False)
    return fig


def get_learning_curve_df(
        filepath=Path.home() / '.nlpia2-data' / 'log' / 'disaster_tweets_cnn_pipeline_14728.json'):
    with Path(filepath).open() as fin:
        results = json.load(fin)

    df = pd.DataFrame(results['learning_curve'],
                      columns=['loss', 'training_accuracy', 'test_accuracy'])
    return df


"""
# from .hist.py
ex = experiments
ex[0].keys()
important_hyperparams='split_random_state torch_random_state filename case_sensitive num_stopwords kernel_lengths epochs learning_rate seq_len vocab_size dropout_portion train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k: e.get(k, None) for k in important_hyperparams} for e in ex])
df_exp.sort_values('test_accuracy')
pd.options.display.float_format='{:05.4f}'.format
pd.options.display.max_columns=20
df_exp.sort_values('test_accuracy')
pd.options.display.float_format='{: 5.4f}'.format
df_exp.sort_values('test_accuracy')
df_exp.fillna('')
df_exp.sort_values('test_accuracy').to_csv('ch05_hyperparameter_tuning_sorted.csv')
bestexp = 'disaster_tweets_cnn_pipeline_24363.json'
experiments[14]
lc = pd.DataFrame(experiments[14]['hyperp'])
experiments[14].keys()
experiments[14]['learning_curve']
lc.columns = 'training_loss training_accuracy test_accuracy'.split()
lc = pd.DataFrame(experiments[4]['learning_curve'], columns='training_loss training_accuracy test_accuracy'.split())
lc[['training set', 'test set']].plot(linewidth=2, grid='on', ylabel='accuracy', xlabel='epochs', title=title)
title='seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'.format(**experiments[14])
lc.columns=['loss', 'training set', 'test set']
lc[['training set', 'test set']].plot(linewidth=3, grid='on', ylabel='Accuracy', xlabel='Epoch', title=title)
plt.show()
"""
