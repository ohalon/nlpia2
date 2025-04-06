import pandas as pd
import jsonlines
with jsonlines.open('experiments.ljson') as fin:
    lines = list(fin)
df = pd.DataFrame(lines).round(4)
cols = 'rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
df[cols].round(2).sort_values('test_loss', ascending=False)
for parameter in model.parameters():
    print(parameter)
who
import torch
torch.load('model_epochs_32_rnn_type_RNN_RELU_hidden_size_200_batch_size_20_bptt_35_num_layers_3')
model = _
for parameter in model.parameters():
    print(parameter)
for parameter in model.parameters():
    print(parameter.size())
who
df
df[cols]
learned_parameters = []
for fn in df['filename']:
    learned_parameters.append(count_parameters(torch.load(fn)))

def count_parameters(model, include_constants=False):
    return sum(
        p.numel() for p in model.parameters()
        if include_constants or p.requires_grad
    )
learned_parameters = []
for fn in df['filename']:
    learned_parameters.append(count_parameters(torch.load(fn)))
df['learned_parameters'] = learned_parameters
df.to_csv('experiments.csv')
cols = 'learned_parameters rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
df[cols]
df[cols].round(2).sort_values('test_loss', ascending=False)
df['rnn_type'].unique()
df.to_csv('experiments.csv')
df['total_time'] = df['epoch_time'] * df['epochs']
df.to_csv('experiments.csv')
cols = 'learned_parameters rnn_type epochs lr num_layers dropout total_time val_loss test_loss'.split()
df[cols].round(2).sort_values('test_loss', ascending=False)
with jsonlines.open('experiments.ljson') as fin:
    lines = list(fin)
df = pd.DataFrame(lines)
df[cols].round(2).sort_values('test_loss', ascending=False)
learned_parameters = []
for fn in df['filename']:
    learned_parameters.append(count_parameters(torch.load(fn)))
df['learned_parameters'] = learned_parameters
df['total_time'] = df['epoch_time'] * df['epochs']
df[cols].round(2).sort_values('test_loss', ascending=False)
dfshort = _
dfshort
cols = list(df.columns)
cols[0] = 'parameters'
cols[4] = 'layers'
dfshort.columns = cols
dfshort
dfshort.columns
cols
cols = list(dfshort.columns)
cols
cols[0] = 'parameters'
cols[4] = 'layers'
dfshort.columns = cols
dfshort
cols[5] = 'drop'
cols[56] = 'time (s)'
cols[5] = 'time (s)'
dfshort.columns = cols
dfshort
cols = 'parameters  rnn_type  epochs   lr  layers  time (s)  total_time  val_loss  test_loss'.split()
cols = 'parameters  rnn_type  epochs   lr  layers  time (s)  total_time  test_loss'.split()
dfshort = dfshort[cols]
cols = 'parameters  rnn_type  epochs   lr  layers  time_(s)  total_time  test_loss'.split()
cols = dfshort.columns
cols
cols = list(dfshort.columns)
dfshort = dfshort[cols[:-2] + cols[-1:]]
dfshort
cols = list(dfshort.columns)
cols[-1] = 'loss'
cols[-2] = 'time (s)'
cols[-3] = 'drop'
dfshort.columns = cols
dfshort
hist -f hypertune_experiments_gru.hist.py
