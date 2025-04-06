>>> import pandas as pd
>>> import jsonlines
>>> with jsonlines.open('experiments.ljson') as fin:
...     lines = list(fin)
...
>>> df = pd.DataFrame(lines).round(4)
>>> cols = 'rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
>>> df[cols].round(2).sort_values('test_loss', ascending=False)
     rnn_type  epochs   lr  num_layers  dropout  epoch_time  val_loss  test_loss
3    RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
147       GRU       1  0.5           5      0.0       58.94      6.96       6.89
155       GRU       1  0.5           5      0.2       72.42      6.95       6.89
146       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1    RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..        ...     ...  ...         ...      ...         ...       ...        ...
173       GRU      12  2.0           2      0.0       33.43      5.12       5.04
54   RNN_TANH      32  2.0           3      0.0       38.78      5.12       5.04
181       GRU      12  2.0           2      0.2       35.69      5.10       5.03
133  RNN_RELU      32  2.0           2      0.2       35.59      5.10       5.02
134  RNN_RELU      32  2.0           3      0.2       46.11      5.07       4.99

[196 rows x 8 columns]
>>> for parameter in model.parameters():
...     print(parameter)
...
>>> who
>>> import torch
>>> torch.load('model_epochs_32_rnn_type_RNN_RELU_hidden_size_200_batch_size_20_bptt_35_num_layers_3')
RNNModel(
  (drop): Dropout(p=0.5, inplace=False)
  (encoder): Embedding(33278, 200)
  (rnn): RNN(200, 200, num_layers=3, dropout=0.5)
  (decoder): Linear(in_features=200, out_features=33278, bias=True)
)
>>> model = _
>>> for parameter in model.parameters():
...     print(parameter)
...
>>> for parameter in model.parameters():
...     print(parameter.size())
...
>>> who
>>> df
     annealing_loss_improvement_pct  batch_size  bptt  ...  val_perplexity  test_loss test_perplexity
0                               1.0          20    35  ...        727.6933     6.5198        678.4218
1                               1.0          20    35  ...       1451.1062     6.8380        932.6119
2                               1.0          20    35  ...        838.4464     6.6617        781.8629
3                               1.0          20    35  ...       1054.8202     6.8972        989.5211
4                               1.0          20    35  ...        485.9174     6.1095        450.0999
..                              ...         ...   ...  ...             ...        ...             ...
191                             1.0          20    35  ...        227.6051     5.3490        210.3901
192                             1.0          20    35  ...        202.5507     5.2228        185.4480
193                             1.0          20    35  ...        199.9628     5.2163        184.2532
194                             1.0          20    35  ...        203.5218     5.2324        187.2414
195                             1.0          20    35  ...        239.4709     5.4012        221.6713

[196 rows x 29 columns]
>>> df[cols]
     rnn_type  epochs   lr  num_layers  dropout  epoch_time  val_loss  test_loss
0    RNN_TANH       1  0.5           1      0.0     22.6788    6.5899     6.5198
1    RNN_TANH       1  0.5           2      0.0     32.1066    7.2801     6.8380
2    RNN_TANH       1  0.5           3      0.0     38.0426    6.7316     6.6617
3    RNN_TANH       1  0.5           5      0.0     55.4573    6.9611     6.8972
4    RNN_TANH       1  2.0           1      0.0     22.6129    6.1860     6.1095
..        ...     ...  ...         ...      ...         ...       ...        ...
191       GRU      12  2.0           5      0.5     72.2765    5.4276     5.3490
192       GRU      32  0.5           1      0.0     23.2800    5.3110     5.2228
193       GRU      32  0.5           2      0.0     33.3500    5.2981     5.2163
194       GRU      32  0.5           3      0.0     39.8417    5.3158     5.2324
195       GRU      32  0.5           5      0.0     59.2928    5.4784     5.4012

[196 rows x 8 columns]
>>> learned_parameters = []
... for fn in df['filename']:
...     learned_parameters.append(count_parameters(torch.load(fn)))
...
>>> 
... def count_parameters(model, include_constants=False):
...     return sum(
...         p.numel() for p in model.parameters()
...         if include_constants or p.requires_grad
...     )
...
>>> learned_parameters = []
... for fn in df['filename']:
...     learned_parameters.append(count_parameters(torch.load(fn)))
...
>>> df['learned_parameters'] = learned_parameters
>>> df.to_csv('experiments.csv')
>>> cols = 'learned_parameters rnn_type epochs lr num_layers dropout epoch_time val_loss test_loss'.split()
>>> df[cols]
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  epoch_time  val_loss  test_loss
0              13424878  RNN_TANH       1  0.5           1      0.0     22.6788    6.5899     6.5198
1              13505278  RNN_TANH       1  0.5           2      0.0     32.1066    7.2801     6.8380
2              13585678  RNN_TANH       1  0.5           3      0.0     38.0426    6.7316     6.6617
3              13746478  RNN_TANH       1  0.5           5      0.0     55.4573    6.9611     6.8972
4              13424878  RNN_TANH       1  2.0           1      0.0     22.6129    6.1860     6.1095
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
191            14550478       GRU      12  2.0           5      0.5     72.2765    5.4276     5.3490
192            13585678       GRU      32  0.5           1      0.0     23.2800    5.3110     5.2228
193            13826878       GRU      32  0.5           2      0.0     33.3500    5.2981     5.2163
194            14068078       GRU      32  0.5           3      0.0     39.8417    5.3158     5.2324
195            14550478       GRU      32  0.5           5      0.0     59.2928    5.4784     5.4012

[196 rows x 9 columns]
>>> df[cols].round(2).sort_values('test_loss', ascending=False)
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  epoch_time  val_loss  test_loss
3              13746478  RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
147            14550478       GRU       1  0.5           5      0.0       58.94      6.96       6.89
155            14550478       GRU       1  0.5           5      0.2       72.42      6.95       6.89
146            14068078       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1              13505278  RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
173            13826878       GRU      12  2.0           2      0.0       33.43      5.12       5.04
54             13585678  RNN_TANH      32  2.0           3      0.0       38.78      5.12       5.04
181            13826878       GRU      12  2.0           2      0.2       35.69      5.10       5.03
133            13505278  RNN_RELU      32  2.0           2      0.2       35.59      5.10       5.02
134            13585678  RNN_RELU      32  2.0           3      0.2       46.11      5.07       4.99

[196 rows x 9 columns]
>>> df['rnn_type'].unique()
array(['RNN_TANH', 'RNN_RELU', 'GRU'], dtype=object)
>>> df.to_csv('experiments.csv')
>>> df['total_time'] = df['epoch_time'] * df['epochs']
>>> df.to_csv('experiments.csv')
>>> cols = 'learned_parameters rnn_type epochs lr num_layers dropout total_time val_loss test_loss'.split()
>>> df[cols].round(2).sort_values('test_loss', ascending=False)
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  total_time  val_loss  test_loss
3              13746478  RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
147            14550478       GRU       1  0.5           5      0.0       58.94      6.96       6.89
155            14550478       GRU       1  0.5           5      0.2       72.42      6.95       6.89
146            14068078       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1              13505278  RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
173            13826878       GRU      12  2.0           2      0.0      401.15      5.12       5.04
54             13585678  RNN_TANH      32  2.0           3      0.0     1240.98      5.12       5.04
181            13826878       GRU      12  2.0           2      0.2      428.28      5.10       5.03
133            13505278  RNN_RELU      32  2.0           2      0.2     1138.91      5.10       5.02
134            13585678  RNN_RELU      32  2.0           3      0.2     1475.43      5.07       4.99

[196 rows x 9 columns]
>>> with jsonlines.open('experiments.ljson') as fin:
...     lines = list(fin)
...
>>> df = pd.DataFrame(lines)
>>> df[cols].round(2).sort_values('test_loss', ascending=False)
>>> learned_parameters = []
... for fn in df['filename']:
...     learned_parameters.append(count_parameters(torch.load(fn)))
...
>>> df['learned_parameters'] = learned_parameters
>>> df['total_time'] = df['epoch_time'] * df['epochs']
>>> df[cols].round(2).sort_values('test_loss', ascending=False)
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  total_time  val_loss  test_loss
3              13746478  RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
155            14550478       GRU       1  0.5           5      0.2       72.42      6.95       6.89
147            14550478       GRU       1  0.5           5      0.0       58.94      6.96       6.89
146            14068078       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1              13505278  RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
133            13505278  RNN_RELU      32  2.0           2      0.2     1138.91      5.10       5.02
134            13585678  RNN_RELU      32  2.0           3      0.2     1475.43      5.07       4.99
198            14068078       GRU      32  2.0           3      0.0     1223.56      5.02       4.94
196            13585678       GRU      32  2.0           1      0.0      754.08      4.98       4.91
197            13826878       GRU      32  2.0           2      0.0      875.17      4.97       4.90

[199 rows x 9 columns]
>>> dfshort = _
>>> dfshort
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  total_time  val_loss  test_loss
3              13746478  RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
155            14550478       GRU       1  0.5           5      0.2       72.42      6.95       6.89
147            14550478       GRU       1  0.5           5      0.0       58.94      6.96       6.89
146            14068078       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1              13505278  RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
133            13505278  RNN_RELU      32  2.0           2      0.2     1138.91      5.10       5.02
134            13585678  RNN_RELU      32  2.0           3      0.2     1475.43      5.07       4.99
198            14068078       GRU      32  2.0           3      0.0     1223.56      5.02       4.94
196            13585678       GRU      32  2.0           1      0.0      754.08      4.98       4.91
197            13826878       GRU      32  2.0           2      0.0      875.17      4.97       4.90

[199 rows x 9 columns]
>>> cols = list(df.columns)
>>> cols[0] = 'parameters'
>>> cols[4] = 'layers'
>>> dfshort.columns = cols
>>> dfshort
     learned_parameters  rnn_type  epochs   lr  num_layers  dropout  total_time  val_loss  test_loss
3              13746478  RNN_TANH       1  0.5           5      0.0       55.46      6.96       6.90
155            14550478       GRU       1  0.5           5      0.2       72.42      6.95       6.89
147            14550478       GRU       1  0.5           5      0.0       58.94      6.96       6.89
146            14068078       GRU       1  0.5           3      0.0       39.83      6.94       6.88
1              13505278  RNN_TANH       1  0.5           2      0.0       32.11      7.28       6.84
..                  ...       ...     ...  ...         ...      ...         ...       ...        ...
133            13505278  RNN_RELU      32  2.0           2      0.2     1138.91      5.10       5.02
134            13585678  RNN_RELU      32  2.0           3      0.2     1475.43      5.07       4.99
198            14068078       GRU      32  2.0           3      0.0     1223.56      5.02       4.94
196            13585678       GRU      32  2.0           1      0.0      754.08      4.98       4.91
197            13826878       GRU      32  2.0           2      0.0      875.17      4.97       4.90

[199 rows x 9 columns]
>>> dfshort.columns
Index(['learned_parameters', 'rnn_type', 'epochs', 'lr', 'num_layers',
       'dropout', 'total_time', 'val_loss', 'test_loss'],
      dtype='object')
>>> cols
['parameters',
 'batch_size',
 'bptt',
 'clip',
 'layers',
 'datapath',
 'device',
 'dropout',
 'dry_run',
 'emsize',
 'epochs',
 'log_interval',
 'lr',
 'rnn_type',
 'nhead',
 'hidden_size',
 'num_layers',
 'onnx_export',
 'save',
 'filename',
 'seed',
 'tied',
 'best_val_loss',
 'epoch_num',
 'epoch_time',
 'val_loss',
 'val_perplexity',
 'test_loss',
 'test_perplexity',
 'learned_parameters',
 'total_time']
>>> cols = list(dfshort.columns)
>>> cols
['learned_parameters',
 'rnn_type',
 'epochs',
 'lr',
 'num_layers',
 'dropout',
 'total_time',
 'val_loss',
 'test_loss']
>>> cols[0] = 'parameters'
>>> cols[4] = 'layers'
>>> dfshort.columns = cols
>>> dfshort
     parameters  rnn_type  epochs   lr  layers  dropout  total_time  val_loss  test_loss
3      13746478  RNN_TANH       1  0.5       5      0.0       55.46      6.96       6.90
155    14550478       GRU       1  0.5       5      0.2       72.42      6.95       6.89
147    14550478       GRU       1  0.5       5      0.0       58.94      6.96       6.89
146    14068078       GRU       1  0.5       3      0.0       39.83      6.94       6.88
1      13505278  RNN_TANH       1  0.5       2      0.0       32.11      7.28       6.84
..          ...       ...     ...  ...     ...      ...         ...       ...        ...
133    13505278  RNN_RELU      32  2.0       2      0.2     1138.91      5.10       5.02
134    13585678  RNN_RELU      32  2.0       3      0.2     1475.43      5.07       4.99
198    14068078       GRU      32  2.0       3      0.0     1223.56      5.02       4.94
196    13585678       GRU      32  2.0       1      0.0      754.08      4.98       4.91
197    13826878       GRU      32  2.0       2      0.0      875.17      4.97       4.90

[199 rows x 9 columns]
>>> cols[5] = 'drop'
>>> cols[56] = 'time (s)'
>>> cols[5] = 'time (s)'
>>> dfshort.columns = cols
>>> dfshort
     parameters  rnn_type  epochs   lr  layers  time (s)  total_time  val_loss  test_loss
3      13746478  RNN_TANH       1  0.5       5       0.0       55.46      6.96       6.90
155    14550478       GRU       1  0.5       5       0.2       72.42      6.95       6.89
147    14550478       GRU       1  0.5       5       0.0       58.94      6.96       6.89
146    14068078       GRU       1  0.5       3       0.0       39.83      6.94       6.88
1      13505278  RNN_TANH       1  0.5       2       0.0       32.11      7.28       6.84
..          ...       ...     ...  ...     ...       ...         ...       ...        ...
133    13505278  RNN_RELU      32  2.0       2       0.2     1138.91      5.10       5.02
134    13585678  RNN_RELU      32  2.0       3       0.2     1475.43      5.07       4.99
198    14068078       GRU      32  2.0       3       0.0     1223.56      5.02       4.94
196    13585678       GRU      32  2.0       1       0.0      754.08      4.98       4.91
197    13826878       GRU      32  2.0       2       0.0      875.17      4.97       4.90

[199 rows x 9 columns]
>>> cols = 'parameters  rnn_type  epochs   lr  layers  time (s)  total_time  val_loss  test_loss'.split()
>>> cols = 'parameters  rnn_type  epochs   lr  layers  time (s)  total_time  test_loss'.split()
>>> dfshort = dfshort[cols]
>>> cols = 'parameters  rnn_type  epochs   lr  layers  time_(s)  total_time  test_loss'.split()
>>> cols = dfshort.columns
>>> cols
Index(['parameters', 'rnn_type', 'epochs', 'lr', 'layers', 'time (s)',
       'total_time', 'val_loss', 'test_loss'],
      dtype='object')
>>> cols = list(dfshort.columns)
>>> dfshort = dfshort[cols[:-2] + cols[-1:]]
>>> dfshort
     parameters  rnn_type  epochs   lr  layers  time (s)  total_time  test_loss
3      13746478  RNN_TANH       1  0.5       5       0.0       55.46       6.90
155    14550478       GRU       1  0.5       5       0.2       72.42       6.89
147    14550478       GRU       1  0.5       5       0.0       58.94       6.89
146    14068078       GRU       1  0.5       3       0.0       39.83       6.88
1      13505278  RNN_TANH       1  0.5       2       0.0       32.11       6.84
..          ...       ...     ...  ...     ...       ...         ...        ...
133    13505278  RNN_RELU      32  2.0       2       0.2     1138.91       5.02
134    13585678  RNN_RELU      32  2.0       3       0.2     1475.43       4.99
198    14068078       GRU      32  2.0       3       0.0     1223.56       4.94
196    13585678       GRU      32  2.0       1       0.0      754.08       4.91
197    13826878       GRU      32  2.0       2       0.0      875.17       4.90

[199 rows x 8 columns]
>>> cols = list(dfshort.columns)
>>> cols[-1] = 'loss'
>>> cols[-2] = 'time (s)'
>>> cols[-3] = 'drop'
>>> dfshort.columns = cols
>>> dfshort
     parameters  rnn_type  epochs   lr  layers  drop  time (s)  loss
3      13746478  RNN_TANH       1  0.5       5   0.0     55.46  6.90
155    14550478       GRU       1  0.5       5   0.2     72.42  6.89
147    14550478       GRU       1  0.5       5   0.0     58.94  6.89
146    14068078       GRU       1  0.5       3   0.0     39.83  6.88
1      13505278  RNN_TANH       1  0.5       2   0.0     32.11  6.84
..          ...       ...     ...  ...     ...   ...       ...   ...
133    13505278  RNN_RELU      32  2.0       2   0.2   1138.91  5.02
134    13585678  RNN_RELU      32  2.0       3   0.2   1475.43  4.99
198    14068078       GRU      32  2.0       3   0.0   1223.56  4.94
196    13585678       GRU      32  2.0       1   0.0    754.08  4.91
197    13826878       GRU      32  2.0       2   0.0    875.17  4.90

[199 rows x 8 columns]
>>> hist -f hypertune_experiments_gru.hist.py
>>> hist -o -p -f hist/hypertune_experiments_gru.hist.md
>>> hist -o -p -f hypertune_experiments_gru.hist.md
