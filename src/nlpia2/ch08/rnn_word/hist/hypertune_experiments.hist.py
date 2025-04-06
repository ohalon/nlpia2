import pandas as pd
import jsonlines
with jsonlines.open('experiments.ljson') as fin:
    lines = list(fin)
df = pd.DataFrame(lines).round(4)
df = pd.DataFrame(lines)
df.to_csv('experiments.csv')
cols = 'dropout  epochs   lr  num_layers  epoch_time  val_loss test_loss'.split()
df[cols].round(2).sort_values('test_loss', ascending=False)
hist -f hypertune_experiments.hist.py
