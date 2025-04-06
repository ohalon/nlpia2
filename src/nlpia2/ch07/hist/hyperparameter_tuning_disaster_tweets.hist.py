%run main_ch07.py
ls -hal
pwd
%run model_ch07.py
who
%run train_ch07.py
who
learning_df[['train_accuracies', 'validation_accuracies']].plot(linewidth=2)
hyperparams
hyperparms
hyyperparms['learning_curve']
hyperparms['learning_curve']
lc = pd.DataFrame(hyperparms['learning_curve'])
lc.plot()
plt.show()
from matplotlib import pyplot as plt
plt.show()
lc.columns = 'training_loss training_accuracy test_accuracy'.split()
lc[['train_accuracies', 'validation_accuracies']].plot(linewidth=2, grid='on')
lc[['train_accuracy', 'test_accuracy']].plot(linewidth=2, grid='on')
lc[['training_accuracy', 'test_accuracy']].plot(linewidth=2, grid='on')
plt.show()
lc[['training_accuracy', 'test_accuracy']].plot(linewidth=2, grid='on', xlabel='epochs')
 {'seq_len': 40,
 'vocab_size': 2000,
 'embedding_size': 50,
 'out_channels': 50,
 'num_stopwords': 0,
 'kernel_lengths': [1, 2, 3, 4, 5, 6],
 'strides': [1, 1, 1, 1, 1, 1],
 'batch_size': 24,
 'learning_rate': 0.002,
 'dropout': 0,
 'num_epochs': 400,
}
main_hyperparams = _
globals().update(main_hyperparams)
title = f'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'
lc[['training_accuracy', 'test_accuracy']].plot(linewidth=2, grid='on', xlabel='epochs', title=title)
plt.show()
more train_ch07.py
hyperp
hyperp['num_epochs'] = 50
hyperp['dropout'] = .2
hyperp['kernel_lengths'] = [2, 3, 4, 5]
hyperp['strides'] = [1, 1, 1, 1]
hyperp['learning_rate'] = .0015
ls *.json
main()
hyperp
globals().update(hyperp)
ls *.json
title = f'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'
lc2 = pd.DataFrame(hyperparms['learning_curve'])
lc2.columns = 'training_loss training_accuracy test_accuracy'.split()
lc2[['training_accuracy', 'test_accuracy']].plot(linewidth=2, grid='on', xlabel='epochs', title=title)
plt.show()
hyperparams['learning_curve']
hyperparms['learning_curve']
more train_ch07.py
more train_ch07.py
hyperp['epochs'] = 25
hyperp['learning_rate'] = .001
hyperp['strides'] = [1, 1, 1, 1, 1, 1]
hyperp['kernel_lengths'] = [1, 2, 3, 4, 5, 6]
pipeline_hyperp_with_dropout = main()
lc3 = pd.DataFrame(pipeline_hyperp_with_dropout['learning_curve'], columns=['training set', 'test set'])
lc3 = pd.DataFrame(pipeline_hyperp_with_dropout['learning curve'], columns=['training set', 'test set'])
lc3 = pd.DataFrame(pipeline_hyperp_with_dropout['hyperp']['learning_curve'], columns=['training set', 'test set'])
lc3 = pd.DataFrame(pipeline_hyperp_with_dropout['hyperp']['learning_curve'], columns=['loss', 'training set', 'test set'])
lc3.index.name = 'epoch'
# lc3[['training set', 'test set']].plot(linewidth=2, grid='on', ylable='accuracy', xlabel='epochs', title=title)
title = 'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'.format(**pipeline_hyperp_with_dropout['hyperp'])
title
lc3[['training set', 'test set']].plot(linewidth=2, grid='on', ylable='accuracy', xlabel='epochs', title=title)
lc3[['training set', 'test set']].plot(linewidth=2, grid='on', ylabel='accuracy', xlabel='epochs', title=title)
plt.show()
mv /home/hobs/overfit_learning_curve.png /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
mv /home/hobs/underfit_learning_curve.png /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
ls
from pathlib import Path
ls data
!find /home/hobs/code/tangibleai/nlpia-manuscript/ -name 'learning*.png'
!find /home/hobs/code/tangibleai/ -name 'learning*.png'
!find /home/hobs/ -name 'learning*.png'
ls data
ls data/hyperparam-tuning/
!mv disaster_tweets_cnn_pipeline_17*.json data/hyperparam-tuning/
experiments = []
for f in Path('data/hyperparam-tuning/').glob('*.json'):
    with f.open() as fin:
        experiments.append(json.load(fin))
len(experiments)
ex = experiments
ex.keys()
ex[0].keys()
pd.DataFrame([{k:e[k] for k in 'training_accuracy test_accuracy'.split()} for e in ex])
df_exp = pd.DataFrame([{k:e[k] for k in 'train_accuracy test_accuracy'.split()} for e in ex])
df_exp
df_exp.sort_values('test_accuracy')
ex[0].keys()
df_exp = pd.DataFrame([{k:e[k] for k in important_hyperparams.split()} for e in ex])
important_hyperparams = 'kernel_lengths vocab_size dropout train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k:e[k] for k in important_hyperparams.split()} for e in ex])
important_hyperparams = 'kernel_lengths vocab_size dropout train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k:e[k] for k in important_hyperparams} for e in ex])
important_hyperparams = 'kernel_lengths vocab_size dropout_portion train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k:e[k] for k in important_hyperparams} for e in ex])
df_exp = pd.DataFrame([{k:e.get(k, None) for k in important_hyperparams} for e in ex])
df_exp.sort_values('test_accuracy')
ex[0].keys()
important_hyperparams = 'kernel_lengths learning_rate seq_len vocab_size dropout_portion train_accuracy test_accuracy'.split()
df_exp.sort_values('test_accuracy')
df_exp = pd.DataFrame([{k:e.get(k, None) for k in important_hyperparams} for e in ex])
df_exp.sort_values('test_accuracy')
ex[0].keys()
important_hyperparams = 'kernel_lengths epochs learning_rate seq_len vocab_size dropout_portion train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k:e.get(k, None) for k in important_hyperparams} for e in ex])
df_exp.sort_values('test_accuracy')
ls *.json
ls data/hyperparam-tuning/
experiments = []
for f in Path('data/hyperparam-tuning/').glob('*.json'):
    with f.open() as fin:
        d = json.laod(fin)
        d['filename'] = f.name
        experiments.append(d)
experiments = []
for f in Path('data/hyperparam-tuning/').glob('*.json'):
    with f.open() as fin:
        d = json.load(fin)
        d['filename'] = f.name
        experiments.append(d)
ex = experiments
ex[0].keys()
important_hyperparams = 'split_random_state torch_random_state filename case_sensitive num_stopwords kernel_lengths epochs learning_rate seq_len vocab_size dropout_portion train_accuracy test_accuracy'.split()
df_exp = pd.DataFrame([{k:e.get(k, None) for k in important_hyperparams} for e in ex])
df_exp.sort_values('test_accuracy')
pd.options.display.float_format = '{:05.4f}'.format
pd.options.display.max_columns = 20
df_exp.sort_values('test_accuracy')
pd.options.display.float_format = '{: 5.4f}'.format
df_exp.sort_values('test_accuracy')
df_exp.fillna(-1)
df_exp.fillna('')
df_exp.sort_values('test_accuracy')
['\n'.join(c.split('_')) for c in ex.columns]
['\n'.join(c.split('_')) for c in df_ex.columns]
['\n'.join(c.split('_')) for c in df_exp.columns]
df_exp.sort_values('test_accuracy')
df_exp.columns = ['\n'.join(c.split('_')) for c in df_exp.columns]
df_exp.sort_values('test_accuracy')
df_exp.sort_values('test\naccuracy')
print(df_exp.sort_values('test\naccuracy'))
df_exp.columns = ['_'.join(c.split('\n')) for c in df_exp.columns]
print(df_exp.sort_values('test_accuracy'))
df_exp.sort_values('test_accuracy').to_csv('hyperparameter_tuning_sorted.csv')
mv hyperparameter_tuning_sorted.csv ch05_hyperparameter_tuning_sorted.csv
best = 'disaster_tweets_cnn_pipeline_24363.json'
experiments[14]
lc = pd.DataFrame(experiments[14])
lc = pd.DataFrame(experiments[14]['hyperp'])
experiments[14].keys()
experiments[14]['learning_curve']
lc = pd.DataFrame(experiments[4]['hyperp'])
lc = experiments[4]['learning_curve']
lc.columns = 'training_loss training_accuracy test_accuracy'.split()
lc = pd.DataFrame(experiments[4]['learning_curve'], columns='training_loss training_accuracy test_accuracy'.split())
lc
lc1 = pd.DataFrame(experiments[14]['learning_curve'], columns='training_loss training_accuracy test_accuracy'.split())
lc1
lc[['training set', 'test set']].plot(linewidth=2, grid='on', ylabel='accuracy', xlabel='epochs', title=title)
title = 'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'.format(**pipelinexperiments[14])
title = 'seq_len={seq_len} vocab_size={vocab_size} embedding_size={embedding_size} kernel_lengths={kernel_lengths}'.format(**experiments[14])
title
lc.columns = 'loss training set test set'.split()
lc.columns = ['loss', 'training set', 'test set']
lc[['training set', 'test set']].plot(linewidth=2, grid='on', ylabel='accuracy', xlabel='epochs', title=title)
plt.show()
!mv -i 'learning*.png' /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
!mv -i ~/learning*.png /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
!mv -i ~/learning*.svg /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
cd /home/hobs/code/tangibleai/nlpia-manuscript/manuscript/images/ch07/
ls *.png
rm learning-curve-85-80.png
mv underfit_learning_curve.png ../unused/ch07/learning_curve_underfit.png
mv overfit_learning_curve.png ../unused/ch07/learning_curve_overfit.png
hist -o -p -f hyperparameter_tuning_disaster_tweets.hist.md
hist -f hyperparameter_tuning_disaster_tweets.hist.py
