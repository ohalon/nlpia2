>>> filepath2 = 'char_rnn_from_scratch_refactored-1_311-17min_28sec'
>>> from char_rnn_from_scratch_refactored import *
>>> from char_rnn_from_scratch_refactored import *
>>> meta3 = load_model_meta(filepath2, model=rnn3)
>>> rnn3 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
>>> meta3 = load_model_meta(filepath2)
>>> rnn3 = RNN(vocab_size=len(meta3['char2i']), n_hidden=128, n_categories=len(meta3['categories']))
>>> meta3 = load_model_meta(filepath2)
>>> meta3 = load_model_meta(filepath2, model=rnn3)
>>> meta3['model']
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> predict_category("Davletyarov", categories=meta3['categories'], char2i=meta3['char2i'], model=meta3['model'])
'Russian'
>>> ls -hal
>>> filepath
>>> filepath2
'char_rnn_from_scratch_refactored-1_311-17min_28sec'
>>> mkdir working
>>> ls $filepath2
>>> ls $filepath2*
>>> mv $filepath2* working/
>>> ls $filepath2*
>>> ls *.json
>>> rm *.meta.json
>>> ls *.pickle
>>> rm *.pickle
>>> hist -o -p -f working/test_load_model_1_311.py
>>> hist -o -p -f working/test_load_model_1_311.md
