# Extreme summarization

Mob programming session 2022-03-30.

```python
>>> from nessvec.indexers import Index
>>> index = Index(num_vecs=200_000)
>>> index.extreme_summarize('hello and goodbye')
array(['hello'], dtype=object)
>>> index['hello']
array([-1.920e-01,  1.544e-01,  4.670e-02,  5.920e-02,  1.369e-01,...
       -4.520e-02,  1.194e-01, -4.770e-02,  3.156e-01,  1.597e-01],
      dtype=float32)
>>> index[0]
array([ 1.0730e-01,  8.9000e-03,  6.0000e-04,  5.5000e-03, -6.4600e-02,...
        2.7600e-02,  1.8600e-02,  5.0000e-03,  1.1730e-01, -4.0000e-02],
      dtype=float32)
>>> index.query_series(index[0])
,      1.192093e-07
and    3.196178e-01
(      3.924445e-01
)      4.218287e-01
23     4.463376e-01
22     4.471740e-01
18     4.490819e-01
19     4.515444e-01
21     4.544248e-01
but    4.546938e-01
dtype: float64
>>> index['hello'] + index['goodby']
>>> index['hello'] + index['goodbye']
array([-1.08100004e-01,  1.88199997e-01,  2.78000012e-02,  1.48399994e-01,...
        2.17500001e-01,  1.63599998e-01,  4.83799994e-01,  2.88699985e-01],
      dtype=float32)
>>> (index['hello'] + index['goodbye']) / 2
array([-5.40500022e-02,  9.40999985e-02,  1.39000006e-02,  7.41999969e-02,...
        1.08750001e-01,  8.17999989e-02,  2.41899997e-01,  1.44349992e-01],
      dtype=float32)
>>> index['petr']
>>> index['peter']
array([-5.230e-02,  5.710e-02, -1.360e-01,  3.160e-02,  1.020e-01,...
        2.940e-02,  5.800e-02,  6.300e-03, -7.950e-02, -8.630e-02],
      dtype=float32)
>>> (index['hello'] + index['goodbye']) / 2
array([-5.40500022e-02,  9.40999985e-02,  1.39000006e-02,  7.41999969e-02,...
        1.08750001e-01,  8.17999989e-02,  2.41899997e-01,  1.44349992e-01],
      dtype=float32)
>>> ave = _
>>> av
>>> ave
array([-5.40500022e-02,  9.40999985e-02,  1.39000006e-02,  7.41999969e-02,...
        1.08750001e-01,  8.17999989e-02,  2.41899997e-01,  1.44349992e-01],
      dtype=float32)
>>> index.query(ave)
(array([[ 20023,  13169,  80641,  18520, 176095,  79916, 127313,  20760,
          87446,   9311]], dtype=int32),
 array([[0.0826809 , 0.08776871, 0.18111987, 0.26886882, 0.29631647,
         0.31067586, 0.31517358, 0.31676069, 0.32904991, 0.33001897]]))
>>> type(Index)
type
>>> type(index)
nessvec.indexers.Index
>>> index??
>>> index.query_series(ave)
goodbye      0.082681
hello        0.087769
good-bye     0.181120
Goodbye      0.268869
adios        0.296316
goodnight    0.310676
bye-bye      0.315174
farewell     0.316761
goodbyes     0.329050
hi           0.330019
dtype: float64
>>> import numpy as np
>>> np.linalg.norm(ave)
2.3451633
>>> index.query(ave / np.linalg.norm(ave))
(array([[ 20023,  13169,  80641,  18520, 176095,  79916, 127313,  20760,
          87446,   9311]], dtype=int32),
 array([[0.0826808 , 0.08776871, 0.18111987, 0.26886888, 0.29631641,
         0.3106758 , 0.31517352, 0.31676069, 0.32904991, 0.33001897]]))
>>> index.query_series(ave / np.linalg.norm(ave))
goodbye      0.082681
hello        0.087769
good-bye     0.181120
Goodbye      0.268869
adios        0.296316
goodnight    0.310676
bye-bye      0.315174
farewell     0.316761
goodbyes     0.329050
hi           0.330019
dtype: float64
>>> v1 = index['hello']; v2 = index['goodbye']
>>> index.query_series(v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2))
goodbye      0.085206
hello        0.085207
good-bye     0.183381
Goodbye      0.269409
adios        0.297123
goodnight    0.312120
bye-bye      0.315700
farewell     0.319290
hi           0.327724
goodbyes     0.331116
dtype: float64
>>> v1 = index['hello']; v2 = index['goodbye']; v3 = index['and']
>>> index.query_series(v1 / np.linalg.norm(v1) + v2 / np.linalg.norm(v2) + v3 / np.linalg.norm(v3))
hello        0.147890
goodbye      0.166779
good-bye     0.262311
Goodbye      0.274039
and          0.297894
adios        0.345147
hi           0.345835
goodnight    0.346922
bye-bye      0.350111
farewell     0.362055
dtype: float64
>>> index.query_series(v1 + v2 + v3 )
hello        0.106531
goodbye      0.113260
good-bye     0.212046
Goodbye      0.257193
adios        0.310980
goodnight    0.317864
bye-bye      0.322091
hi           0.327165
farewell     0.328947
goodbyes     0.348454
dtype: float64
>>> index.query_series((v1 + v2 + v3)/3)
hello        0.106531
goodbye      0.113260
good-bye     0.212046
Goodbye      0.257193
adios        0.310980
goodnight    0.317864
bye-bye      0.322091
hi           0.327165
farewell     0.328947
goodbyes     0.348454
dtype: float64
>>> index.get('schmutz')
>>> sent = 'Word embeddings are vectors we use to represent the meaning of words.'
>>> index.extreme_summarize(sent)
array(['of'], dtype=object)
>>> import sentence_transformers
>>> embed = sentence_transformers.SentenceTransformer()
>>> embed??
>>> embed = sentence_transformers.SentenceTransformer('all-roberta-large-v1')
>>> embed(sent)
>>> dir(embed)
['T_destination',...
 'add_module',
 'apply',
 'bfloat16',
 'buffers',
 'children',
 'cpu',
 'cuda',
 'device',
 'double',
 'dump_patches',
 'encode',
 'encode_multi_process',
 'eval',
 'evaluate',
 'extra_repr',
 'fit',
 'float',
 'forward',
 'get_buffer',
 'get_max_seq_length',
 'get_parameter',
 'get_sentence_embedding_dimension',
 'get_sentence_features',
 'get_submodule',
 'half',
 'load_state_dict',
 'max_seq_length',
 'modules',
 'named_buffers',
 'named_children',
 'named_modules',
 'named_parameters',
 'parameters',
 'register_backward_hook',
 'register_buffer',
 'register_forward_hook',
 'register_forward_pre_hook',
 'register_full_backward_hook',
 'register_parameter',
 'requires_grad_',
 'save',
 'save_to_hub',
 'share_memory',
 'smart_batching_collate',
 'start_multi_process_pool',
 'state_dict',
 'stop_multi_process_pool',
 'to',
 'to_empty',
 'tokenize',
 'tokenizer',
 'train',
 'training',
 'type',
 'xpu',
 'zero_grad']
>>> tokens = embed.tokenize(sent)
>>> tokens
{'input_ids': tensor([[  0, 771,   2],...
         [  0,  29,   2],
         [  0,   4,   2]]),
 'attention_mask': tensor([[1, 1, 1],...
         [1, 1, 1],
         [1, 1, 1]])}
>>> embed(tokens)
{'input_ids': tensor([[  0, 771,   2],...
         [  0,   4,   2]]),
 'attention_mask': tensor([[1, 1, 1],...
         [1, 1, 1]]),
 'token_embeddings': tensor([[[-0.1953,  0.4816,  0.3520,  ..., -0.3917,  0.2082,  0.1283],
          [-0.3725,  0.8092,  0.3861,  ..., -0.4367, -0.3818, -0.3703],
          ...
          [-1.2090,  0.2265,  0.3692,  ...,  1.3969, -1.4376, -1.6023]]],
        grad_fn=<NativeLayerNormBackward>),
 'sentence_embedding': tensor([[-0.0179,  0.0198,  0.0043,  ..., -0.0120, -0.0133, -0.0074],
         [-0.0466,  0.0344, -0.0091,  ...,  0.0233, -0.0227, -0.0126],
         [-0.0130,  0.0270,  0.0222,  ...,  0.0054, -0.0264, -0.0236],
         ...,
         [-0.0038,  0.0254,  0.0293,  ...,  0.0034, -0.0419, -0.0163],
         [ 0.0038,  0.0237,  0.0348,  ...,  0.0344, -0.0141, -0.0006],
         [-0.0537,  0.0112,  0.0290,  ...,  0.0221, -0.0436, -0.0449]],
        grad_fn=<DivBackward0>)}
>>> output = _
>>> output['sentence_embedding']
tensor([[-0.0179,  0.0198,  0.0043,  ..., -0.0120, -0.0133, -0.0074],
        [-0.0466,  0.0344, -0.0091,  ...,  0.0233, -0.0227, -0.0126],
        [-0.0130,  0.0270,  0.0222,  ...,  0.0054, -0.0264, -0.0236],
        ...,
        [-0.0038,  0.0254,  0.0293,  ...,  0.0034, -0.0419, -0.0163],
        [ 0.0038,  0.0237,  0.0348,  ...,  0.0344, -0.0141, -0.0006],
        [-0.0537,  0.0112,  0.0290,  ...,  0.0221, -0.0436, -0.0449]],
       grad_fn=<DivBackward0>)
>>> dir(sentence_transformers)
['CrossEncoder',
 'InputExample',
 'LoggingHandler',
 'ParallelSentencesDataset',
 'SentenceTransformer',
 'SentencesDataset',
 '__MODEL_HUB_ORGANIZATION__',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__path__',
 '__spec__',
 '__version__',
 'cross_encoder',
 'datasets',
 'evaluation',
 'model_card_templates',
 'models',
 'readers',
 'util']
>>> dir(sentence_transformers.util)
['Callable',
 'Dict',
 'HUGGINGFACE_HUB_CACHE',
 'HfApi',
 'List',
 'Optional',
 'Path',
 'REPO_ID_SEPARATOR',
 'Tensor',
 'Union',
 '__builtins__',
 '__cached__',
 '__doc__',
 '__file__',
 '__loader__',
 '__name__',
 '__package__',
 '__spec__',
 'batch_to_device',
 'cached_download',
 'community_detection',
 'cos_sim',
 'device',
 'dot_score',
 'fnmatch',
 'fullname',
 'hf_hub_url',
 'http_get',
 'import_from_string',
 'importlib',
 'information_retrieval',
 'logger',
 'logging',
 'normalize_embeddings',
 'np',
 'os',
 'pairwise_cos_sim',
 'pairwise_dot_score',
 'paraphrase_mining',
 'paraphrase_mining_embeddings',
 'pytorch_cos_sim',
 'queue',
 'requests',
 'semantic_search',
 'snapshot_download',
 'sys',
 'torch',
 'tqdm']
>>> dir(embed)
['T_destination',
 'zero_grad']
>>> embed.get_sentence_features(sent)
>>> embed.get_sentence_features(sent)
>>> embed.get_sentence_embedding_dimension
<bound method SentenceTransformer.get_sentence_embedding_dimension of SentenceTransformer(
  (0): Transformer({'max_seq_length': 256, 'do_lower_case': False}) with Transformer model: RobertaModel 
  (1): Pooling({'word_embedding_dimension': 1024, 'pooling_mode_cls_token': False, 'pooling_mode_mean_tokens': True, 'pooling_mode_max_tokens': False, 'pooling_mode_mean_sqrt_len_tokens': False})
  (2): Normalize()
)>
>>> embed.get_sentence_embedding_dimension()
1024
>>> dir(embed)
['T_destination',
 '_version',
 'add_module',
 'apply',
 'bfloat16',
 'buffers',
 'children',
 'cpu',
 'cuda',
 'device',
 'double',
 'dump_patches',
 'encode',
 'encode_multi_process',
 'eval',
 'evaluate',
 'extra_repr',
 'fit',
 'float',
 'forward',
 'get_buffer',
 'get_max_seq_length',
 'get_parameter',
 'get_sentence_embedding_dimension',
 'get_sentence_features',
 'get_submodule',
 'half',
 'load_state_dict',
 'max_seq_length',
 'modules',
 'named_buffers',
 'named_children',
 'named_modules',
 'named_parameters',
 'parameters',
 'register_backward_hook',
 'register_buffer',
 'register_forward_hook',
 'register_forward_pre_hook',
 'register_full_backward_hook',
 'register_parameter',
 'requires_grad_',
 'save',
 'save_to_hub',
 'share_memory',
 'smart_batching_collate',
 'start_multi_process_pool',
 'state_dict',
 'stop_multi_process_pool',
 'to',
 'to_empty',
 'tokenize',
 'tokenizer',
 'train',
 'training',
 'type',
 'xpu',
 'zero_grad']
>>> encode.encode(sent)
>>> embed.encode(sent)
array([-0.00646174,  0.01818253, -0.02262557, ...,  0.04628414,
        0.02597147,  0.0260355 ], dtype=float32)
>>> embed.encode('hello')
array([ 0.02065461,  0.02013085, -0.0104713 , ...,  0.0120449 ,
       -0.04717987, -0.0155734 ], dtype=float32)
>>> from nessvec.util import tokenize
>>> index.tokenize_key(sent)
'word embeddings are vectors we use to represent the meaning of words.'
>>> list(index.tokenize_key(sent))
['w',
 'o',
 ...
 's',
 '.']
>>> from nessvec.indexers import tokenize
>>> tokenize(sent)
['Word',
 'embeddings',
 'are',
 'vectors',
 'we',
 'use',
 'to',
 'represent',
 'the',
 'meaning',
 'of',
 'words',
 '.']
>>> tokens = tokenize(sent)
>>> rob = embed
>>> vecs = [rob.encode(t) for t in tokenize(sent)]
>>> vecs
[array([-0.01417434,  0.0048887 ,  0.01639565, ...,  0.02831655,
         0.04392671, -0.00379271], dtype=float32),
 ...
 array([-0.01876554, -0.01400215,  0.01148623, ...,  0.0395399 ,
        -0.04358005, -0.04492196], dtype=float32)]
>>> import pandas as pd
>>> np.array(vecs)
array([[-0.01417434,  0.0048887 ,  0.01639565, ...,  0.02831655,
         0.04392671, -0.00379271],
       [-0.01876554, -0.01400215,  0.01148623, ...,  0.0395399 ,
         0.02065022,  0.03793244],
       [-0.03497126,  0.02983398,  0.0065646 , ...,  0.09265038,
        -0.03346676,  0.01213683],
       ...,
       [ 0.00357197,  0.04912729,  0.03024391, ...,  0.00202209,
        -0.0048542 ,  0.01219975],
       [-0.00466492,  0.0430425 ,  0.02193295, ...,  0.05129678,
        -0.02026499,  0.01158769],
       [-0.05368135,  0.01117349,  0.02903732, ...,  0.02205243,
        -0.04358005, -0.04492196]], dtype=float32)
>>> pd.DataFrame(vecs)
        0         1         2         3         4     ...      1019      1020      1021      1022      1023
0  -0.014174  0.004889  0.016396 -0.000967 -0.017485  ...  0.018526 -0.030945  0.028317  0.043927 -0.003793
1  -0.018766 -0.014002  0.011486 -0.028952 -0.011244  ...  0.004181 -0.026628  0.039540  0.020650  0.037932
2  -0.034971  0.029834  0.006565  0.036389 -0.035220  ...  0.017858 -0.013139  0.092650 -0.033467  0.012137
3  -0.009524 -0.010052  0.012873  0.005408 -0.020306  ... -0.037779 -0.020111  0.005436 -0.007143 -0.013850
4  -0.043796  0.000415  0.020635 -0.000410 -0.026677  ... -0.022855 -0.020759 -0.008214 -0.002737  0.000511
5   0.000734 -0.025564  0.045768  0.001130  0.015857  ... -0.048281 -0.010704  0.014749  0.000611  0.019210
6  -0.014069  0.000335  0.056669 -0.021378 -0.002843  ... -0.018278 -0.013829  0.006448  0.022317 -0.009153
7  -0.024326  0.007676 -0.009451 -0.006520 -0.012713  ... -0.006048  0.014162  0.031692 -0.005574 -0.014500
8   0.016171  0.034788  0.024558 -0.016146  0.004880  ...  0.031065 -0.031826  0.016486 -0.025607  0.031421
9  -0.052561  0.043329 -0.013103 -0.043526 -0.026749  ... -0.030600  0.013197  0.022611  0.042450  0.031043
10  0.003572  0.049127  0.030244  0.029665  0.035807  ...  0.026937  0.011941  0.002022 -0.004854  0.012200
11 -0.004665  0.043043  0.021933  0.003436 -0.075763  ... -0.004952 -0.018448  0.051297 -0.020265  0.011588
12 -0.053681  0.011173  0.029037 -0.042446 -0.000206  ... -0.019440 -0.020624  0.022052 -0.043580 -0.044922

[13 rows x 1024 columns]
>>> np.array(vecs).shap
>>> np.array(vecs).shape
(13, 1024)
>>> vecs = pd.DataFrame(vecs)
>>> vecs.dot(vecs.T)
          0         1         2         3         4   ...        8         9         10        11        12
0   0.999999  0.175314  0.247606  0.173924  0.209978  ...  0.231346  0.220297  0.232254  0.581477  0.241268
1   0.175314  0.999999  0.202634  0.310357  0.121027  ...  0.195998  0.228788  0.149912  0.282668  0.199174
2   0.247606  0.202634  1.000000  0.299384  0.322818  ...  0.250295  0.264648  0.220011  0.321531  0.305026
3   0.173924  0.310357  0.299384  1.000000  0.219475  ...  0.296139  0.239081  0.191266  0.330167  0.270657
4   0.209978  0.121027  0.322818  0.219475  0.999999  ...  0.297506  0.166124  0.185014  0.205347  0.225938
5   0.228808  0.223860  0.301795  0.301845  0.194298  ...  0.233383  0.269488  0.215438  0.317724  0.329553
6   0.199941  0.187432  0.197813  0.303800  0.234637  ...  0.332896  0.200998  0.239909  0.243932  0.284505
7   0.308047  0.158880  0.279418  0.167739  0.281357  ...  0.235416  0.183661  0.208445  0.156998  0.232854
8   0.231346  0.195998  0.250295  0.296139  0.297506  ...  1.000000  0.253688  0.354610  0.271380  0.289022
9   0.220297  0.228788  0.264648  0.239081  0.166124  ...  0.253688  1.000000  0.281858  0.445346  0.266768
10  0.232254  0.149912  0.220011  0.191266  0.185014  ...  0.354610  0.281858  1.000000  0.318204  0.252901
11  0.581477  0.282668  0.321531  0.330167  0.205347  ...  0.271380  0.445346  0.318204  0.999999  0.200572
12  0.241268  0.199174  0.305026  0.270657  0.225938  ...  0.289022  0.266768  0.252901  0.200572  0.999999

[13 rows x 13 columns]
>>> vecs.dot(pd.DataFrame([rob.encode(sent)]).T)
           0
0   0.172085
1   0.526918
2   0.095572
3   0.362195
4   0.126602
5   0.119102
6   0.084988
7   0.021332
8   0.137256
9   0.336264
10  0.114800
11  0.407962
12  0.040519
>>> vecs.dot(pd.DataFrame([rob.encode(sent)]).T).sort_values()
>>> vecs.dot(pd.DataFrame([rob.encode(sent)]).T).sort_values('0')
>>> vecs.dot(pd.DataFrame([rob.encode(sent)]).T)
           0
0   0.172085
1   0.526918
2   0.095572
3   0.362195
4   0.126602
5   0.119102
6   0.084988
7   0.021332
8   0.137256
9   0.336264
10  0.114800
11  0.407962
12  0.040519
>>> similarities = _
>>> similarities['0']
>>> similarities[0]
0     0.172085
1     0.526918
2     0.095572
3     0.362195
4     0.126602
5     0.119102
6     0.084988
7     0.021332
8     0.137256
9     0.336264
10    0.114800
11    0.407962
12    0.040519
Name: 0, dtype: float32
>>> similarities[0].sort()
>>> similarities[0].sort_values()
7     0.021332
12    0.040519
6     0.084988
2     0.095572
10    0.114800
5     0.119102
4     0.126602
8     0.137256
0     0.172085
9     0.336264
3     0.362195
11    0.407962
1     0.526918
Name: 0, dtype: float32
>>> sent
'Word embeddings are vectors we use to represent the meaning of words.'
>>> index.extreme_summarize('hello but goodbye')
array(['hello'], dtype=object)
>>> %hist -o -p -f 2022-03-30-mob-programming-ch06-extreme-summarization.hist.py
```