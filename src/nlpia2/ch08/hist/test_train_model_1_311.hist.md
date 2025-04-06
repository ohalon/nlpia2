>>> from char_rnn_from_scratch_refactored import *
>>> cd nlpia-rnn/
>>> from char_rnn_from_scratch_refactored import *
>>> len(CHAR2I)
58
>>> len(CATEGORIES)
18
>>> 186 / 58
3.206896551724138
>>> META
{'categories': ['Arabic',
  'Irish',
  'Spanish',
  'French',
  'German',
  'English',
  'Korean',
  'Vietnamese',
  'Scottish',
  'Japanese',
  'Polish',
  'Greek',
  'Czech',
  'Italian',
  'Portuguese',
  'Russian',
  'Dutch',
  'Chinese'],
 'char2i': {'g': 0,
  'J': 1,
  'j': 2,
  'l': 3,
  'X': 4,
  'e': 5,
  'L': 6,
  'H': 7,
  ' ': 8,
  "'": 9,
  'w': 10,
  'O': 11,
  'U': 12,
  'E': 13,
  'c': 14,
  'F': 15,
  'a': 16,
  'Q': 17,
  'y': 18,
  'u': 19,
  'I': 20,
  'W': 21,
  ',': 22,
  'p': 23,
  'b': 24,
  'z': 25,
  'G': 26,
  'T': 27,
  't': 28,
  'q': 29,
  'S': 30,
  'm': 31,
  'd': 32,
  'K': 33,
  'n': 34,
  'i': 35,
  'x': 36,
  'Y': 37,
  'M': 38,
  'R': 39,
  'r': 40,
  'N': 41,
  '-': 42,
  'f': 43,
  'Z': 44,
  's': 45,
  'D': 46,
  'P': 47,
  'o': 48,
  ';': 49,
  'v': 50,
  'k': 51,
  'V': 52,
  'h': 53,
  'C': 54,
  'A': 55,
  '.': 56,
  'B': 57},
 'n_hidden': 128,
 'n_categories': 18,
 'model': RNN(
   (i2h): Linear(in_features=186, out_features=128, bias=True)
   (i2o): Linear(in_features=186, out_features=18, bias=True)
   (softmax): LogSoftmax(dim=1)
 )}
>>> META['model']
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> META['rnn']
>>> META['model']
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> rnn
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> predict_category('Nakamoto')
'Greek'
>>> predict_category("Abe")
'Japanese'
>>> predict_category("Abe'")
'Greek'
>>> predict_category("Abe'")
'Greek'
>>> predict_category("Satoshi")
'Vietnamese'
>>> predict_category("Chen")
'Spanish'
>>> predict_category("O'Leary")
'English'
>>> predict_category("O'Callaghan")
'German'
>>> rnn.state_dict
<bound method Module.state_dict of RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)>
>>> rnn.state_dict()
OrderedDict([('i2h.weight',
              tensor([[-0.0384, -0.0705,  0.0361,  ...,  0.0002,  0.0140,  0.0105],
                      [ 0.0306, -0.0194, -0.0563,  ...,  0.0277, -0.0372,  0.0335],
                      [ 0.0413, -0.0502, -0.0354,  ...,  0.0089,  0.0272, -0.0238],
                      ...,
                      [ 0.0184,  0.0276,  0.0686,  ..., -0.0717, -0.0696, -0.0469],
                      [-0.0594, -0.0634,  0.0280,  ..., -0.0422, -0.0535, -0.0435],
                      [-0.0447,  0.0522, -0.0498,  ..., -0.0383,  0.0164, -0.0028]])),
             ('i2h.bias',
              tensor([-0.0711, -0.0148, -0.0559, -0.0120, -0.0167, -0.0009, -0.0403,  0.0306,
                      -0.0491, -0.0441,  0.0249,  0.0455, -0.0222,  0.0683,  0.0251,  0.0188,
                       0.0685,  0.0197,  0.0239,  0.0563, -0.0398,  0.0072, -0.0078,  0.0045,
                       0.0377,  0.0181, -0.0407, -0.0397,  0.0508,  0.0452, -0.0716,  0.0463,
                      -0.0346,  0.0460,  0.0551,  0.0399,  0.0701, -0.0424,  0.0149, -0.0682,
                      -0.0407,  0.0691,  0.0398, -0.0185,  0.0586, -0.0439,  0.0588, -0.0403,
                      -0.0503,  0.0307,  0.0411, -0.0672, -0.0427,  0.0529, -0.0342, -0.0579,
                       0.0338, -0.0711,  0.0233,  0.0418,  0.0728,  0.0649, -0.0505, -0.0365,
                      -0.0434, -0.0177,  0.0660, -0.0310,  0.0050, -0.0539,  0.0303,  0.0237,
                      -0.0556,  0.0318,  0.0413,  0.0191, -0.0397, -0.0604,  0.0522, -0.0391,
                       0.0494, -0.0608,  0.0471,  0.0592,  0.0476,  0.0274, -0.0548,  0.0164,
                       0.0121,  0.0166, -0.0596, -0.0454,  0.0057, -0.0499,  0.0214, -0.0497,
                      -0.0430,  0.0263,  0.0599, -0.0684,  0.0661, -0.0252, -0.0193, -0.0117,
                       0.0256,  0.0372, -0.0467,  0.0202,  0.0102,  0.0491,  0.0678, -0.0541,
                      -0.0279,  0.0250, -0.0077, -0.0266,  0.0327,  0.0050, -0.0525,  0.0063,
                       0.0663,  0.0356,  0.0482, -0.0687,  0.0669, -0.0328, -0.0520,  0.0235])),
             ('i2o.weight',
              tensor([[-0.0177, -0.0672, -0.0108,  ...,  0.0093, -0.0492,  0.0146],
                      [-0.0685,  0.0410,  0.0482,  ..., -0.0210, -0.0448, -0.0595],
                      [ 0.0180,  0.0183, -0.0312,  ..., -0.0261, -0.0661,  0.0270],
                      ...,
                      [-0.0055, -0.0411, -0.0246,  ..., -0.0477,  0.0583,  0.0575],
                      [ 0.0152,  0.0162,  0.0353,  ...,  0.0624, -0.0538,  0.0071],
                      [ 0.0484,  0.0726, -0.0408,  ..., -0.0435,  0.0466, -0.0490]])),
             ('i2o.bias',
              tensor([-0.0723, -0.0666,  0.0290,  0.0090,  0.0287,  0.0354, -0.0258,  0.0361,
                      -0.0067,  0.0442, -0.0554,  0.0381,  0.0637, -0.0539, -0.0430,  0.0260,
                       0.0282, -0.0243]))])
>>> meta['state_dict']
>>> META['state_dict']
>>> META.keys()
dict_keys(['categories', 'char2i', 'n_hidden', 'n_categories', 'model'])
>>> filename = 'char_rnn_from_scratch_refactored-1_517-09min_46sec'
>>> help(load_model)
>>> load_model(filename)
{'categories': ['Arabic',
  'Irish',
  'Spanish',
  'French',
  'German',
  'English',
  'Korean',
  'Vietnamese',
  'Scottish',
  'Japanese',
  'Polish',
  'Greek',
  'Czech',
  'Italian',
  'Portuguese',
  'Russian',
  'Dutch',
  'Chinese'],
 'char2i': {'g': 0,
  'J': 1,
  'j': 2,
  'l': 3,
  'X': 4,
  'e': 5,
  'L': 6,
  'H': 7,
  ' ': 8,
  "'": 9,
  'w': 10,
  'O': 11,
  'U': 12,
  'E': 13,
  'c': 14,
  'F': 15,
  'a': 16,
  'Q': 17,
  'y': 18,
  'u': 19,
  'I': 20,
  'W': 21,
  ',': 22,
  'p': 23,
  'b': 24,
  'z': 25,
  'G': 26,
  'T': 27,
  't': 28,
  'q': 29,
  'S': 30,
  'm': 31,
  'd': 32,
  'K': 33,
  'n': 34,
  'i': 35,
  'x': 36,
  'Y': 37,
  'M': 38,
  'R': 39,
  'r': 40,
  'N': 41,
  '-': 42,
  'f': 43,
  'Z': 44,
  's': 45,
  'D': 46,
  'P': 47,
  'o': 48,
  ';': 49,
  'v': 50,
  'k': 51,
  'V': 52,
  'h': 53,
  'C': 54,
  'A': 55,
  '.': 56,
  'B': 57},
 'train_time': '09:46',
 'min_loss': 1.5169210672662594,
 'state_dict': OrderedDict([('i2h.weight',
               tensor([[-0.1872,  0.0917,  0.1273,  ..., -0.0024,  0.0200,  0.0740],
                       [-0.0193,  0.1486, -0.0909,  ...,  0.0519, -0.0073, -0.0794],
                       [ 0.0858,  0.0537, -0.0179,  ..., -0.0536, -0.0108, -0.0528],
                       ...,
                       [-0.1062, -0.0835, -0.0270,  ..., -0.0982,  0.0405, -0.0557],
                       [ 0.0602,  0.0626,  0.0770,  ..., -0.0782, -0.0237,  0.0163],
                       [ 0.0971,  0.0016,  0.0607,  ..., -0.0359,  0.0024,  0.0347]])),
              ('i2h.bias',
               tensor([ 0.1321, -0.1389, -0.1859,  0.0301,  0.0612,  0.0863, -0.0064,  0.0915,
                        0.0743,  0.0274,  0.0936,  0.0055,  0.1933, -0.0174, -0.0141,  0.0546,
                        0.1205,  0.0773, -0.0993,  0.0329, -0.0196, -0.0108,  0.1413,  0.2396,
                       -0.0879, -0.1237, -0.1116,  0.0561, -0.0399,  0.1303, -0.0142, -0.0815,
                       -0.0752,  0.1747,  0.0010,  0.1609, -0.2132, -0.0485, -0.0716,  0.0962,
                       -0.1021, -0.1304,  0.1294, -0.0742,  0.1089, -0.0047, -0.0176, -0.1423,
                        0.0773,  0.0501,  0.0980, -0.0491,  0.0247, -0.0984,  0.0899,  0.0570,
                        0.0913,  0.1049, -0.0833, -0.0064, -0.0371, -0.0038, -0.2154, -0.1101,
                       -0.0664, -0.0444,  0.1084, -0.0515, -0.0577, -0.1352,  0.0815,  0.1978,
                        0.0052, -0.0407,  0.0040, -0.0462, -0.0765,  0.0474,  0.0030, -0.1754,
                       -0.0623, -0.1604, -0.0340, -0.1568,  0.0231,  0.0463, -0.1097,  0.0813,
                        0.1926,  0.0941, -0.0720, -0.0843, -0.0037, -0.0240, -0.0100, -0.0492,
                        0.0047,  0.1914,  0.0672, -0.0624, -0.1347, -0.1013, -0.2341, -0.1719,
                        0.1404,  0.0703,  0.1536,  0.0111, -0.0395,  0.0121, -0.0788, -0.0493,
                        0.0185, -0.0195,  0.0997, -0.0214, -0.1933,  0.0686, -0.0889, -0.0483,
                        0.1610, -0.0473,  0.0697,  0.1975, -0.0724,  0.0004,  0.0683, -0.0624])),
              ('i2o.weight',
               tensor([[-0.2475,  0.0465, -0.0153,  ..., -0.0179, -0.0480, -0.1672],
                       [-0.1684,  0.0577, -0.0507,  ..., -0.1271, -0.0884,  0.0643],
                       [-0.1621,  0.0726, -0.0780,  ...,  0.0551, -0.0528, -0.0076],
                       ...,
                       [-0.1716,  0.0691, -0.0109,  ..., -0.1680, -0.1683,  0.1056],
                       [-0.0861, -0.0131,  0.0596,  ...,  0.0343, -0.1316,  0.1356],
                       [ 1.0327, -0.0235, -0.0887,  ...,  0.0933,  0.1434, -0.2967]])),
              ('i2o.bias',
               tensor([-0.0534,  0.1016, -0.1624,  0.0807,  0.0394,  0.2112,  0.5198,  0.4940,
                        0.0317, -0.1152, -0.1817, -0.6404,  0.0377, -0.2976, -0.1822,  0.0640,
                       -0.1650,  0.1110]))])}
>>> rnn_old = rnn
>>> rnn = RNN()
>>> rnn = RNN(vocab_size=len(META['char2i']))
>>> rnn = RNN(vocab_size=len(META['char2i']), n_hidden=128, n_categories=len(CATEGORIES))
>>> rnn
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> META2 = load_model(filename)
>>> rnn = RNN(vocab_size=len(META2['char2i']), n_hidden=128, n_categories=len(META2['categories']))
>>> rnn
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> rnn.load_state_dict(META['state_dict'])
>>> rnn.load_state_dict(META2['state_dict'])
<All keys matched successfully>
>>> predict_category("O'Callaghan")
'German'
>>> predict_category("O'Leary")
'English'
>>> predict_category("Nakamoto")
'Greek'
>>> ls -hal
>>> filename
'char_rnn_from_scratch_refactored-1_517-09min_46sec'
>>> print_dataset_samples(df)
>>> df = load_dataset()
>>> print_dataset_samples(df)
>>> print_dataset_samples(df)
>>> predict_category("Davletyarov")
'German'
>>> print_predictions('Devorak')
   rank     text  log_loss category
0     0  Devorak -2.742045    Dutch
1     1  Devorak -2.775972  Spanish
2     2  Devorak -2.802665    Greek
>>>     print(f"META['categories']: {META['categories']}")
...     print(f'CATEGORIES: {CATEGORIES}')
...     print()
...     print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
...     print_predictions(input_line='Fyodor', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Dostoevsky', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
...     print_predictions(input_line='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Sanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Koyejo', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
...     print_predictions(input_line='Satoshi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Nakamoto', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
...     print_predictions(input_line='Rediet', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Abebe', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
...     print_predictions(input_line='Silvio', n_predictions=3, categories=CATEGORIES)
...     print_predictions(input_line='Micali', n_predictions=3, categories=CATEGORIES)
...
>>>     print(f"META['categories']: {META['categories']}")
...     print(f'CATEGORIES: {CATEGORIES}')
...     print()
...     print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
...     print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
...     print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
...     print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
...     print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
...     print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
...
   rank    text  log_loss  category
0     0  Micali -2.805666     Czech
1     1  Micali -2.813764  Scottish
2     2  Micali -2.819311   English
>>> print_example_tensor()
>>> 58 + 128
186
>>> help(train)
>>> ls -hal
>>> df
         name category
0      Khoury   Arabic
1       Nahas   Arabic
2       Daher   Arabic
3      Gerges   Arabic
4      Nazari   Arabic
...       ...      ...
20069    Zhai  Chinese
20070   Zhang  Chinese
20071     Zhi  Chinese
20072   Zhuan  Chinese
20073    Zhui  Chinese

[20074 rows x 2 columns]
>>> results = train(df)
>>> results
{'model': RNN(
   (i2h): Linear(in_features=186, out_features=128, bias=True)
   (i2o): Linear(in_features=186, out_features=18, bias=True)
   (softmax): LogSoftmax(dim=1)
 ),
 'n_hidden': 128,
 'losses': [2.8671472182273865,
  2.821946612358093,
  2.768683669805527,
  2.7211787350177765,
  2.6067244070768356,
  2.4925117932260035,
  2.3585279549360276,
  2.328374351620674,
  2.2578060274720193,
  2.1829546317458153,
  2.133399403221905,
  2.081378333926201,
  2.0949830952584745,
  2.047797423169017,
  1.9575361510664224,
  1.9046579734161495,
  1.9634913218319416,
  1.8884026370942593,
  1.8125248789479955,
  1.875375485625118,
  1.8152059813300148,
  1.8272708428502082,
  1.727591314867139,
  1.7891129239257426,
  1.7041916830940171,
  1.698797692248947,
  1.6969761557490566,
  1.7509024651125074,
  1.7166071840180084,
  1.6962616560962052,
  1.7260312985219062,
  1.6468711970821024,
  1.5964861724972725,
  1.6509366240557284,
  1.627774701528251,
  1.6151056950660423,
  1.6209595620185138,
  1.6122024231052492,
  1.6082114250604063,
  1.5654603112242185,
  1.6160839821263215,
  1.5893379600048065,
  1.4944561518342234,
  1.4522892135940493,
  1.4541176807638259,
  1.5510416472572834,
  1.5438817436133103,
  1.4844900171358604,
  1.4944846707405524,
  1.4412343637180747,
  1.5037207862595097,
  1.498835348906694,
  1.4255677196120378,
  1.512218948060181,
  1.411633187209023,
  1.3554458860089507,
  1.438542943635024,
  1.4003859191554948,
  1.4319411287871189,
  1.3899608713750131,
  1.3670121905936394,
  1.4038232047640486,
  1.3753902504259021,
  1.4618199928840623,
  1.3995068790671503,
  1.4296890313476325,
  1.311817888944177,
  1.4120396478781185,
  1.435596553599229,
  1.4244902503085322],
 'train_time': '17:28',
 'categories': ['Arabic',
  'Irish',
  'Spanish',
  'French',
  'German',
  'English',
  'Korean',
  'Vietnamese',
  'Scottish',
  'Japanese',
  'Polish',
  'Greek',
  'Czech',
  'Italian',
  'Portuguese',
  'Russian',
  'Dutch',
  'Chinese'],
 'char2i': {'g': 0,
  'J': 1,
  'j': 2,
  'l': 3,
  'X': 4,
  'e': 5,
  'L': 6,
  'H': 7,
  ' ': 8,
  "'": 9,
  'w': 10,
  'O': 11,
  'U': 12,
  'E': 13,
  'c': 14,
  'F': 15,
  'a': 16,
  'Q': 17,
  'y': 18,
  'u': 19,
  'I': 20,
  'W': 21,
  ',': 22,
  'p': 23,
  'b': 24,
  'z': 25,
  'G': 26,
  'T': 27,
  't': 28,
  'q': 29,
  'S': 30,
  'm': 31,
  'd': 32,
  'K': 33,
  'n': 34,
  'i': 35,
  'x': 36,
  'Y': 37,
  'M': 38,
  'R': 39,
  'r': 40,
  'N': 41,
  '-': 42,
  'f': 43,
  'Z': 44,
  's': 45,
  'D': 46,
  'P': 47,
  'o': 48,
  ';': 49,
  'v': 50,
  'k': 51,
  'V': 52,
  'h': 53,
  'C': 54,
  'A': 55,
  '.': 56,
  'B': 57}}
>>> filename
'char_rnn_from_scratch_refactored-1_517-09min_46sec'
>>> min(results['losses'])
1.311817888944177
>>> filename = 'char_rnn_from_scratch_refactored-1_311-17min_28sec'
>>> save_model(filename, **results)
PosixPath('char_rnn_from_scratch_refactored-1_311-17min_28sec')
>>> filepath2 = _
>>> predict_categ
>>> predict_category("Davletyarov")
'Russian'
>>>     print(f"META['categories']: {META['categories']}")
...     print(f'CATEGORIES: {CATEGORIES}')
...     print()
...     print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
...     print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
...     print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
...     print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
...     print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
...     print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
...
   rank    text  log_loss category
0     0  Micali -0.098465  Italian
1     1  Micali -3.839420   French
2     2  Micali -3.971620    Irish
>>> meta2 = load_model(filepath2)
>>> meta2
{'n_hidden': 128,
 'losses': [2.8671472182273865,
  2.821946612358093,
  2.768683669805527,
  2.7211787350177765,
  2.6067244070768356,
  2.4925117932260035,
  2.3585279549360276,
  2.328374351620674,
  2.2578060274720193,
  2.1829546317458153,
  2.133399403221905,
  2.081378333926201,
  2.0949830952584745,
  2.047797423169017,
  1.9575361510664224,
  1.9046579734161495,
  1.9634913218319416,
  1.8884026370942593,
  1.8125248789479955,
  1.875375485625118,
  1.8152059813300148,
  1.8272708428502082,
  1.727591314867139,
  1.7891129239257426,
  1.7041916830940171,
  1.698797692248947,
  1.6969761557490566,
  1.7509024651125074,
  1.7166071840180084,
  1.6962616560962052,
  1.7260312985219062,
  1.6468711970821024,
  1.5964861724972725,
  1.6509366240557284,
  1.627774701528251,
  1.6151056950660423,
  1.6209595620185138,
  1.6122024231052492,
  1.6082114250604063,
  1.5654603112242185,
  1.6160839821263215,
  1.5893379600048065,
  1.4944561518342234,
  1.4522892135940493,
  1.4541176807638259,
  1.5510416472572834,
  1.5438817436133103,
  1.4844900171358604,
  1.4944846707405524,
  1.4412343637180747,
  1.5037207862595097,
  1.498835348906694,
  1.4255677196120378,
  1.512218948060181,
  1.411633187209023,
  1.3554458860089507,
  1.438542943635024,
  1.4003859191554948,
  1.4319411287871189,
  1.3899608713750131,
  1.3670121905936394,
  1.4038232047640486,
  1.3753902504259021,
  1.4618199928840623,
  1.3995068790671503,
  1.4296890313476325,
  1.311817888944177,
  1.4120396478781185,
  1.435596553599229,
  1.4244902503085322],
 'train_time': '17:28',
 'categories': ['Arabic',
  'Irish',
  'Spanish',
  'French',
  'German',
  'English',
  'Korean',
  'Vietnamese',
  'Scottish',
  'Japanese',
  'Polish',
  'Greek',
  'Czech',
  'Italian',
  'Portuguese',
  'Russian',
  'Dutch',
  'Chinese'],
 'char2i': {'g': 0,
  'J': 1,
  'j': 2,
  'l': 3,
  'X': 4,
  'e': 5,
  'L': 6,
  'H': 7,
  ' ': 8,
  "'": 9,
  'w': 10,
  'O': 11,
  'U': 12,
  'E': 13,
  'c': 14,
  'F': 15,
  'a': 16,
  'Q': 17,
  'y': 18,
  'u': 19,
  'I': 20,
  'W': 21,
  ',': 22,
  'p': 23,
  'b': 24,
  'z': 25,
  'G': 26,
  'T': 27,
  't': 28,
  'q': 29,
  'S': 30,
  'm': 31,
  'd': 32,
  'K': 33,
  'n': 34,
  'i': 35,
  'x': 36,
  'Y': 37,
  'M': 38,
  'R': 39,
  'r': 40,
  'N': 41,
  '-': 42,
  'f': 43,
  'Z': 44,
  's': 45,
  'D': 46,
  'P': 47,
  'o': 48,
  ';': 49,
  'v': 50,
  'k': 51,
  'V': 52,
  'h': 53,
  'C': 54,
  'A': 55,
  '.': 56,
  'B': 57},
 'state_dict': OrderedDict([('i2h.weight',
               tensor([[-0.0857, -0.2011,  0.0355,  ...,  0.0254,  0.0691, -0.0199],
                       [-0.0454,  0.0275, -0.1081,  ...,  0.0356, -0.0539,  0.0066],
                       [ 0.0774, -0.0716, -0.1427,  ...,  0.0391, -0.0303,  0.0045],
                       ...,
                       [-0.1663,  0.0812,  0.1243,  ..., -0.0265, -0.1084, -0.0283],
                       [ 0.0013, -0.0473,  0.0264,  ..., -0.1098, -0.0496, -0.0729],
                       [-0.0564, -0.1088, -0.0805,  ..., -0.0360,  0.0612,  0.0022]])),
              ('i2h.bias',
               tensor([-0.1421,  0.1197,  0.0206, -0.0216, -0.0465,  0.0084,  0.0399, -0.1008,
                        0.0488,  0.0414,  0.0149, -0.0192,  0.0714,  0.0813,  0.1196,  0.0112,
                        0.0037,  0.0078,  0.0458,  0.0376, -0.0958,  0.0224, -0.0571,  0.0138,
                        0.0034,  0.0606,  0.0106, -0.0571,  0.1492, -0.0139,  0.0270,  0.0327,
                       -0.1947,  0.0612,  0.0612,  0.0210, -0.0606,  0.0097,  0.0434,  0.0023,
                       -0.0638,  0.0689, -0.1105, -0.1213, -0.0420, -0.1569,  0.0402, -0.1228,
                       -0.0392,  0.0563,  0.1197, -0.1060, -0.1890,  0.0144, -0.0224, -0.0591,
                        0.0101, -0.1126,  0.0497, -0.0485,  0.0767,  0.1616, -0.0911, -0.0362,
                       -0.0288, -0.0963,  0.0228, -0.0245, -0.0553, -0.0207,  0.1048, -0.0617,
                       -0.0693,  0.0278,  0.0483,  0.0691, -0.1587, -0.0689, -0.0175, -0.1212,
                        0.0694,  0.0540,  0.1198,  0.2359,  0.1043,  0.0321,  0.1130, -0.1440,
                       -0.1777,  0.0637, -0.1824,  0.0253, -0.0216, -0.0067,  0.0351, -0.1525,
                       -0.0446, -0.0250,  0.1283, -0.0049, -0.0675, -0.0628,  0.0426,  0.0314,
                        0.0076,  0.0061, -0.0149,  0.0145,  0.0504,  0.0748,  0.2153, -0.0430,
                        0.0005,  0.0421,  0.0010, -0.1259,  0.0233,  0.0853, -0.0032, -0.0050,
                       -0.1043,  0.0894,  0.0833,  0.0103,  0.0728, -0.0484, -0.0383,  0.0222])),
              ('i2o.weight',
               tensor([[-0.2815, -0.0672, -0.0370,  ..., -0.1170, -0.1320,  0.0742],
                       [-0.3031,  0.0410,  0.0293,  ..., -0.1706, -0.0377, -0.3837],
                       [-0.2143,  0.0183, -0.0483,  ...,  0.0527, -0.1062,  0.0707],
                       ...,
                       [-0.1438, -0.0411, -0.0318,  ..., -0.2376,  0.0815,  0.1580],
                       [-0.0332,  0.0162,  0.1559,  ...,  0.1144, -0.0276, -0.0189],
                       [ 1.2896,  0.0726, -0.0898,  ...,  0.0743,  0.0072, -0.1444]])),
              ('i2o.bias',
               tensor([-0.0449,  0.0489, -0.0954,  0.1373,  0.2341,  0.3368,  0.5902,  0.4655,
                        0.0224, -0.1709, -0.3004, -0.7787,  0.1166, -0.3638, -0.2438,  0.0113,
                       -0.0683,  0.0934]))])}
>>> results
{'model': RNN(
   (i2h): Linear(in_features=186, out_features=128, bias=True)
   (i2o): Linear(in_features=186, out_features=18, bias=True)
   (softmax): LogSoftmax(dim=1)
 ),
 'n_hidden': 128,
 'losses': [2.8671472182273865,
  2.821946612358093,
  2.768683669805527,
  2.7211787350177765,
  2.6067244070768356,
  2.4925117932260035,
  2.3585279549360276,
  2.328374351620674,
  2.2578060274720193,
  2.1829546317458153,
  2.133399403221905,
  2.081378333926201,
  2.0949830952584745,
  2.047797423169017,
  1.9575361510664224,
  1.9046579734161495,
  1.9634913218319416,
  1.8884026370942593,
  1.8125248789479955,
  1.875375485625118,
  1.8152059813300148,
  1.8272708428502082,
  1.727591314867139,
  1.7891129239257426,
  1.7041916830940171,
  1.698797692248947,
  1.6969761557490566,
  1.7509024651125074,
  1.7166071840180084,
  1.6962616560962052,
  1.7260312985219062,
  1.6468711970821024,
  1.5964861724972725,
  1.6509366240557284,
  1.627774701528251,
  1.6151056950660423,
  1.6209595620185138,
  1.6122024231052492,
  1.6082114250604063,
  1.5654603112242185,
  1.6160839821263215,
  1.5893379600048065,
  1.4944561518342234,
  1.4522892135940493,
  1.4541176807638259,
  1.5510416472572834,
  1.5438817436133103,
  1.4844900171358604,
  1.4944846707405524,
  1.4412343637180747,
  1.5037207862595097,
  1.498835348906694,
  1.4255677196120378,
  1.512218948060181,
  1.411633187209023,
  1.3554458860089507,
  1.438542943635024,
  1.4003859191554948,
  1.4319411287871189,
  1.3899608713750131,
  1.3670121905936394,
  1.4038232047640486,
  1.3753902504259021,
  1.4618199928840623,
  1.3995068790671503,
  1.4296890313476325,
  1.311817888944177,
  1.4120396478781185,
  1.435596553599229,
  1.4244902503085322],
 'train_time': '17:28',
 'categories': ['Arabic',
  'Irish',
  'Spanish',
  'French',
  'German',
  'English',
  'Korean',
  'Vietnamese',
  'Scottish',
  'Japanese',
  'Polish',
  'Greek',
  'Czech',
  'Italian',
  'Portuguese',
  'Russian',
  'Dutch',
  'Chinese'],
 'char2i': {'g': 0,
  'J': 1,
  'j': 2,
  'l': 3,
  'X': 4,
  'e': 5,
  'L': 6,
  'H': 7,
  ' ': 8,
  "'": 9,
  'w': 10,
  'O': 11,
  'U': 12,
  'E': 13,
  'c': 14,
  'F': 15,
  'a': 16,
  'Q': 17,
  'y': 18,
  'u': 19,
  'I': 20,
  'W': 21,
  ',': 22,
  'p': 23,
  'b': 24,
  'z': 25,
  'G': 26,
  'T': 27,
  't': 28,
  'q': 29,
  'S': 30,
  'm': 31,
  'd': 32,
  'K': 33,
  'n': 34,
  'i': 35,
  'x': 36,
  'Y': 37,
  'M': 38,
  'R': 39,
  'r': 40,
  'N': 41,
  '-': 42,
  'f': 43,
  'Z': 44,
  's': 45,
  'D': 46,
  'P': 47,
  'o': 48,
  ';': 49,
  'v': 50,
  'k': 51,
  'V': 52,
  'h': 53,
  'C': 54,
  'A': 55,
  '.': 56,
  'B': 57}}
>>> rnn.state_dict()
OrderedDict([('i2h.weight',
              tensor([[-0.1872,  0.0917,  0.1273,  ..., -0.0024,  0.0200,  0.0740],
                      [-0.0193,  0.1486, -0.0909,  ...,  0.0519, -0.0073, -0.0794],
                      [ 0.0858,  0.0537, -0.0179,  ..., -0.0536, -0.0108, -0.0528],
                      ...,
                      [-0.1062, -0.0835, -0.0270,  ..., -0.0982,  0.0405, -0.0557],
                      [ 0.0602,  0.0626,  0.0770,  ..., -0.0782, -0.0237,  0.0163],
                      [ 0.0971,  0.0016,  0.0607,  ..., -0.0359,  0.0024,  0.0347]])),
             ('i2h.bias',
              tensor([ 0.1321, -0.1389, -0.1859,  0.0301,  0.0612,  0.0863, -0.0064,  0.0915,
                       0.0743,  0.0274,  0.0936,  0.0055,  0.1933, -0.0174, -0.0141,  0.0546,
                       0.1205,  0.0773, -0.0993,  0.0329, -0.0196, -0.0108,  0.1413,  0.2396,
                      -0.0879, -0.1237, -0.1116,  0.0561, -0.0399,  0.1303, -0.0142, -0.0815,
                      -0.0752,  0.1747,  0.0010,  0.1609, -0.2132, -0.0485, -0.0716,  0.0962,
                      -0.1021, -0.1304,  0.1294, -0.0742,  0.1089, -0.0047, -0.0176, -0.1423,
                       0.0773,  0.0501,  0.0980, -0.0491,  0.0247, -0.0984,  0.0899,  0.0570,
                       0.0913,  0.1049, -0.0833, -0.0064, -0.0371, -0.0038, -0.2154, -0.1101,
                      -0.0664, -0.0444,  0.1084, -0.0515, -0.0577, -0.1352,  0.0815,  0.1978,
                       0.0052, -0.0407,  0.0040, -0.0462, -0.0765,  0.0474,  0.0030, -0.1754,
                      -0.0623, -0.1604, -0.0340, -0.1568,  0.0231,  0.0463, -0.1097,  0.0813,
                       0.1926,  0.0941, -0.0720, -0.0843, -0.0037, -0.0240, -0.0100, -0.0492,
                       0.0047,  0.1914,  0.0672, -0.0624, -0.1347, -0.1013, -0.2341, -0.1719,
                       0.1404,  0.0703,  0.1536,  0.0111, -0.0395,  0.0121, -0.0788, -0.0493,
                       0.0185, -0.0195,  0.0997, -0.0214, -0.1933,  0.0686, -0.0889, -0.0483,
                       0.1610, -0.0473,  0.0697,  0.1975, -0.0724,  0.0004,  0.0683, -0.0624])),
             ('i2o.weight',
              tensor([[-0.2475,  0.0465, -0.0153,  ..., -0.0179, -0.0480, -0.1672],
                      [-0.1684,  0.0577, -0.0507,  ..., -0.1271, -0.0884,  0.0643],
                      [-0.1621,  0.0726, -0.0780,  ...,  0.0551, -0.0528, -0.0076],
                      ...,
                      [-0.1716,  0.0691, -0.0109,  ..., -0.1680, -0.1683,  0.1056],
                      [-0.0861, -0.0131,  0.0596,  ...,  0.0343, -0.1316,  0.1356],
                      [ 1.0327, -0.0235, -0.0887,  ...,  0.0933,  0.1434, -0.2967]])),
             ('i2o.bias',
              tensor([-0.0534,  0.1016, -0.1624,  0.0807,  0.0394,  0.2112,  0.5198,  0.4940,
                       0.0317, -0.1152, -0.1817, -0.6404,  0.0377, -0.2976, -0.1822,  0.0640,
                      -0.1650,  0.1110]))])
>>> rnn.state_dict() == meta2['model'].state_dict()
>>> meta2.keys()
dict_keys(['n_hidden', 'losses', 'train_time', 'categories', 'char2i', 'state_dict'])
>>> rnn2 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
>>>     print(f"META['categories']: {META['categories']}")
...     print(f'CATEGORIES: {CATEGORIES}')
...     print()
...     print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
...     print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
...     print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
...     print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
...     print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
...     print()
...     print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
...     print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
...     print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
...
   rank    text  log_loss category
0     0  Micali -0.098465  Italian
1     1  Micali -3.839420   French
2     2  Micali -3.971620    Irish
>>> rnn_old = rnn
>>> rnn2 = rnn2.load_state_dict(meta2['state_dict'])
>>> rnn2
<All keys matched successfully>
>>> def predict_category(name, categories=CATEGORIES, char2i=CHAR2I, model=rnn):
...     tensor = encode_one_hot_seq(name, char2i=char2i)
...     pred_i = evaluate_tensor(tensor, model=model).topk(1)[1][0].item()
...     return categories[pred_i]
...
>>> predict_category("Davletyarov", model=rnn)
'Russian'
>>> predict_category("Davletyarov", model=rnn2)
>>> RNN
char_rnn_from_scratch_refactored.RNN
>>> RNN??
>>> RNN??
>>> dir(rnn2)
['__add__',
 '__class__',
 '__class_getitem__',
 '__contains__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getitem__',
 '__getnewargs__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__len__',
 '__lt__',
 '__module__',
 '__mul__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__rmul__',
 '__setattr__',
 '__sizeof__',
 '__slots__',
 '__str__',
 '__subclasshook__',
 '_asdict',
 '_field_defaults',
 '_fields',
 '_make',
 '_replace',
 'count',
 'index',
 'missing_keys',
 'unexpected_keys']
>>> dir(rnn)
['T_destination',
 '__annotations__',
 '__call__',
 '__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattr__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__setstate__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_apply',
 '_backward_hooks',
 '_buffers',
 '_call_impl',
 '_forward_hooks',
 '_forward_pre_hooks',
 '_get_backward_hooks',
 '_get_name',
 '_is_full_backward_hook',
 '_load_from_state_dict',
 '_load_state_dict_pre_hooks',
 '_maybe_warn_non_full_backward_hook',
 '_modules',
 '_named_members',
 '_non_persistent_buffers_set',
 '_parameters',
 '_register_load_state_dict_pre_hook',
 '_register_state_dict_hook',
 '_replicate_for_data_parallel',
 '_save_to_state_dict',
 '_slow_forward',
 '_state_dict_hooks',
 '_version',
 'add_module',
 'apply',
 'bfloat16',
 'buffers',
 'children',
 'cpu',
 'cuda',
 'double',
 'dump_patches',
 'eval',
 'extra_repr',
 'float',
 'forward',
 'get_buffer',
 'get_extra_state',
 'get_parameter',
 'get_submodule',
 'half',
 'i2h',
 'i2o',
 'init_hidden',
 'load_state_dict',
 'modules',
 'n_categories',
 'n_hidden',
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
 'register_module',
 'register_parameter',
 'requires_grad_',
 'set_extra_state',
 'share_memory',
 'softmax',
 'state_dict',
 'to',
 'to_empty',
 'train',
 'training',
 'type',
 'xpu',
 'zero_grad']
>>> rnn2.load_state_dict(meta2['state_dict'])
>>> rnn2 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
>>> rnn2.load_state_dict(meta2['state_dict'])
<All keys matched successfully>
>>> filepath2
PosixPath('char_rnn_from_scratch_refactored-1_311-17min_28sec')
>>> rnn2
RNN(
  (i2h): Linear(in_features=186, out_features=128, bias=True)
  (i2o): Linear(in_features=186, out_features=18, bias=True)
  (softmax): LogSoftmax(dim=1)
)
>>> predict_category("Davletyarov", model=rnn2)
'Russian'
>>> predict_category("Davletyarov", model=rnn_old)
'Russian'
>>> 
... def load_model_meta(filepath, model=None):
...     """ Return meta dict from filepath.meta.json & state_dict from filepath.state_dict.pickle """
...     filepath = Path(filepath)
...     with filepath.with_suffix('.meta.json').open('rt') as fin:
...         meta = json.load(fin)
...     with filepath.with_suffix('.state_dict.pickle').open('rb') as fin:
...         state_dict = torch.load(fin)
...     meta['state_dict'] = state_dict
...     if model is not None:
...         model.load_state_dict(state_dict)
...     meta['model'] = model
...     return meta
...
>>> load_model_meta(filepath2, model=rnn3)
>>> rnn3 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
>>> predict_category("Davletyarov", categories=meta2['categories'], char2i=meta2['char2i'], model=rnn_old)
'Russian'
>>> load_model_meta(filepath2, model=rnn3)
>>> import json
>>> meta3 = load_model_meta(filepath2, model=rnn3)
>>> predict_category("Davletyarov", categories=meta3['categories'], char2i=meta3['char2i'], model=meta3['model'])
'Russian'
>>> filepath2
PosixPath('char_rnn_from_scratch_refactored-1_311-17min_28sec')
>>> ls
>>> filepath2
PosixPath('char_rnn_from_scratch_refactored-1_311-17min_28sec')
>>> ls working
>>> hist -o -p -f working/test_train_model_1_311.md
