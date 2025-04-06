# Mob Programming

August 9, 2022

By printing out the activations and classifications for each character in and character-based RNN you can find several patterns that the character-based RNN is learning.
This one appears to keep track of the length of the name and assumes all short names that start with a capital letter are Chinese.
So there must be one or more of the 128 hidden activations that is keeping track of the position within the name.

```python
>>> %run ch08_rnn_char_nationality.py
>>> model.predict('Nakamoto')
>>> model.predict_category('Nakamoto')
'Chilean'
>>> model.predict_category('Dostoevsky')
'Dutch'
>>> model.predict_category("O'Neal")
'Dutch'
>>> model.categories
['Algerian',
 'Arabic',
 'Brazilian',
 'Chilean',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Ethiopian',
 'Finnish']
>>> model.predict_category("Smith")
'Ethiopian'
>>> model.predict_category("James")
'English'
>>> model.predict_category("Johnson")
'English'
>>> model.predict_category("Khalid")
'Arabic'
>>> CATEGORIES
['Algerian',
 'Arabic',
 'Brazilian',
 'Chilean',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Ethiopian',
 'Finnish',
 'French',
 'German',
 'Greek',
 'Honduran',
 'Indian',
 'Irish',
 'Italian',
 'Japanese',
 'Korean',
 'Malaysian',
 'Mexican',
 'Moroccan',
 'Nepalese',
 'Nicaraguan',
 'Nigerian',
 'Palestinian',
 'Papua New Guinean',
 'Peruvian',
 'Polish',
 'Portuguese',
 'Russian',
 'Scottish',
 'South African',
 'Spanish',
 'Ukrainian',
 'Venezuelan',
 'Vietnamese']
>>> model.state_dict()
OrderedDict([('i2h.weight',
              tensor([[-0.0159, -0.0292,  0.0555,  ...,  0.0676,  0.0859, -0.0281],
                      [ 0.0611,  0.0419, -0.0440,  ..., -0.0521,  0.0645,  0.0066],
                      [ 0.0416,  0.0150,  0.0066,  ..., -0.0435, -0.0311,  0.0581],
                      ...,
                      [-0.1286, -0.0697,  0.0274,  ...,  0.1119, -0.0069,  0.0131],
                      [ 0.0826, -0.0340,  0.0248,  ...,  0.0924,  0.0565, -0.0148],
                      [ 0.0273, -0.0582,  0.0723,  ...,  0.0499, -0.0120,  0.0083]])),
             ('i2h.bias',
              tensor([ 0.0381, -0.1273, -0.0323, -0.0542, -0.0313,  0.0675, -0.0139, -0.0173,
                      -0.1220,  0.0991, -0.0466, -0.0124, -0.0598,  0.0172,  0.0267, -0.0092,
                      -0.1348, -0.0463,  0.0430,  0.0028,  0.1434,  0.1191,  0.0186, -0.0261,
                       0.0194,  0.0147,  0.0641, -0.0505, -0.0369,  0.0462,  0.0064, -0.0817,
                       0.1338,  0.0161,  0.0399, -0.0628,  0.0875, -0.0318, -0.1597,  0.1073,
                      -0.0541,  0.0158,  0.0141, -0.0192,  0.0397, -0.1550, -0.1628,  0.0460,
                      -0.0371, -0.0806, -0.0856, -0.0055,  0.0714, -0.0408, -0.1104,  0.0695,
                       0.1219,  0.0831,  0.0290, -0.0150,  0.0751,  0.0427, -0.0896, -0.0140,
                       0.0155,  0.0636,  0.1210,  0.0679, -0.0597, -0.0409,  0.0984, -0.1256,
                      -0.0052, -0.0053, -0.0846,  0.0725, -0.0871, -0.0614,  0.0356,  0.0172,
                       0.1581,  0.0232, -0.0411,  0.0211,  0.0118, -0.0382, -0.0589, -0.0474,
                       0.0940,  0.0393,  0.1366, -0.0647,  0.0261,  0.0660,  0.0120, -0.0346,
                      -0.0209,  0.1113, -0.0876,  0.0257,  0.0314, -0.0110, -0.0685,  0.0445,
                      -0.0353,  0.0510, -0.1095,  0.0445,  0.0908,  0.0561, -0.0415,  0.0398,
                       0.0793, -0.0009, -0.0240, -0.0047,  0.0761, -0.1821,  0.0588, -0.0761,
                       0.0388,  0.0968, -0.0498,  0.0141, -0.0173,  0.0104,  0.1341,  0.0420])),
             ('i2o.weight',
              tensor([[-0.0254, -0.0141,  0.0193,  ...,  0.0199, -0.1368, -0.1137],
                      [-0.0206,  0.0245,  0.0151,  ..., -0.0609,  0.0534, -0.0130],
                      [-0.0233,  0.0198,  0.0273,  ..., -0.0843, -0.1326, -0.0880],
                      ...,
                      [-0.0632, -0.0645,  0.0564,  ..., -0.1301, -0.0281,  0.0034],
                      [-0.0084,  0.0419, -0.0479,  ...,  0.0786,  0.0023, -0.0546],
                      [ 0.0334, -0.0090, -0.0676,  ...,  0.0093,  0.0609, -0.0879]])),
             ('i2o.bias',
              tensor([-0.0258,  0.0819, -0.1086, -0.0778,  0.2819,  0.1494, -0.0703,  0.1429,
                       0.0500, -0.3158]))])
>>> model.char2i
{' ': 0,
 "'": 1,
 ',': 2,
 '-': 3,
 '.': 4,
 ';': 5,
 'A': 6,
 'B': 7,
 'C': 8,
 'D': 9,
 'E': 10,
 'F': 11,
 'G': 12,
 'H': 13,
 'I': 14,
 'J': 15,
 'K': 16,
 'L': 17,
 'M': 18,
 'N': 19,
 'O': 20,
 'P': 21,
 'Q': 22,
 'R': 23,
 'S': 24,
 'T': 25,
 'U': 26,
 'V': 27,
 'W': 28,
 'X': 29,
 'Y': 30,
 'Z': 31,
 'a': 32,
 'b': 33,
 'c': 34,
 'd': 35,
 'e': 36,
 'f': 37,
 'g': 38,
 'h': 39,
 'i': 40,
 'j': 41,
 'k': 42,
 'l': 43,
 'm': 44,
 'n': 45,
 'o': 46,
 'p': 47,
 'q': 48,
 'r': 49,
 's': 50,
 't': 51,
 'u': 52,
 'v': 53,
 'w': 54,
 'x': 55,
 'y': 56,
 'z': 57}
>>>     category_tensor = torch.tensor([model.categories.index('Arabic')], dtype=torch.long)
...     char_seq_tensor = encode_one_hot_seq('Rochdi', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...
>>>     category_tensor = torch.tensor([model.categories.index('Arabic')], dtype=torch.long)
...     char_seq_tens = encode_one_hot_seq('Rochdi', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...
>>> outpu
>>> output
tensor([[-0.7978, -2.6131, -2.7192, -2.6588, -5.8693, -2.5005, -2.6670, -2.5685,
         -2.5130, -3.5603]], grad_fn=<LogSoftmaxBackward0>)
>>> output.numpy().exp()
>>> output.detach().numpy().exp()
>>> np.exp(output.detach().numpy())
array([[0.4503028 , 0.0733084 , 0.06592839, 0.07002873, 0.00282486,
        0.08204309, 0.0694619 , 0.07665113, 0.08102089, 0.02842977]],
      dtype=float32)
>>> model.categories
['Algerian',
 'Arabic',
 'Brazilian',
 'Chilean',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Ethiopian',
 'Finnish']
>>>     category_tensor = torch.tensor([model.categories.index('Brazilian')], dtype=torch.long)
...     char_seq_tens = encode_one_hot_seq('James', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...
>>> category_tensor
tensor([2])
>>> char_seq_tens
tensor([[[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0.]],

        [[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
          0., 0., 0., 0., 0., 0., 0.]]])
>>> model.char2i['J']
15
>>> char_seq_tens[0][0]
tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0.])
>>> char_seq_tens[0][0].argmax()
tensor(15)
>>> output.detach().numpy().exp
>>> np.exp(output.detach().numpy())
array([[0.05592799, 0.09724023, 0.11984343, 0.11412288, 0.00150427,
        0.1222087 , 0.15766343, 0.16288577, 0.1380381 , 0.03056507]],
      dtype=float32)
>>> np.exp(output.detach().numpy()).argmax()
7
>>> model.categories[np.exp(output.detach().numpy()).argmax()]
'English'
>>>     char_seq_tens = encode_one_hot_seq('Khalid', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...
>>> model.categories[np.exp(output.detach().numpy()).argmax()]
'Arabic'
>>>     char_seq_tens = encode_one_hot_seq('Khalid', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     outputs = []
...     hiddens = []
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...         outputs.append(output)
...         hiddens.append(hidden)
...
>>> cats = []
... for v in outputs:
...     cats.append(model.categories[np.exp(output.detach().numpy()).argmax()])
...
>>> cats
['Arabic', 'Arabic', 'Arabic', 'Arabic', 'Arabic', 'Arabic']
>>>     char_seq_tens = encode_one_hot_seq('James', char2i=char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     outputs = []
...     hiddens = []
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...         outputs.append(output)
...         hiddens.append(hidden)
...
>>> cats = []
... for v in outputs:
...     cats.append(model.categories[np.exp(output.detach().numpy()).argmax()])
...
>>> cats
['English', 'English', 'English', 'English', 'English']
>>> cats = []
... for v in outputs:
...     cats.append(model.categories[np.exp(v.detach().numpy()).argmax()])
...
>>> hist
>>> def visualize_outputs(model, text):
...     char_seq_tens = encode_one_hot_seq(text, char2i=model.char2i)
...     hidden = torch.zeros(1, model.n_hidden)
...     model.zero_grad()
...     outputs = []
...     hiddens = []
...     for i in range(char_seq_tens.size()[0]):
...         output, hidden = model(char_tens=char_seq_tens[i], hidden=hidden)
...         outputs.append(output)
...         hiddens.append(hidden)
...     cats = []
...     for v in outputs:
...         cats.append(model.categories[np.exp(v.detach().numpy()).argmax()])
...     return cats
...
>>> visualize_outputs(model, 'Khalid')
['Chinese', 'Chinese', 'Chinese', 'Chinese', 'Algerian', 'Arabic']
>>> visualize_outputs(model, 'Kho')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'James')
['Chinese', 'Chinese', 'Chinese', 'Ethiopian', 'English']
>>> model.categories
['Algerian',
 'Arabic',
 'Brazilian',
 'Chilean',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Ethiopian',
 'Finnish']
>>> model.categories[2]
'Brazilian'
>>> df.nationality == model.categories[2]
0        False
1        False
2        False
3        False
4        False
         ...  
36236    False
36237    False
36238    False
36239    False
36240    False
Name: nationality, Length: 36241, dtype: bool
>>> mask = df.nationality == model.categories[2]
>>> df[mask]
        rank        surname     count frequency  freq_numerator  freq_denominator nationality
6054     1.0       da Silva  12585868      1:17             1.0              17.0   Brazilian
6055     2.0     dos Santos   7427753      1:29             1.0              29.0   Brazilian
6056     3.0        Pereira   5594058      1:38             1.0              38.0   Brazilian
6057     4.0          Alves   4872987      1:44             1.0              44.0   Brazilian
6058     5.0       Ferreira   4848931      1:44             1.0              44.0   Brazilian
...      ...            ...       ...       ...             ...               ...         ...
7050   997.0       Watanabe     10191   1:21006             1.0           21006.0   Brazilian
7051   998.0         Calado     10173   1:21043             1.0           21043.0   Brazilian
7052   999.0  Lima da Silva     10151   1:21089             1.0           21089.0   Brazilian
7053  1000.0       Carolino     10145   1:21101             1.0           21101.0   Brazilian
7054  1000.0       Viturino     10145   1:21101             1.0           21101.0   Brazilian

[1001 rows x 7 columns]
>>> df[mask]['surname']
6054         da Silva
6055       dos Santos
6056          Pereira
6057            Alves
6058         Ferreira
            ...      
7050         Watanabe
7051           Calado
7052    Lima da Silva
7053         Carolino
7054         Viturino
Name: surname, Length: 1001, dtype: object
>>> visualize_outputs(model, 'Silva')
['Chinese', 'Chinese', 'Chinese', 'English', 'Czech']
>>> visualize_outputs(model, 'da Silva')
['English',
 'Chilean',
 'Chinese',
 'Ethiopian',
 'Algerian',
 'Chilean',
 'Finnish',
 'Chilean']
>>> visualize_outputs(model, 'd')
['English']
>>> visualize_outputs(model, 'dos')
['English', 'Brazilian', 'Chinese']
>>> mask = df.nationality == "Ethiopian"
>>> df[mask]['surname']
0         Tesfaye
1        Mohammed
2        Getachew
3           Abebe
4           Girma
           ...   
326        Zegeye
327        Halima
35992      Rediet
35993       Abebe
35994      Seyoum
Name: surname, Length: 331, dtype: object
>>> visualize_outputs(model, 'Seyoum')
['Chinese', 'Chinese', 'Chinese', 'Ethiopian', 'Ethiopian', 'Arabic']
>>> model.categories
['Algerian',
 'Arabic',
 'Brazilian',
 'Chilean',
 'Chinese',
 'Czech',
 'Dutch',
 'English',
 'Ethiopian',
 'Finnish']
>>> visualize_outputs(model, 'Finnish')
['Chinese', 'Algerian', 'Chinese', 'Chinese', 'Algerian', 'Chilean', 'English']
>>> df[df.nationality == "Finnish"]['surname']
11016      Korhonen
11017      Virtanen
11018      Nieminen
11019       Makinen
11020    Hamalainen
            ...    
12012      Jurvanen
12013       Pakkala
12014      Sorjonen
12015         Kopra
12016        Kunnas
Name: surname, Length: 1001, dtype: object
>>> visualize_outputs(model, 'Virtanen')
['Chinese',
 'Algerian',
 'Chinese',
 'English',
 'Chilean',
 'Dutch',
 'Chilean',
 'Finnish']
>>> visualize_outputs(model, 'Nieminen')
['Chinese',
 'Chinese',
 'Chinese',
 'English',
 'Algerian',
 'Dutch',
 'Dutch',
 'Finnish']
>>> visualize_outputs(model, 'nieminen')
['Finnish',
 'Chinese',
 'Chinese',
 'English',
 'Algerian',
 'Dutch',
 'Dutch',
 'Finnish']
>>> visualize_outputs(model, 'Ibe')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'James')
['Chinese', 'Chinese', 'Chinese', 'Ethiopian', 'English']
>>> visualize_outputs(model, 'Kho')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'Khe')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'Chi')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'Che')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'Ist')
['Chinese', 'Chinese', 'Chinese']
>>> visualize_outputs(model, 'ABC')
['Chinese', 'Chinese', 'Chinese']
>>> hidden
tensor([[ 0.7650,  0.3478,  0.2797, -0.2389,  0.1785, -0.1259,  0.1729,  0.1893,
         -0.3880, -0.3227,  0.5396,  0.1074, -0.0694, -0.0647, -0.0202, -0.1899,
          0.1021, -0.6102,  0.6791,  0.2874, -0.4168,  0.2491, -0.0199,  0.1803,
          0.3995, -0.1678, -0.2863,  0.0587,  0.3349, -0.2488,  0.3305,  0.1535,
         -0.2493, -0.3401,  0.3626,  0.1651, -0.2134, -0.0799,  0.1989, -0.1644,
          0.2997, -0.5028,  0.2657, -0.0229,  0.0573, -0.1310,  0.1033, -0.4050,
          0.4664, -0.4144, -0.3775, -0.4920, -0.2207, -0.0866, -0.3021, -0.2861,
         -0.1821, -0.0122,  0.3899,  0.2603,  0.0315,  0.2969, -0.0812,  0.0949,
          0.0011,  0.4148, -0.4361, -0.1083, -0.0033, -0.1239,  0.1645, -0.1882,
          0.1380,  0.5273,  0.1907,  0.2701,  0.1493,  0.1028,  0.5549, -0.1272,
         -0.3273,  0.2066, -0.3364,  0.4339, -0.4331,  0.2160, -0.3260,  0.0295,
          0.0203,  0.1927, -0.0330, -0.1137,  0.1529, -0.1593,  0.0412,  0.0080,
         -0.1514, -0.1885,  0.0810,  0.0055,  0.0676, -0.1401, -0.0742,  0.1509,
         -0.2833, -0.1239,  0.3334, -0.1608,  0.0121,  0.1697, -0.0599,  0.0183,
          0.0157,  0.1904,  0.4377,  0.7200, -0.1752,  0.2512,  0.1098,  0.5695,
          0.0877, -0.4999,  0.4887, -0.0151,  0.2072, -0.3981, -0.0878, -0.3235]],
       grad_fn=<AddmmBackward0>)
>>> hist -o -p -f ch08_mobprog_surname_nationality_visualization.md
```
