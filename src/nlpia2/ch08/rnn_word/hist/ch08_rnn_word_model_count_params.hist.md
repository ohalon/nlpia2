>>> %run generate
>>> from main import *
>>> !pip install -U tensorboard
>>> !pip install -U tensorboard
>>> from main import *
>>> model
RNNModel(
  (embedding_dropout): Dropout(p=0, inplace=False)
  (encoder): Embedding(33278, 200)
  (rnn): GRU(200, 200)
  (decoder): Linear(in_features=200, out_features=33278, bias=True)
)
>>> describe_model(model)
>>> from torch_utils import describe_model
>>> from nlpia2.torch_utils import describe_model
>>> describe_model(model)
     learned_params  total_params          size
NaN         6655600       6655600  (33278, 200)
NaN          120000        120000    (600, 200)
NaN          120000        120000    (600, 200)
NaN             600           600        (600,)
NaN             600           600        (600,)
NaN         6655600       6655600  (33278, 200)
NaN           33278         33278      (33278,)
>>> for p in model.parameters():
...     print(vars(p)
...     )
...
>>> for p in model.parameters():
...     print(dir(p))
...
>>> for p in model.parameters():
...     print([k for k in dir(p) if not k.startswith('_') and not k.endswith('_')])
...
>>> for p in model.parameters():
...     for k in dir(p) if not k.startswith('_') and not k.endswith('_'):
...         print(k)
...
>>> for p in model.parameters():
...     for k in dir(p):
...         if not k.startswith('_') and not k.endswith('_'):
...             print(k)
...
>>> for p in model.parameters():
...     print(p)
...     for k in dir(p):
...         if not k.startswith('_') and not k.endswith('_'):
...             print(k)
...
>>> for p in model.parameters():
...     print('-'*100)
...     print(p.name)
...     for k in dir(p):
...         if k.startswith('_') or k.endswith('_'):
...             continue
...         if k in dir(torch.Tensor()):
...             continue
...         print(k)
...
>>> for p in model.parameters():
...     print('-'*100)
...     print(p.names)
...     for k in dir(p):
...         if k.startswith('_') or k.endswith('_'):
...             continue
...         if k in dir(torch.Tensor()):
...             continue
...         print(k)
...
>>> for p in model.parameters():
...     print('-'*100)
...     # print(p.names)
...     for k in dir(p):
...         if k.startswith('_') or k.endswith('_'):
...             continue
...         if k in dir(torch.Tensor()):
...             continue
...         print(k)
...
>>> model.state_dict()
OrderedDict([('encoder.weight',
              tensor([[-0.0457,  0.0184, -0.1925,  ..., -0.4014,  0.0417,  0.0677],
                      [-0.0872,  0.3312, -0.1791,  ..., -0.0611, -0.0574,  0.0399],
                      [-0.0354,  0.0738,  0.1283,  ...,  0.0583, -0.1608,  0.0356],
                      ...,
                      [-0.0272, -0.0195,  0.0592,  ..., -0.0929,  0.1029,  0.0353],
                      [ 0.0868, -0.0176,  0.0032,  ..., -0.0371,  0.0580, -0.0935],
                      [-0.0173,  0.1002,  0.0865,  ...,  0.0805, -0.0515, -0.1107]])),
             ('rnn.weight_ih_l0',
              tensor([[ 0.0972, -0.0317,  0.0298,  ..., -0.0327, -0.0138,  0.0246],
                      [-0.1144,  0.0478, -0.0114,  ..., -0.0063,  0.0431, -0.1188],
                      [-0.0512, -0.1021, -0.1000,  ..., -0.0644,  0.0082,  0.0574],
                      ...,
                      [ 0.1175,  0.1131, -0.1800,  ...,  0.0957,  0.0934, -0.0152],
                      [-0.2769,  0.1287, -0.2022,  ..., -0.0250, -0.0938, -0.2105],
                      [ 0.1502, -0.3340,  0.0776,  ..., -0.2602,  0.2005, -0.2711]])),
             ('rnn.weight_hh_l0',
              tensor([[ 0.2287, -0.0620, -0.0430,  ..., -0.1111, -0.0066,  0.0050],
                      [-0.0009, -0.0591, -0.0033,  ..., -0.0706,  0.0092,  0.0253],
                      [-0.0360, -0.0808, -0.1405,  ..., -0.1021,  0.0330,  0.0932],
                      ...,
                      [ 0.0590, -0.2034,  0.2074,  ...,  0.2251, -0.0614, -0.0880],
                      [-0.1015, -0.0074, -0.0671,  ...,  0.0575, -0.0062, -0.2099],
                      [ 0.0541, -0.0913, -0.1645,  ..., -0.1317, -0.0068,  0.0534]])),
             ('rnn.bias_ih_l0',
              tensor([ 0.0144, -0.0793, -0.0420, -0.0861, -0.0682, -0.0223, -0.0075, -0.1043,
                       0.0419, -0.1825, -0.0940,  0.0965,  0.0697, -0.0592,  0.0763, -0.0843,
                      -0.1854, -0.2361, -0.1074, -0.1172,  0.0720, -0.0062, -0.0500, -0.2390,
                       0.0262, -0.2024,  0.0471, -0.0894,  0.0213, -0.1953, -0.1161, -0.0432,
                       0.1293, -0.0425,  0.0828, -0.2454, -0.1084, -0.2029, -0.1443, -0.0401,
                      -0.0478, -0.0605,  0.0220, -0.0082,  0.0845, -0.0280, -0.0925,  0.0321,
                       0.1100, -0.0481,  0.0216, -0.0504,  0.0367,  0.2164, -0.1279,  0.0537,
                      -0.0473, -0.2325, -0.1595,  0.1271, -0.0540, -0.0356, -0.0206,  0.0801,
                       0.0798, -0.0983, -0.1156,  0.0839,  0.0378, -0.1887,  0.0037,  0.1409,
                      -0.1288, -0.1974, -0.0574, -0.0163,  0.0874, -0.1294, -0.0993, -0.0770,
                      -0.1622,  0.0743,  0.1004,  0.0088, -0.0792, -0.0406,  0.1587, -0.1898,
                      -0.0019, -0.0681,  0.0912, -0.2401,  0.1386,  0.0925,  0.1223, -0.0982,
                       0.0096, -0.0861, -0.1475, -0.1132, -0.1735, -0.1373, -0.2571,  0.0899,
                       0.0876,  0.1157, -0.0933, -0.0499,  0.0144, -0.1516, -0.0519, -0.0751,
                      -0.2357, -0.1167, -0.6221, -0.0698, -0.1047,  0.1703,  0.0686, -0.2197,
                      -0.1296, -0.0584,  0.0715,  0.0012,  0.0529,  0.0906, -0.0455, -0.0877,
                       0.0699,  0.0118,  0.1512,  0.0248, -0.0259, -0.2286, -0.0871, -0.0065,
                      -0.1769,  0.0712,  0.1087, -0.0688, -0.3563, -0.1799,  0.0400, -0.0217,
                      -0.1899,  0.0625,  0.0215, -0.1690, -0.0577, -0.1056, -0.1079,  0.0173,
                       0.1784,  0.0399,  0.1180, -0.0163, -0.0526,  0.2027, -0.1926, -0.0073,
                       0.0434, -0.1830,  0.0112, -0.1246,  0.1758, -0.1032,  0.1259, -0.1028,
                       0.1956,  0.1749,  0.0264,  0.0466, -0.1827,  0.0439, -0.3420, -0.0517,
                      -0.2790,  0.0575,  0.0201, -0.1194, -0.1307, -0.0874,  0.0021,  0.1243,
                      -0.0539, -0.0046,  0.0219,  0.1409,  0.1266,  0.0568, -0.0274,  0.1149,
                       0.0399, -0.2045, -0.1143, -0.0699,  0.1180, -0.0355, -0.1018, -0.1246,
                       0.0473, -0.3486, -0.3329, -0.2611, -0.3454, -0.2229,  0.3417, -0.1534,
                      -0.3461, -0.0911, -0.3025,  0.5777,  0.4430, -0.2570, -0.2787, -0.2716,
                      -0.2697,  0.3762, -0.2258, -0.4479, -0.0777, -0.2003, -0.2981, -0.2224,
                      -0.2643, -0.2340, -0.1565, -0.1358, -0.3508, -0.2465, -0.3113, -0.0707,
                      -0.0424, -0.3493, -0.2823,  0.4913, -0.0215, -0.2863, -0.3056, -0.3720,
                       0.0733, -0.3475,  0.0596, -0.1582,  0.4496, -0.2384, -0.4051,  0.0380,
                      -0.2419, -0.2446, -0.2953, -0.2122, -0.0505, -0.0364, -0.4286, -0.3075,
                      -0.2368, -0.3168, -0.2424, -0.3597, -0.2850, -0.3733, -0.1379, -0.1114,
                       0.2531, -0.3258, -0.2310, -0.3043,  0.3885, -0.0870, -0.1748,  0.1273,
                      -0.3281, -0.3919, -0.0550, -0.3299, -0.2978, -0.2947, -0.2891, -0.2433,
                      -0.3826,  0.1313, -0.0524, -0.4339, -0.3291, -0.2225, -0.0373, -0.3352,
                      -0.2505, -0.3623, -0.0485, -0.2776, -0.0742,  0.6599, -0.0194,  0.0758,
                      -0.2692, -0.2092, -0.3961, -0.3382,  0.1802,  0.4904, -0.3183, -0.1393,
                       0.5425,  0.4332, -0.2110, -0.2480, -0.3100, -0.2772, -0.2318, -0.2781,
                      -0.3601, -0.3438, -0.2603, -0.1991, -0.2605,  0.0942,  0.6380, -0.3324,
                      -0.3143, -0.0808, -0.1107, -0.2797,  0.0642,  0.1806, -0.3168, -0.2713,
                       0.2112, -0.2521,  0.1164, -0.2623,  0.0791, -0.3252, -0.1436, -0.4149,
                      -0.2460, -0.1724,  0.0087, -0.4076, -0.2405, -0.3905, -0.1295, -0.1202,
                      -0.3147, -0.2562, -0.2296, -0.3411, -0.0639, -0.1123, -0.2471, -0.0725,
                       0.6649, -0.0604,  0.0653, -0.4084, -0.1849, -0.1789, -0.2831, -0.0077,
                      -0.4063, -0.3072, -0.1073, -0.2770, -0.0953, -0.2620,  0.1172, -0.1541,
                      -0.1059, -0.2824,  0.0199, -0.1380,  0.0759, -0.2625, -0.2614, -0.2891,
                      -0.3528,  0.0086,  0.2335, -0.2500, -0.1913, -0.0932,  0.0224,  0.6150,
                       0.1109, -0.1435, -0.0530, -0.2122, -0.0886,  0.0947, -0.0994,  0.1032,
                      -0.2109, -0.2336, -0.1863, -0.0983, -0.0242, -0.2879, -0.1916, -0.2748,
                       0.0197, -0.0531, -0.0949,  0.0643, -0.2226,  0.2016, -0.2279, -0.2524,
                       0.1606, -0.0130, -0.2415,  0.5402,  0.7405,  0.1128, -0.0593, -0.1247,
                      -0.0983, -0.1740, -0.1019, -0.0490,  0.1676, -0.7532,  0.1926,  0.1296,
                      -0.2274, -0.6406,  0.7812, -0.0933,  0.0562,  0.3265,  0.0785, -0.2774,
                      -0.2537, -0.1287, -0.2985, -0.3605, -0.3541,  0.2278,  0.0783,  0.0582,
                       0.0248,  0.2872,  0.5931, -0.3243, -0.1672, -0.1708,  0.0962, -0.3378,
                       0.0313,  0.1229, -0.2025, -0.2086,  0.1672,  0.0674,  0.0377, -0.0235,
                      -0.1062,  0.0917, -0.0846,  0.0656, -0.2207,  0.0494,  0.1529, -0.2732,
                      -0.8790, -0.0667, -0.1218, -0.0114,  0.2885,  0.2642, -0.4396, -0.0590,
                      -0.0385, -0.2683,  0.3570, -0.0360, -0.3707,  0.2044,  0.2078,  0.4082,
                      -0.3161,  0.2112,  0.1676,  0.0238, -0.0159, -0.0044,  0.1726,  0.1417,
                       0.0674,  0.1568,  0.1726,  0.3920, -0.2925,  0.1091, -0.5577, -0.4583,
                      -0.1027,  0.1850,  0.1877, -0.0076,  0.0934, -0.3189, -0.0679, -0.4643,
                      -0.3016,  0.7665, -0.0646,  0.1132, -0.2600, -0.2067, -0.2174,  0.2619,
                       0.3179, -0.1882, -0.3814,  0.0702, -0.0466, -0.0898, -0.1256, -0.1069,
                       0.0581, -0.0392,  0.0899,  0.0553, -0.2099,  0.8845,  0.0105,  0.2478,
                      -0.5980, -0.4339, -0.2928,  0.2147,  0.0814,  0.0437,  0.1101, -0.0728,
                       0.4297,  0.5435, -0.7064, -0.1031,  0.1672,  0.1999,  0.2444, -0.2750,
                      -0.2588,  0.0009, -0.1309, -0.2676,  0.4205, -0.1035,  0.2284,  0.0417,
                      -0.3413,  0.3274,  0.3618,  0.2200, -0.7810,  0.0561,  0.4125, -0.2559,
                      -0.0156, -0.0576, -0.8028,  0.1947, -0.3021, -0.3670,  0.3564, -0.1740,
                       0.0664, -0.0489,  0.3384, -0.1988, -0.6642, -0.3580, -0.0694,  0.0670,
                      -0.1255,  0.1737, -0.4623, -0.2208,  0.2510,  0.3341,  0.4476, -0.5895,
                      -0.0366, -0.7915,  0.3908, -0.4839, -0.6301, -0.5891, -0.4927,  0.1082,
                      -0.2633, -0.2918,  0.1866,  0.4702, -0.5056,  0.1034, -0.0159,  0.2667])),
             ('rnn.bias_hh_l0',
              tensor([ 0.0700,  0.0530, -0.0546, -0.0633, -0.1557,  0.0683,  0.0598, -0.0439,
                       0.0023, -0.1596, -0.2156,  0.0725,  0.0241,  0.0099, -0.0246, -0.0406,
                      -0.1102, -0.2544, -0.1519, -0.0814,  0.1224,  0.0923, -0.0495, -0.2852,
                      -0.0237, -0.2035,  0.0876, -0.1384,  0.0042, -0.1039, -0.1250, -0.0025,
                       0.1374, -0.0390,  0.1137, -0.2211, -0.0949, -0.1923, -0.2627,  0.0033,
                      -0.1758,  0.0055,  0.0244,  0.0125,  0.0102,  0.0509, -0.0677, -0.0255,
                      -0.0224, -0.0216, -0.0120, -0.1088,  0.1125,  0.3008, -0.2164,  0.0453,
                      -0.1119, -0.1958, -0.2170,  0.0107, -0.0100, -0.1226, -0.0338,  0.0057,
                       0.1430, -0.0535, -0.0916,  0.0267,  0.1194, -0.1623,  0.0969,  0.1308,
                      -0.1098, -0.1405,  0.0315,  0.0633,  0.1166, -0.0859, -0.2361, -0.1532,
                      -0.1683,  0.1939,  0.1003, -0.0425, -0.0388,  0.0091,  0.1574, -0.1258,
                      -0.1017, -0.0302,  0.1334, -0.2858,  0.1491,  0.1401,  0.0990, -0.2024,
                       0.0464, -0.1184, -0.0865, -0.1127, -0.1967, -0.1796, -0.2499,  0.1012,
                       0.0903,  0.1605, -0.1311, -0.1212,  0.0268, -0.0822, -0.0149, -0.0425,
                      -0.2854, -0.0955, -0.5535, -0.1541, -0.1665,  0.1730,  0.1824, -0.2624,
                      -0.2024, -0.0016,  0.0922, -0.0314, -0.0640,  0.1544, -0.1073, -0.0801,
                       0.0880, -0.0869,  0.1546, -0.0550, -0.0581, -0.2942, -0.0583,  0.0887,
                      -0.1752,  0.0307,  0.1601, -0.0657, -0.2754, -0.1422, -0.0263, -0.0670,
                      -0.1525,  0.0297,  0.0495, -0.1617,  0.0345, -0.0423, -0.1194, -0.0038,
                       0.1026,  0.0307,  0.0612,  0.0292, -0.0782,  0.1004, -0.2080,  0.0448,
                       0.0659, -0.1310,  0.0565, -0.0963,  0.1325, -0.1088,  0.0495, -0.1169,
                       0.2215,  0.0728,  0.0361,  0.0621, -0.2265,  0.0580, -0.3660,  0.0636,
                      -0.2942,  0.0947,  0.0960, -0.1250, -0.1215, -0.0866,  0.0506,  0.1671,
                       0.0788,  0.0308,  0.0331,  0.0637,  0.0380,  0.0297,  0.0278,  0.0977,
                       0.0869, -0.2041, -0.1104,  0.0572,  0.0762, -0.1684, -0.0161, -0.0872,
                       0.0981, -0.3519, -0.2147, -0.3127, -0.2357, -0.1989,  0.3029, -0.2442,
                      -0.2553, -0.1049, -0.3058,  0.5833,  0.4336, -0.2933, -0.3094, -0.2499,
                      -0.3475,  0.3830, -0.1899, -0.3440, -0.0729, -0.2965, -0.2327, -0.3174,
                      -0.3340, -0.2109, -0.1145, -0.2206, -0.3497, -0.3586, -0.3472, -0.1474,
                      -0.0414, -0.3482, -0.2275,  0.5515, -0.0410, -0.2160, -0.3106, -0.4351,
                       0.1626, -0.3855,  0.0583, -0.1387,  0.4697, -0.2237, -0.3437, -0.0249,
                      -0.2879, -0.2782, -0.3074, -0.2119,  0.0516,  0.0603, -0.4474, -0.2335,
                      -0.2559, -0.2411, -0.3384, -0.4095, -0.2406, -0.3217, -0.2412, -0.0874,
                       0.2432, -0.3390, -0.2586, -0.2725,  0.3270, -0.0137, -0.1176,  0.1250,
                      -0.2264, -0.4034, -0.0948, -0.3364, -0.3906, -0.3258, -0.4016, -0.2118,
                      -0.2959,  0.1023, -0.1243, -0.3431, -0.3225, -0.1400, -0.0166, -0.3479,
                      -0.3060, -0.2574, -0.0154, -0.2665,  0.0205,  0.7202,  0.0708,  0.0832,
                      -0.3545, -0.2194, -0.3728, -0.3149,  0.1811,  0.4192, -0.3150, -0.1604,
                       0.5307,  0.3804, -0.1066, -0.2044, -0.2663, -0.2103, -0.1760, -0.2753,
                      -0.3660, -0.4223, -0.3748, -0.2773, -0.1918,  0.0863,  0.6321, -0.2142,
                      -0.3189, -0.0486, -0.0541, -0.3168,  0.0714,  0.2738, -0.1838, -0.3150,
                       0.2568, -0.2535,  0.0923, -0.2312,  0.0098, -0.2905, -0.2602, -0.3784,
                      -0.3002, -0.1411, -0.1143, -0.3462, -0.2240, -0.3426, -0.2160, -0.1591,
                      -0.2412, -0.3286, -0.1881, -0.2793, -0.1186,  0.0221, -0.1982,  0.0574,
                       0.6228, -0.1421,  0.0331, -0.3779, -0.1462, -0.2285, -0.3447, -0.1220,
                      -0.3639, -0.3181, -0.0711, -0.3456, -0.0956, -0.1768,  0.1412, -0.1525,
                      -0.1606, -0.2062, -0.0645, -0.1499,  0.0180, -0.2492, -0.3595, -0.2374,
                      -0.2520,  0.0009,  0.3176, -0.2707, -0.2339, -0.1584,  0.0328,  0.6172,
                       0.0809, -0.0947,  0.0013, -0.2078, -0.1467,  0.0962, -0.1595,  0.1058,
                      -0.2634, -0.2825, -0.1788, -0.1740,  0.0444, -0.2943, -0.1808, -0.2458,
                       0.2219,  0.0022,  0.2641,  0.1377,  0.0045,  0.2725, -0.1233, -0.2504,
                       0.4256,  0.4484,  0.0097,  0.2429, -0.2394, -0.2368, -0.3634, -0.2688,
                      -0.2058, -0.0307, -0.3946,  0.3034, -0.1649, -0.1218, -0.1390,  0.0902,
                      -0.0834,  0.2003,  0.0536, -0.5251, -0.1889,  0.0608, -0.2641, -0.4220,
                      -0.0686,  0.3276, -0.3451,  0.1060, -0.3923,  0.0759, -0.2581,  0.3060,
                       0.0559, -0.2578,  0.4882, -0.3604, -0.0521, -0.3547,  0.1998, -0.3011,
                       0.2273,  0.2519,  0.2683, -0.2827, -0.3472, -0.6051,  0.0051, -0.5640,
                      -0.1599,  0.3622, -0.3005,  0.5446, -0.1706, -0.2439,  0.5143,  0.3568,
                      -0.3970, -0.1912,  0.1266,  0.3827,  0.0659,  0.1035, -0.2687, -0.3743,
                       0.0300, -0.1452, -0.3304, -0.2533,  0.5085,  0.0847,  0.0648,  0.2627,
                      -0.2031,  0.4301,  0.6882,  0.3791,  0.0342,  0.3908,  0.5324,  0.3513,
                       0.3132,  0.3881,  0.4512,  0.2533, -0.6073,  0.1622, -0.2008, -0.0652,
                      -0.0729,  0.3899,  0.2749,  0.1890, -0.1019,  0.0027,  0.1059, -0.3369,
                      -0.3578,  0.1196,  0.2364,  0.1845, -0.4021,  0.1472,  0.0305, -0.2825,
                      -0.0036, -0.0607, -0.2439, -0.2341, -0.1863,  0.6244, -0.0911, -0.1975,
                      -0.0629,  0.4330,  0.2853,  0.1436,  0.3281,  0.3033, -0.0877,  0.1831,
                      -0.2726, -0.1117, -0.4887,  0.3054,  0.4592,  0.1154, -0.3014, -0.2294,
                       0.1176,  0.3833, -0.4685,  0.3033, -0.2465,  0.0217,  0.2643, -0.2494,
                      -0.0827,  0.4635, -0.4060, -0.1581, -0.0329, -0.2480, -0.0297, -0.3526,
                      -0.0084,  0.3296,  0.3235,  0.1419, -0.0860, -0.5586, -0.2659, -0.4063,
                      -0.3924,  0.1884, -0.3097,  0.1889, -0.4515, -0.1389, -0.4714, -0.1340,
                      -0.5252,  0.5510,  0.3939, -0.4015, -0.0971, -0.3539, -0.2085, -0.2789,
                      -0.2623, -0.4814,  0.2836, -0.2473,  0.0075,  0.2242,  0.3998, -0.2934,
                      -0.2900, -0.4914,  0.5509,  0.3887, -0.4487, -0.4464, -0.4527,  0.3377,
                      -0.3883, -0.0989,  0.3015,  0.3452, -0.2648, -0.1860,  0.0379,  0.3732])),
             ('decoder.weight',
              tensor([[ 0.0588,  0.0777, -0.0657,  ..., -0.1593, -0.0995,  0.2933],
                      [-0.0457,  0.1932, -0.1245,  ..., -0.1464, -0.0727,  0.2412],
                      [-0.2169,  0.0635,  0.0648,  ...,  0.0127,  0.1388,  0.0272],
                      ...,
                      [-0.0064, -0.0460,  0.0070,  ..., -0.0756, -0.0825, -0.1171],
                      [-0.0628,  0.0032, -0.0136,  ..., -0.0254,  0.0177, -0.0116],
                      [ 0.0472,  0.0049, -0.0065,  ...,  0.0433,  0.0409, -0.0477]])),
             ('decoder.bias',
              tensor([ 3.9778,  1.8732,  0.0371,  ..., -0.0587, -0.0355, -0.0522]))])
>>> for k in model.state_dict():
...     print k
...
>>> for k in model.state_dict():
...     print(k)
...
>>> for k, v in model.state_dict().items():
...     print(k, v)
...
>>> for k, v in model.state_dict().items():
...     print(k, v.size().product())
...
>>> for k, v in model.state_dict().items():
...     print(k, v.size().prod())
...
>>> for k, v in model.state_dict().items():
...     print(k, v.size().prod())
...
>>> import math
>>> torch.product
>>> for k, v in model.state_dict().items():
...     print(k, torch.prod(v.size()))
...
>>> s = v.size()
>>> s
torch.Size([33278, 200])
>>> s.numel()
6655600
>>> for k, v in model.state_dict().items():
...     print(k, torch.prod(v.numel()))
...
>>> for k, v in model.state_dict().items():
...     print(k, v.numel())
...
>>> count_layer_parameters(model)
>>> from nlpia2.torch_utils import count_layer_parameters
>>> count_layer_parameters(model)
[6655600, 120000, 120000, 600, 600, 6655600, 33278]
>>> hist
>>> %run ../../../torch_utils
>>> %run ../../../torch_utils.py
>>> %run ../../torch_utils.py
>>> count_layer_parameters(model)
[6655600, 120000, 120000, 600, 600, 6655600, 33278]
>>> describe_model(model)
>>> describe_model(model)
>>> %run ../../torch_utils.py
>>> describe_model(model)
>>> %run ../../torch_utils.py
>>> describe_model(model)
                  learned_params  total_params          size
name                                                        
encoder.weight           6655600       6655600  (33278, 200)
rnn.weight_ih_l0          120000        120000    (600, 200)
rnn.weight_hh_l0          120000        120000    (600, 200)
rnn.bias_ih_l0               600           600        (600,)
rnn.bias_hh_l0               600           600        (600,)
decoder.weight           6655600       6655600  (33278, 200)
decoder.bias               33278         33278      (33278,)
>>> hist -o -p -f hist/ch08_rnn_word_model_count_params.hist.md
