import pandas as pd

import numpy as np

from tqdm import tqdm

num_vecs = 100000

num_radii = 20

num_dim_list = [2, 4, 8, 18, 32, 64, 128]

radii = np.array(list(range(1, num_radii + 1)))

radii = radii / len(radii)

counts = np.zeros((len(radii), len(num_dims_list)))

rand = np.random.rand

for j, num_dims in enumerate(tqdm(num_dim_list)):
    x = rand(num_vecs, num_dims)
    denom = (1. / np.linalg.norm(x, axis=1))  # <1>
    x *= denom.reshape(-1, 1).dot(np.ones((1, x.shape[1])))
    for i, r in enumerate(radii):
        mask = (-r < x) & (x < r)
        counts[i, j] = (mask.sum(axis=1) == mask.shape[1]).sum()

df = pd.DataFrame(counts, index=radii, columns=num_dim_list) / num_vecs

df = df.round(2)

df[df == 0] = ''

df
