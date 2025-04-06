import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

palette = sns.color_palette("muted")
sns.set_theme()

vecs = pd.DataFrame([
    [0, 0, 1, 0],
    [0, 0, 2, 1],
    [2, 1, -1, -1]
], columns=['x0', 'y0', 'x', 'y'])
vecs['color'] = palette[:3]
vecs['label'] = [f'vec{i}' for i in range(1, len(vecs))] + [" "]
vecs['ls'] = 'solid solid dashed'.split()
fig, ax = plt.subplots()
for i, row in vecs.iterrows():
    ax.quiver(
        row['x0'], row['y0'], row['x'], row['y'], ec=row['color'],
        linewidth=2, fc='none',
        angles='xy', scale_units='xy', scale=1, ls=row['ls'])
    ax.annotate(
        row['label'], (row['x'], row['y']),
        color=row['color'], verticalalignment='top'
    )
plt.xlim(-1, 3)
plt.ylim(-1, 2)
plt.xlabel('X (e.g. frequency of word "vector")')
plt.ylabel('Y (e.g. frequency of word "space")')
plt.show()
