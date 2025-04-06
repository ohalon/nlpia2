from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression  # , Lasso
from tqdm import tqdm

DATA_DIR = Path('.nlpia2-data')
df = pd.read_csv(DATA_DIR / 'baby-names-region.csv.gz')

df = df.sample(1_000_000, random_state=1989)
np.random.seed(451)
istrain = np.random.rand(len(df)) < .9

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(1, 3))
vectorizer.fit(df['name'][istrain])
vecs = vectorizer.transform(df['name'])

""" Use the count rather than freq in order to work with modern names

>>> model = LogisticRegression(class_weight='balanced', max_iter=10000)
>>> model.fit(vecs[istrain], df['sex'][istrain], sample_weight=df['count'][istrain])
LogisticRegression(class_weight='balanced', max_iter=10000)
>>> model.score(vecs[istrain], df['sex'][istrain], df['count'][istrain])
0.9826107512114707
>>> model.score(vecs[~istrain], df['sex'][~istrain], df['count'][~istrain])
0.983129802663326
>>> model.score(vecs[~istrain], df['sex'][~istrain])
0.9395149985982618
"""


model = LogisticRegression(class_weight='balanced', max_iter=10000)
model.fit(vecs[istrain], df['sex'][istrain], sample_weight=df['count'][istrain])
model.score(vecs[istrain], df['sex'][istrain], df['count'][istrain])
model.score(vecs[~istrain], df['sex'][~istrain], df['count'][~istrain])


from torch import nn, optim, manual_seed, tensor
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# https://medium.com/deep-learning-study-notes/multi-layer-perceptron-mlp-in-pytorch-21ea46d50e62

# create custom dataset class


class CustomDataset(Dataset):

    def __init__(self, X, y):
        self.vectors = X
        self.labels = y

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = tensor(self.labels[idx],
                       dtype=torch.float32)  # or torch.long or torch.float64
        vector = tensor(np.asarray(self.vectors[idx].todense()).squeeze(),
                        dtype=torch.long)
        return vector, label


NUM_FEATURES = len(vectorizer.get_feature_names())
BATCH_SIZE = 256 * 256

# https://medium.com/biaslyai/pytorch-linear-and-logistic-regression-models-5c5f0da2cb9


class NeuralLogisticRegression(torch.nn.Module):
    loss = torch.nn.BCELoss(size_average=True)

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(NUM_FEATURES, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


model = NeuralLogisticRegression()
loss_fun = NeuralLogisticRegression.loss


# for epoch in tqdm(range(20)):
#     model.train()
#     optimizer.zero_grad()
#     # Forward pass
#     y_pred = model(x_data)
#     # Compute Loss
#     loss = loss_fun(y_pred, y_data)
#     # Backward pass
#     loss.backward()
#     optimizer.step()

manual_seed(1989)
dataset = CustomDataset(vecs, (df['sex'] == 'F').astype(int).values)
print(next(iter(dataset)))
trainloader = DataLoader(
    dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)

p.train()
epoch_losses = []
for epoch in tqdm(range(10)):
    losses = []
    for batch_num, input_data in tqdm(enumerate(trainloader)):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device).long()

        output = p(x)
        loss = loss_function(output, y)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if not batch_num % 2000:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    epoch_losses.append(sum(losses) / len(losses))
    print('Epoch {} | Loss {:6.2f}'.format(epoch + 1, epoch_losses[-1]))


class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(num_features, 10),
            nn.Linear(10, 2),
            # nn.SoftMax(2)
        )
        self.linear = nn.Linear(num_features, 2)

    def forward(self, x):
        out = self.linear(x)
        return out


manual_seed(1989)
dataset = CustomDataset(vecs, (df['sex'] == 'F').astype(int).values)
print(next(iter(dataset)))
trainloader = DataLoader(
    dataset, batch_size=32, shuffle=True, num_workers=4)

p = Perceptron()
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(p.parameters(), lr=4e-5)


p.train()
epoch_losses = []
for epoch in range(10):
    losses = []
    for batch_num, input_data in enumerate(trainloader):
        optimizer.zero_grad()
        x, y = input_data
        x = x.to(device).float()
        y = y.to(device).long()

        output = p(x)
        loss = loss_function(output, y)
        loss.backward()
        losses.append(loss.item())

        optimizer.step()

        if not batch_num % 200:
            print('\tEpoch %d | Batch %d | Loss %6.2f' % (epoch, batch_num, loss.item()))
    epoch_losses.append(sum(losses) / len(losses))
    print('Epoch {} | Loss {:6.2f}'.format(epoch + 1, epoch_losses[-1]))
