import pandas as pd

import spacy

nlp = spacy.load('en_core_web_md')  # <1>

text = 'right ones in the right order you can nudge the world'

doc = nlp(text)

df = pd.DataFrame([
   {k: getattr(t, k) for k in 'text pos_'.split()}
   for t in doc])

pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')

def corr(a, b):
   """ Compute the Pearson correlation coefficient R """
   a = a - np.mean(a)
   b = b - np.mean(b)
   return sum(a * b) / np.sqrt(sum(a*a) * sum(b*b))

nlp = spacy.load('en_core_web_md')

quote = "The right word may be effective, but no word was ever" \
   " as effective as a rightly timed pause."

tagged_words = {
   t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
   for t in nlp(quote)}

df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])

print(df_quote)

inpt = list(df_quote.loc['ADV'])

print(inpt)

kernel = [.5, .5]  # <1>

output = []

for i in range(len(inpt) - 1):  # <2>
   z = 0
   for k, weight in enumerate(kernel):  # <3>
       z = z + weight * inpt[i + k]
   output.append(z)

print(f'inpt:\n{inpt}')

print(f'len(inpt): {len(inpt)}')

print(f'output:\n{[int(o) if int(o)==o else o for o in output]}')

print(f'len(output): {len(output)}')

import pandas as pd

from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 120  # <1>

import seaborn as sns

sns.set_theme('paper')  # <2>

df = pd.DataFrame([inpt, output], index=['inpt', 'output']).T

ax = df.plot(style=['+-', 'o:'], linewidth=3)

def convolve(inpt, kernel):
   output = []
   for i in range(len(inpt) - len(kernel) + 1):  # <1>
       output.append(
           sum(
               [
                   inpt[i + k] * kernel[k]
                   for k in range(len(kernel))  # <2>
               ]
           )
       )
   return output

tags = 'ADV ADJ VERB NOUN'.split()

tagged_words = [
   [tok.text] + [int(tok.pos_ == tag) for tag in tags]  # <1>
   for tok in nlp(quote)]  # <2>

df = pd.DataFrame(tagged_words, columns=['token'] + tags).T

print(df)

import torch

x = torch.tensor(
    df.iloc[1:].astype(float).values,
    dtype=torch.float32)  # <1>

x = x.unsqueeze(0) # <2>

kernel = pd.DataFrame(
          [[1, 0, 0.],
           [0, 0, 0.],
           [0, 1, 0.],
           [0, 0, 1.]], index=tags)

print(kernel)

kernel = torch.tensor(kernel.values, dtype=torch.float32)

kernel = kernel.unsqueeze(0)  # <1>

conv = torch.nn.Conv1d(in_channels=4,
                    out_channels=1,
                    kernel_size=3,
                    bias=False)

conv.load_state_dict({'weight': kernel})

print(conv.weight)

y = np.array(conv.forward(x).detach()).squeeze()

df.loc['y'] = pd.Series(y)

df

from nlpia2.init import maybe_download

url = 'https://upload.wikimedia.org/wikipedia/' \

filepath = maybe_download(url)  # <1>

print(filepath)

from scipy.io import wavfile

sample_rate, audio = wavfile.read(filepath)

print(f'sample_rate: {sample_rate}')

print(f'audio:\n{audio}')

pd.options.display.max_rows = 7

audio = audio[:sample_rate * 2]  # <1>

audio = np.abs(audio - audio.max() / 2) - .5  # <2>

audio = audio / audio.max()  # <3>

audio = audio[::sample_rate // 400]  # <4>

audio = pd.Series(audio, name='audio')

audio.index = 1000 * audio.index / sample_rate  # <5>

audio.index.name = 'time (ms)'

print(f'audio:\n{audio}')

kernel = [-1] * 24 + [1] * 24 + [-1] * 24  # <1>

kernel = pd.Series(kernel, index=2.5 * np.arange(len(kernel)))

kernel.index.name = 'Time (ms)'

ax = kernel.plot(linewidth=3, ylabel='Kernel weight')

kernel = np.array(kernel) / sum(np.abs(kernel))  # <1>

pad = [0] * (len(kernel) // 2)  # <2>

isdot = convolve(audio.values, kernel)

isdot =  np.array(pad[:-1] + list(isdot) + pad)  # <3>

df = pd.DataFrame()

df['audio'] = audio

df['isdot'] = isdot - isdot.min()

ax = df.plot()

isdot = np.convolve(audio.values, kernel, mode='same')  # <1>

df['isdot'] = isdot - isdot.min()

ax = df.plot()

def describe_model(model):  # <1>
    state = model.state_dict()
    names = state.keys()
    weights = state.values()
    params = model.parameters()

    df = pd.DataFrame()

    df['name'] = list(state.keys())

    df['all'] = p.numel(),
    df['learned'] = [
        p.requires_grad  # <2>
        for p in params],  # <3>
    size=p.size(),
    )
