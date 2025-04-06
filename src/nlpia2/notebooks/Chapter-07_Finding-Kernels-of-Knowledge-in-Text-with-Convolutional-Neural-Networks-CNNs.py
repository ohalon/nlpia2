#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-07_Finding-Kernels-of-Knowledge-in-Text-with-Convolutional-Neural-Networks-CNNs`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-07_Finding-Kernels-of-Knowledge-in-Text-with-Convolutional-Neural-Networks-CNNs.adoc)

# #### 

# In[ ]:


import pandas as pd
import spacy
nlp = spacy.load('en_core_web_md')  # <1>
text = 'right ones in the right order you can nudge the world'
doc = nlp(text)
df = pd.DataFrame([
   {k: getattr(t, k) for k in 'text pos_'.split()}
   for t in doc])


# #### 

# In[ ]:


pd.get_dummies(df, columns=['pos_'], prefix='', prefix_sep='')


# #### .Python implementation of correlation

# In[ ]:


def corr(a, b):
   """ Compute the Pearson correlation coefficient R """
   a = a - np.mean(a)
   b = b - np.mean(b)
   return sum(a * b) / np.sqrt(sum(a*a) * sum(b*b))
a = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2])
b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
corr(a, b)


# #### .Python implementation of correlation

# In[ ]:


corr(a, a)


# #### .Tag a quote with parts of speech

# In[ ]:


nlp = spacy.load('en_core_web_md')
quote = "The right word may be effective, but no word was ever" \
   " as effective as a rightly timed pause."
tagged_words = {
   t.text: [t.pos_, int(t.pos_ == 'ADV')]  # <1>
   for t in nlp(quote)}
df_quote = pd.DataFrame(tagged_words, index=['POS', 'ADV'])
print(df_quote)


# #### .Tag a quote with parts of speech

# In[ ]:


inpt = list(df_quote.loc['ADV'])
print(inpt)


# #### .Tag a quote with parts of speech

# In[ ]:


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


# #### .Line plot of input (is_adv) and output (adverbiness)

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams['figure.dpi'] = 120  # <1>
import seaborn as sns
sns.set_theme('paper')  # <2>
df = pd.DataFrame([inpt, output], index=['inpt', 'output']).T
ax = df.plot(style=['+-', 'o:'], linewidth=3)


# #### .Line plot of input (is_adv) and output (adverbiness)

# In[ ]:


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


# #### 

# In[ ]:


tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = [
   [tok.text] + [int(tok.pos_ == tag) for tag in tags]  # <1>
   for tok in nlp(quote)]  # <2>
df = pd.DataFrame(tagged_words, columns=['token'] + tags).T
print(df)


# #### 

# In[ ]:


import torch
x = torch.tensor(
    df.iloc[1:].astype(float).values,
    dtype=torch.float32)  # <1>
x = x.unsqueeze(0) # <2>


# #### 

# In[ ]:


kernel = pd.DataFrame(
          [[1, 0, 0.],
           [0, 0, 0.],
           [0, 1, 0.],
           [0, 0, 1.]], index=tags)
print(kernel)


# #### .Load hard-coded weights into a Conv1d layer

# In[ ]:


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


# #### .Download secret message

# In[ ]:


from nlpia2.init import maybe_download
url = 'https://upload.wikimedia.org/wikipedia/' \


# #### .Download secret message

# In[ ]:


filepath = maybe_download(url)  # <1>
filepath


# #### .Download secret message

# In[ ]:


from scipy.io import wavfile
sample_rate, audio = wavfile.read(filepath)
print(f'sample_rate: {sample_rate}')
print(f'audio:\n{audio}')


# #### .Download secret message

# In[ ]:


pd.options.display.max_rows = 7
audio = audio[:sample_rate * 2]  # <1>
audio = np.abs(audio - audio.max() / 2) - .5  # <2>
audio = audio / audio.max()  # <3>
audio = audio[::sample_rate // 400]  # <4>
audio = pd.Series(audio, name='audio')
audio.index = 1000 * audio.index / sample_rate  # <5>
audio.index.name = 'time (ms)'
print(f'audio:\n{audio}')


# #### .Dot detecting kernel

# In[ ]:


kernel = [-1] * 24 + [1] * 24 + [-1] * 24  # <1>
kernel = pd.Series(kernel, index=2.5 * np.arange(len(kernel)))
kernel.index.name = 'Time (ms)'
ax = kernel.plot(linewidth=3, ylabel='Kernel weight')


# #### .Dot detecting kernel

# In[ ]:


kernel = np.array(kernel) / sum(np.abs(kernel))  # <1>
pad = [0] * (len(kernel) // 2)  # <2>
isdot = convolve(audio.values, kernel)
isdot =  np.array(pad[:-1] + list(isdot) + pad)  # <3>
df = pd.DataFrame()
df['audio'] = audio
df['isdot'] = isdot - isdot.min()
ax = df.plot()


# #### .Dot detecting kernel

# In[ ]:


isdot = np.convolve(audio.values, kernel, mode='same')  # <1>
df['isdot'] = isdot - isdot.min()
ax = df.plot()


# #### .Load news posts

# In[ ]:


df = pd.read_csv(HOME_DATA_DIR / 'news.csv')
df = df[['text', 'target']]  # <1>
print(df)


# #### .Learn your embeddings from scratch

# In[ ]:


from torch import nn

embedding = nn.Embedding(
    num_embeddings=2000,  # <1>
    embedding_dim=64,  # <2>
    padding_idx=0)
----
<1> your vocab must be the same same as in your tokenizer
<2> 50-100 dimensions are fine for small vocabularies and corpora

The embedding layer will be the first layer in your CNN.
That will convert your token IDs into their own unique 64-D word vectors.
And backpropagation during training will adjust the weights in each dimension for each word to match 64 different ways that words can be used to talk about news-worthy disasters.
These embeddings won't represent the complete meaning of words the way the FastText and GloVe vectors did in chapter 6.
These embeddings are good for only one thing, determining if a Tweet contains newsworthy disaster information or not.

Finally you can train your CNN to see how well it will do on an extremely narrow dataset like the Kaggle disaster tweets dataset.
Those hours of work crafting a CNN will pay off with super-fast training time and impressive accuracy.

.Learn your embeddings from scratch
[source,python]
----
from nlpia2.ch07.cnn.train79 import Pipeline  # <1>

pipeline = Pipeline(
    vocab_size=2000,
    embeddings=(2000, 64),
    epochs=7,
    torch_random_state=433994,  # <2>
    split_random_state=1460940,
)

pipeline = pipeline.train()
----
<1> nlpia2/src/nlpia2/ch07/cnn/train79.py (https://gitlab.com/tangibleai/nlpia2/-/tree/main/src/nlpia2/ch07/cnn/train79.py)
<2> set random seeds so others can reproduce your results

[source,text]
----
Epoch: 1, loss: 0.66147, Train accuracy: 0.61392, Test accuracy: 0.63648
Epoch: 2, loss: 0.64491, Train accuracy: 0.69712, Test accuracy: 0.70735
Epoch: 3, loss: 0.55865, Train accuracy: 0.73391, Test accuracy: 0.74278
Epoch: 4, loss: 0.38538, Train accuracy: 0.76558, Test accuracy: 0.77165
Epoch: 5, loss: 0.27227, Train accuracy: 0.79288, Test accuracy: 0.77690
Epoch: 6, loss: 0.29682, Train accuracy: 0.82119, Test accuracy: 0.78609
Epoch: 7, loss: 0.23429, Train accuracy: 0.82951, Test accuracy: 0.79003
----

After only 7 passes through your training dataset you achieved 79% accuracy on your test set.
And on modern laptop CPU this should take less than a minute.
And you kept the overfitting to a minimum by minimizing the total parameters in your model.
The CNN uses very few parameters compared to the embedding layer.

get_ipython().run_line_magic('pinfo', 'longer')

.Continue training
[source,python]
----
pipeline.epochs = 13  # <1>
pipeline = pipeline.train()
----
<1> 7 + 13 will give you 20 total epochs of training

[source,python]


# #### 

# In[ ]:


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


# #### .Make room for GloVE embeddings

# In[ ]:


from torch import nn
embedding = nn.Embedding(
    num_embeddings=2000,  # <1>
    embedding_dim=50,  # <2>
    padding_idx=0)


# #### .Make room for GloVE embeddings

# In[ ]:


from nessvec.files import load_vecs_df
glove = load_vecs_df(HOME_DATA_DIR / 'glove.6B.50d.txt')
zeroes = [0.] * 50
embed = []
for tok in vocab:  # <1>
    if tok in glove.index:
        embed.append(glove.loc[tok])
    else:
        embed.append(zeros.copy())  # <2>
embed = np.array(embed)
embed.shape


# #### .Make room for GloVE embeddings

# In[ ]:


pd.Series(vocab)


# #### .Initialize your embedding layer with GloVE vectors

# In[ ]:


embed = torch.Tensor(embed)  # <1>
print(f'embed.size(): {embed.size()}')
embed = nn.Embedding.from_pretrained(embed, freeze=False)  # <2>
print(embed)
----
<1> convert Pandas DataFrame to a torch.Tensor
<2> freeze=False allows your Embedding layer to fine-tune your embeddings


==== Detecting meaningful patterns

How you say something, the order of the words, makes a big difference.
You combine words to create patterns that mean something significant to you, so that you can convey that meaning to someone else.

If you want your machine to be a meaningful natural language processor, it will need to be able to detect more than just the presence or absence of particular tokens.
You want your machine to detect meaningful patterns hidden within word sequences.footnote:[_International Association of Facilitators Handbook_, https://books.google.com/books?id=TgWsY7oSgtsC&lpg=PT35&dq=%22beneath%20the%20words%22%20empathy%20listening&pg=PT35#v=onepage&q=%22beneath%20the%20words%22%20empathy%20listening&f=false]

Convolutions are the filters that bring out meaningful patterns from words.
And the best part is, you don't have no longer have to hard-code these patterns into the convolutional kernel.
The training process will search for the best possible pattern-matching convolutions for your particular problem.
Each time you propagate the error from your labeled dataset back through the network (backpropagation), the optimizer will adjust the weights in each of your filters so that they get better and better at detecting meaning and classifying your text examples.

=== Robustifying your CNN with dropout
// SUM: Dropout is critical to prevent overfitting for neural networks because of they have so many degrees of freedom (learnable parameters).

Most neural networks are susceptible to adversarial examples that trick them into outputting incorrect classifications or text.
And sometimes neural networks are susceptible to changes as straight forward as synonym substitution, misspellings, or insertion of slang.
Sometimes all it takes is a little "word salad" -- nonsensical random words -- to distract and confuse an NLP algorithm.
Humans know how to ignore noise and filter out distractors, but machines sometimes have trouble with this.

_Robust NLP_ is the study of approaches and techniques for building machines that are smart enough to handle unusual text from diverse sources.footnote:[Robin Jia's thesis on Robust NLP (https://robinjia.github.io/assets/pdf/robinjia_thesis.pdf) and his presentation with Kai-Wei Chang, He He and Sameer Singh (https://robustnlp-tutorial.github.io)]
In fact, research into robust NLP may uncover paths toward artificial general intelligence.
Humans are able to learn new words and concepts from just a few examples.
And we generalize well, not too much and not too little.
Machines need a little help.
And if you can figure out the "secret sauce" that makes us humans good at this, then you can encode it into your NLP pipelines.

One popular technique for increasing the robustness of neural networks  is _random dropout_.
_Random dropout_, or just _dropout_, has become popular because of its ease and effectiveness.
Your neural networks will almost always benefit from a dropout layer.
A dropout layer randomly hides some of the neurons outputs from the neurons listening to them.
This causes that pathway in your artificial brain to go quiet and forces the other neurons to learn from the particular examples that are in front of it during that dropout.

It's counter-intuitive, but dropout helps your neural network to spread the learning around.
Without a dropout layer, your network will focus on the words and patterns and convolutional filters that helped it achieve the greatest accuracy boost.
But you need your neurons to diversify their patterns so that your network can be "robust" to common variations on natural language text.

The best place in your neural network to install a dropout layer is close to the end, just before you run the fully connected linear layer that computes the predictions on a batch of data.
This vector of weights passing into your linear layer are the outputs from your CNN and pooling layers.
Each one of these values represents a sequence of words, or patterns of meaning and syntax.
By hiding some of these patterns from your prediction layer, it forces your prediction layer to diversify its "thinking."
Though your software isn't really thinking about anything, it's OK to anthropomorphize it a bit, if it helps you develop intuitions about why techniques like random dropout can improve your model's accuracy.

== PyTorch CNN to process disaster toots
// SUM: With CNNs you can separate personal rants on Twitter from newsworthy factual content and the only new CNN layers to your pipeline are convolution and pooling.

Now comes the fun part.
You are going to build a real world CNN that can distinguish real world news from sensationalism.
Your model can help you filter out Tweets abiout the culture wars so you can focus on news from real war zones.

First you will see where your new convolution layers fit into the pipeline.
Then you'll assemble all the pieces to train a CNN on a dataset of "disaster tweets."
And if doom scrolling and disaster is not your thing, the CNN is easily adaptable to any labeled dataset of tweets.
You can even pick a hashtag that you like and use that as you target label.


# #### .CNN hyperparameters

# In[ ]:


class CNNTextClassifier(nn.Module):

    def __init__(self, embeddings):
        super().__init__()

        self.seq_len = 40  # <1>
        self.vocab_size = 10000  # <2>
        self.embedding_size = 50  # <3>
        self.out_channels = 5  # <4>
        self.kernel_lengths = [2, 3, 4, 5, 6]  # <5>
        self.stride = 1  # <6>
        self.dropout = nn.Dropout(0)  # <7>
        self.pool_stride = self.stride  # <8>
        self.conv_out_seq_len = calc_out_seq_len(  # <9>
            seq_len=self.seq_len,
            kernel_lengths=self.kernel_lengths,
            stride=self.stride,
            )
----
<1> `N_`: assume a maximum text length of 40 tokens
<2> `V`: number of unique tokens (words) in your vocabulary
<3> `E`: number of word embedding dimensions (kernel input channels)
<4> `F`: number of filters (kernel output channels)
<5> `K`: number of columns of weights in each kernel
<6> `S`: number of time steps (tokens) to slide the kernel forward with each step
<7> `D`: portion of convolution output to ignore. 0 dropout increases overfitting
<8> `P`: pooling strides greater than 1 will increase feature reduction
<9> `C`: total convolutional output size based on kernel and pooling hyperparameters

Just as for your hand-crafted convolutions earlier in this chapter, the sequence length is reduced by each convolutional operation.
And the amount of shortening depends on the size of the kernel and the stride.
The PyTorch documentation for a `Conv1d` layer provides this formula and a detailed explanation of the terms.footnote:[(https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)]

[source,python]
----
def calc_conv_out_seq_len(seq_len, kernel_len,
                          stride=1, dilation=1, padding=0):
    """
    L_out =     (L_in + 2 * padding - dilation * (kernel_size - 1) - 1)
            1 + _______________________________________________________
                                        stride
    """
    return (
        1 + (seq_len +
             2 * padding - dilation * (kernel_len - 1) - 1
            ) //
        stride
        )
----

Your first CNN layer is an `nn.Embedding` layer that converts a sequence of word id integers into a sequence of embedding vectors.
It has as many rows as you have unique tokens in your vocabulary (including the new padding token).
And it has a column for each dimension of the embedding vectors.
You can load these embedding vectors from GloVe or any other pretrained embeddings.


.Initialize CNN embedding
[source,python]
----
self.embed = nn.Embedding(
    self.vocab_size,  # <1>
    self.embedding_size,  # <2>
    padding_idx=0)
state = self.embed.state_dict()


# #### .Construct convolution and pooling layers

# In[ ]:


self.convolvers = []
self.poolers = []
total_out_len = 0
for i, kernel_len in enumerate(self.kernel_lengths):
    self.convolvers.append(
        nn.Conv1d(in_channels=self.embedding_size,
                  out_channels=self.out_channels,
                  kernel_size=kernel_len,
                  stride=self.stride))
    print(f'conv[{i}].weight.shape: {self.convolvers[-1].weight.shape}')
    conv_output_len = calc_conv_out_seq_len(
        seq_len=self.seq_len, kernel_len=kernel_len, stride=self.stride)
    print(f'conv_output_len: {conv_output_len}')
    self.poolers.append(
        nn.MaxPool1d(kernel_size=conv_output_len, stride=self.stride))
    total_out_len += calc_conv_out_seq_len(
        seq_len=conv_output_len, kernel_len=conv_output_len,
        stride=self.stride)
    print(f'total_out_len: {total_out_len}')
    print(f'poolers[{i}]: {self.poolers[-1]}')
print(f'total_out_len: {total_out_len}')
self.linear_layer = nn.Linear(self.out_channels * total_out_len, 1)
print(f'linear_layer: {self.linear_layer}')
----

Unlike the previous examples, you're going to now create multiple convolution and pooling layers.
For this example we won't layer them up as is often done in computer vision.
Instead you will concatenate the convolution and pooling outputs together.
This is effective because you've limited the dimensionality of your convolution and pooling output by performing global max pooling and keeping the number of output channels much smaller than the number of embedding dimensions.



You can use print statements to help debug mismatching matrix shapes for each layer of your CNN.
And you want to make sure you don't unintentionally create too many trainable parameters that cause more overfitting than you'd like:
Your pooling outputs each contain a sequence length of 1, but they also contain 5 channels for the embedding dimensions combined together during convolution.
So the concatenated and pooled convolution outout is a 5x5 tensor which produces a 25-D linear layer for the output tensor that encodes the meaning of each text.

.CNN layer shapes
[source,python]
----
conv[0].weight.shape: torch.Size([5, 50, 2])
conv_output_len: 39
total_pool_out_len: 1
poolers[0]: MaxPool1d(kernel_size=39, stride=1, padding=0, dilation=1,
    ceil_mode=False)
conv[1].weight.shape: torch.Size([5, 50, 3])
conv_output_len: 38
total_pool_out_len: 2
poolers[1]: MaxPool1d(kernel_size=38, stride=1, padding=0, dilation=1,
    ceil_mode=False)
conv[2].weight.shape: torch.Size([5, 50, 4])
conv_output_len: 37
total_pool_out_len: 3
poolers[2]: MaxPool1d(kernel_size=37, stride=1, padding=0, dilation=1,
    ceil_mode=False)
conv[3].weight.shape: torch.Size([5, 50, 5])
conv_output_len: 36
total_pool_out_len: 4
poolers[3]: MaxPool1d(kernel_size=36, stride=1, padding=0, dilation=1,
    ceil_mode=False)
conv[4].weight.shape: torch.Size([5, 50, 6])
conv_output_len: 35
total_pool_out_len: 5
poolers[4]: MaxPool1d(kernel_size=35, stride=1, padding=0, dilation=1,


# #### 

# In[ ]:


Epoch:  1, loss: 0.76782, Train accuracy: 0.59028, Test accuracy: 0.64961
Epoch:  2, loss: 0.64052, Train accuracy: 0.65947, Test accuracy: 0.67060
Epoch:  3, loss: 0.51934, Train accuracy: 0.68632, Test accuracy: 0.68766
...
Epoch: 55, loss: 0.04995, Train accuracy: 0.80558, Test accuracy: 0.72966
Epoch: 65, loss: 0.05682, Train accuracy: 0.80835, Test accuracy: 0.72178
Epoch: 75, loss: 0.04491, Train accuracy: 0.81287, Test accuracy: 0.71522
----

By reducing the number of channels from 5 to 3 for each embedding you can reduce the total output dimensionality from 25 to 15.
This will limit the overfitting but reduce the convergence rate unless you increase the learning coefficient:


[source,python]
----
Epoch:  1, loss: 0.61644, Train accuracy: 0.57773, Test accuracy: 0.58005
Epoch:  2, loss: 0.52941, Train accuracy: 0.63232, Test accuracy: 0.64567
Epoch:  3, loss: 0.45162, Train accuracy: 0.67202, Test accuracy: 0.65486
...
Epoch: 55, loss: 0.21011, Train accuracy: 0.79200, Test accuracy: 0.69816
Epoch: 65, loss: 0.21707, Train accuracy: 0.79434, Test accuracy: 0.69423
Epoch: 75, loss: 0.20077, Train accuracy: 0.79784, Test accuracy: 0.70079
----

=== Pooling

Pooling aggregates the data from a large tensor to compress the information into fewer values.
This is often called a "reduce" operation in the world of "Big Data" where the map-reduce software pattern is common.
Convolution and pooling lend themselves well to the map-reduce software pattern and can be parallelized within a GPU automatically using PyTorch.
You can even use multi-server HPC (high performance computing) systems to speed up your training.
But CNNs are so efficient, you aren't likely to need this kind of horsepower.

All the statistics you're used to calculating on a matrix of data can be useful as pooling functions for CNNs:

* `min`
* `max`
* `std`
* `sum`
* `mean`

The most common and most successful aggregations


// * For each input example you applied a filter (weights and activation function).
// * Convolved across the length of the input, which would output a 1D vector slightly smaller than the original input (1x398 which is input with the filter starting left-aligned and finishing right-aligned) for each filter
// * For each filter output (there are 250 of them, remember), you took the single maximum value from that 1D vector.
// * At this point you have a single vector (per input example) that is 1x250 (the number of filters).

=== Linear layer

The concatenated encodings approach gave you a lot of information about each microblog post.
The encoding vector had 1856 values.
The largest word vectors you worked with in chapter 6 were 300 dimensions.
And all you really want for this particular pipeline is the binary answer to the question "is it news worthy or not?"

get_ipython().run_line_magic('pinfo', 'words')
Even though you didn't really pay attention to the answer to all those thousands of questions (one for each word in your vocabulary), it was the same problem you have now.
So you can use the same approach, a `torch.nn.Linear` layer will optimally combine all the pieces of information together from a high dimensional vector to answer whatever question you pose it.

So you need to add a Linear layer with as many weights as you have encoding dimensions that are being output from your pooling layers.

Listing 7.26 shows the code you can use to calculate the size of the linear layer.

// Listing 7.26


# #### .Compute the tensor size for the output of a 1D convolution

# In[ ]:


out_pool_total = 0
for kernel_len, stride in zip(kernel_lengths, strides):
    out_conv = (
        (in_seq_len - dilation * (kernel_len - 1) - 1) // stride) + 1
    out_pool = (
        (out_conv - dilation * (kernel_len - 1) - 1) // stride) + 1
    out_pool_total += out_pool
----

=== Getting fit
// SUM: A convolutional neural network can train quickly on even a modest laptop so you can experiment with various hyperparameter combinations and often achieve better performance faster than with more complicated models.

Before you can train your CNN you need to tell it how to adjust the weights (parameters) with each batch of training data.
You need to compute two pieces, the slopes of the weights relative to the loss function (the gradient) and an estimate of how far to try to descend that slope (the learning rate).
For the single-layer perceptrons and even the logistic regressions of the previous chapters you were able to get away with using some general purpose optimizers like "Adam."
And you can often set the learning rate to a fixed value for CNNs
And those will work well for CNNs too.
However, if you want to speed up your training you can try to find an optimizer that's a bit more clever about how it adjusts all those parameters of your model.
Geoffrey Hinton called this approach "rmsprop" because he uses the root mean square (RMS) formula to compute the moving average of the recent gradients.
RMSprop aggregates an exponentially decaying window of the weights for each batch of data to improve the estimate of the parameter gradient (slopes) and speed up learning.footnote:[Slide 14 "Four ways to speed up machine learning" from "Overview of mini‐batch gradient descent" by Hinton (https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)] footnote:[Ph D thesis "Optimizing Neural Networks that Generate Images" by Tijmen Tieleman (https://www.cs.toronto.edu/~tijmen/tijmen_thesis.pdf)]
It is usually a good bet for backpropagation within a convolutional neural network for NLP.


=== Hyperparameter Tuning

Explore the hyperparameter space to see if you can beat my performance.
Fernando Lopez and others have achieved 80% validation and test set accuracy on this dataset using 1-D convolution.
There's likely a lot of room to grow.

The nlpia2 package contains a command line script that accepts arguments for many of the hyperparameters you might want to adjust.
Give it a try and see if you can find a more fertile part of the hyperspace universe of possibilities.
You can see my latest attempt in listing 7.27

// Listing 7.27
.Command line script for optimizing hyperparameters
[source,bash]
----
python train.py --dropout_portion=.35 --epochs=16 --batch_size=8 --win=True
----

[source,text]
----
Epoch:  1, loss: 0.44480, Train accuracy: 0.58152, Test accuracy: 0.64829
Epoch:  2, loss: 0.27265, Train accuracy: 0.63640, Test accuracy: 0.69029
...
Epoch: 15, loss: 0.03373, Train accuracy: 0.83871, Test accuracy: 0.79396
Epoch: 16, loss: 0.09545, Train accuracy: 0.84718, Test accuracy: 0.79134
----

Did you notice the `win=True` flag in listing 7.27?
That is an Easter Egg or cheat code I created for myself within my CNN pipeline.
Whenever I discover a winning ticket in the "Lottery Ticket Hypothesis" game, I hard code it into my pipeline.
In order for this to work, you have to keep track of the random seeds you use and the exact dataset and software you are using.
If you can recreate all of these pieces, it's usually possible to recreate a particularly lucky "draw" to build on and improve later as you think of new architecture or parameter tweaks.

In fact, this winning random number sequence initialized the weights of the model so well that the test accuracy started off better than the training set accuracy.
It took 8 epochs for the training accuracy to overtake the test set accuracy.
After 16 passes through the dataset (epochs), the model is fit 5% better to the training set than the test set.

If you want to achieve higher test set accuracy and reduce the overfitting, you can try adding some regularization or increasing the amount of data ignored within the Dropout layer.
For most neural networks, dropout ratios of 30% to 50% often work well to prevent overfitting without delaying the learning too long.
A single-layer CNN doesn't benefit much from dropout ratios above 20%.

.CNN hyperparameter tuning

