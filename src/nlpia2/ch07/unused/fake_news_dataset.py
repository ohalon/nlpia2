import pandas as pd
import spacy
import re
import numpy as np
from torchtext.vocab import vocab
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split

nlp = spacy.load("en_core_web_sm")

## Load Data
DATA_DIR = ('https://gitlab.com/prosocialai/nlpia2/-/raw/main/.nlpia2-data')

true = pd.read_csv(DATA_DIR+'/fake-news-dataset-true.csv')
fake = pd.read_csv(DATA_DIR+'/fake-news-dataset-fake.csv')

true['label'] = 1
fake['label'] = 0

fake.drop(labels=['subject','date', 'text'],axis=1,inplace=True)
true.drop(labels=['subject','date', 'text'],axis=1,inplace=True)

df = pd.concat([fake,true])

x_raw = df['title'].values
y = df['label'].values

## Remove punctuation
x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in x_raw]

##Tokenize
def spacy_tokenizer(sentence):
    return [token.text for token in nlp(sentence.lower())]

x_raw = [spacy_tokenizer(sentence) for sentence in x_raw]

##Construct vocabulary
counter = Counter()
for sentence in x_raw:
    counter.update(sentence)

sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[1], reverse=True)
ordered_dict = OrderedDict(sorted_by_freq_tuples)
vocabulary = vocab(ordered_dict)
unk_token = '<unk>'
pad_token = '<pad>'
vocabulary.insert_token(pad_token,0)
vocabulary.insert_token(unk_token,1)

## Transforming sentences into index based representation
x_tokenized = list()

max_seq_len = 0
for sentence in x_raw:
    temp_sentence = list()
    for word in sentence:
        temp_sentence.append(vocabulary[word])
    x_tokenized.append(temp_sentence)
    if len(temp_sentence) > max_seq_len:
        max_seq_len = len(temp_sentence)

## Padding
x_padded = list()
pad_idx = 0

for sentence in x_tokenized:
    while len(sentence) < max_seq_len:
        sentence.insert(len(sentence), pad_idx)
    x_padded.append(sentence)

x_padded = np.array(x_padded)

## Train and Test Split
x_train, x_test, y_train, y_test = train_test_split(x_padded, y, test_size=0.25, random_state=42)

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.cuda
from argparse import Namespace
import random

from torch.utils.data import Dataset, DataLoader

## Creating a DataLoader
args = Namespace(
    embed_dim = 64,
    learning_rate = 0.25,
    batch_size = 50,
    num_epochs = 10
)

class TitleDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# Initialize loaders
loader_train = DataLoader(TitleDataset(x_train, y_train), batch_size=args.batch_size)
loader_test = DataLoader(TitleDataset(x_test, y_test), batch_size=args.batch_size)

class NewsTitleClassifier(nn.Module):
    def __init__(self,
                 vocab_size=None,
                 embed_dim=300,
                 filter_sizes=[3, 4, 5],
                 num_filters=[100, 100, 100],
                 num_classes=2,
                 dropout=0.5):
        """
        The constructor for CNN_NLP class.

        Args:
            vocab_size (int): Need to be specified when not pretrained word
                embeddings are not used.
            embed_dim (int): Dimension of word vectors. Need to be specified
                when pretrained word embeddings are not used. Default: 300
            filter_sizes (List[int]): List of filter sizes. Default: [3, 4, 5]
            num_filters (List[int]): List of number of filters, has the same
                length as `filter_sizes`. Default: [100, 100, 100]
            n_classes (int): Number of classes. Default: 2
            dropout (float): Dropout rate. Default: 0.5
        """

        super(NewsTitleClassifier, self).__init__()
        # Embedding layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                          embedding_dim=embed_dim,
                                          padding_idx=0)
        # Conv Network
        self.conv1d_list = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters[i],
                      kernel_size=filter_sizes[i])
            for i in range(len(filter_sizes))
        ])
        # Fully-connected layer and Dropout
        self.fc = nn.Linear(np.sum(num_filters), num_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x_title, apply_softmax=False):
        """The forward pass of the classifier
        Args:
        x_title (torch.Tensor): an input data tensor
        x_surname.shape should be (batch, initial_num_channels,
        max_title_length)
        apply_softmax (bool): a flag for the softmax activation
        should be false if used with the cross­entropy losses
        Returns:
        the resulting tensor. tensor.shape should be (batch, num_classes).
        """

        features = self.convnet(x_title).squeeze(dim=2)
        prediction_vector = self.fc(features)
        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)
        return prediction_vector

    def forward(self, input_ids):
        """Perform a forward pass through the network.

        Args:
            input_ids (torch.Tensor): A tensor of token ids with shape
                (batch_size, max_sent_length)

        Returns:
            logits (torch.Tensor): Output logits with shape (batch_size,
                n_classes)
        """

        # Get embeddings from `input_ids`. Output shape: (b, max_len, embed_dim)
        x_embed = self.embedding(input_ids).float()

        # Permute `x_embed` to match input shape requirement of `nn.Conv1d`.
        # Output shape: (b, embed_dim, max_len)
        x_reshaped = x_embed.permute(0, 2, 1)

        # Apply CNN and ReLU. Output shape: (b, num_filters[i], L_out)
        x_conv_list = [F.relu(conv1d(x_reshaped)) for conv1d in self.conv1d_list]

        # Max pooling. Output shape: (b, num_filters[i], 1)
        x_pool_list = [F.max_pool1d(x_conv, kernel_size=x_conv.shape[2])
                       for x_conv in x_conv_list]

        # Concatenate x_pool_list to feed the fully connected layer.
        # Output shape: (b, sum(num_filters))
        x_fc = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list],
                         dim=1)

        # Compute logits. Output shape: (b, n_classes)
        logits = self.fc(self.dropout(x_fc))

        return logits





if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")

classifier = NewsTitleClassifier(vocab_size=len(vocabulary),
                                 embed_dim=args.embed_dim)

classifier = classifier.to(args.device)

loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adadelta(classifier.parameters(),
                           lr=args.learning_rate,
                           rho=0.95)

def set_seed(seed_value=42):
    """Set seed for reproducibility."""

    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

import time

def train(classifier, optimizer, loader_train, loader_test=None, epochs=10):
    """Train the CNN model."""

    # Tracking best validation accuracy
    best_accuracy = 0

    # Start training loop
    print("Start training...\n")
    print(f"{'Epoch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | { 'Val Acc': ^ 9} | {'Elapsed': ^ 9}")
    print("-" * 60)

    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================

        # Tracking time and loss
        t0_epoch = time.time()
        total_loss = 0

        # Put the model into the training mode
        classifier.train()

        for step, batch in enumerate(loader_train):
            # Load batch to GPU
            b_input_ids, b_labels = tuple(t.to(args.device) for t in batch)

            # Zero out any previously calculated gradients
            classifier.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = classifier(b_input_ids)

            # Compute loss and accumulate the loss values
            loss = loss_func(logits, b_labels)
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Update parameters
            optimizer.step()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(loader_train)

        # =======================================
        #               Evaluation
        # =======================================
        if loader_test is not None:
            # After the completion of each training epoch, measure the model's
            # performance on our validation set.
            val_loss, val_accuracy = evaluate(classifier, loader_test)

            # Track the best accuracy
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            print(f"{epoch_i + 1:^7} | {avg_train_loss:^12.6f} | { val_loss: ^ 10.6f} | {val_accuracy: ^ 9.2f} | {time_elapsed: ^ 9.2f}")

    print("\n")
    print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")


def evaluate(model, test_dataloader):
    """After the completion of each training epoch, measure the model's
    performance on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled
    # during the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(args.device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids)

        # Compute loss
        loss = loss_func(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


num_epochs = 10

best_accuracy = 0
# Start training loop
print("Start training...\n")
print(f"{'Epoch':^7}|{'Train Loss':^12}|{'Val Loss':^10}|{'Val Acc':^9}|{'Elapsed':^9}")
print("-" * 60)

for epoch_i in range(num_epochs):
    # =======================================
    #               Training
    # =======================================

    # Tracking time and loss
    t0_epoch = time.time()
    total_loss = 0

    # Put the model into the training mode
    classifier.train()

    for step, batch in enumerate(loader_train):
        # Load batch to GPU
        b_input_ids, b_labels = tuple(t.to(args.device) for t in batch)

        # Zero out any previously calculated gradients
        classifier.zero_grad()

        # Perform a forward pass. This will return logits.
        logits = classifier(b_input_ids)

        # Compute loss and accumulate the loss values
        loss = loss_func(logits, b_labels)
        total_loss += loss.item()

        # Perform a backward pass to calculate gradients
        loss.backward()

        # Update parameters
        optimizer.step()

    # Calculate the average loss over the entire training data
    avg_train_loss = total_loss / len(loader_train)

    # =======================================
    #               Evaluation
    # =======================================
    if loader_test is not None:
        # After the completion of each training epoch, measure the model's
        # performance on our validation set.
        val_loss, val_accuracy = evaluate(classifier, loader_test)

        # Track the best accuracy
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy

        # Print performance over the entire training data
        time_elapsed = time.time() - t0_epoch
        print(f"{epoch_i + 1:^7}|{avg_train_loss:^12.6f}|{val_loss:^10.6f}|{val_accuracy:^9.2f}|{time_elapsed:^9.2f}")

print("\n")
print(f"Training complete! Best accuracy: {best_accuracy:.2f}%.")

