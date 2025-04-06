>>> import spacy

>>> spacy_de = spacy.load('de')
>>> spacy_en = spacy.load('en')

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
from pytorch import nn
from torch import nn
>>> class PositionalEncoding(nn.Module):
...     def __init__(self, d_model, dropout=0.1, max_len=5000):
...         super(PositionalEncoding, self).__init__()
...         self.dropout = nn.Dropout(p=dropout)

...         pe = torch.zeros(max_len, d_model)
...         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
...         div_term = torch.exp(torch.arange(0, d_model, 2).float() *
...                              (-math.log(10000.0) / d_model))
...         pe[:, 0::2] = torch.sin(position * div_term)
...         pe[:, 1::2] = torch.cos(position * div_term)
...         pe = pe.unsqueeze(0).transpose(0, 1)
...         self.register_buffer('pe', pe)

...     def forward(self, x):
...         x = x + self.pe[:x.size(0), :]
...         return self.dropout(x)
from spacy import cli
cli.download("de")
>>> import spacy

>>> spacy_de = spacy.load('de')
>>> spacy_en = spacy.load('en')

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
spacy.load("de")
nlp_de = spacy.load('de_core_news_sm')
nlp_de("Sprechen Sie Deutche")
doc = nlp_de("Sprechen Sie Deutche")
for tok in doc:
    print(tok.text)
for tok in doc:
    print(tok.pos_, tok.text)
for tok in doc:
    print(tok.vector[:3], tok.pos_, tok.text)
>>> import spacy

>>> spacy_de = spacy.load('de_core_news_sm')
>>> spacy_en = spacy.load('en')

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
>>> import spacy

>>> spacy_de = spacy.load('de_core_news_sm')
>>> spacy_en = spacy.load("en_core_web_sm")

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
>>> import spacy

>>> spacy_de = spacy.load('de_core_news_sm')
>>> spacy_en = spacy.load("en_core_web_md")

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
cli.download("en_core_web_sm")
>>> import spacy

>>> spacy_de = spacy.load('de_core_news_sm')
>>> spacy_en = spacy.load("en_core_web_md")

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
>>> import spacy

>>> spacy_de = spacy.load('de_core_news_sm')
>>> spacy_en = spacy.load("en_core_web_sm")

>>> def tokenize_de(text):
...     return [tok.text for tok in spacy_de.tokenizer(text)]

>>> def tokenize_en(text):
...     return [tok.text for tok in spacy_en.tokenizer(text)]
>>> import torchtext
>>> from torchtext.datasets import Multi30k
>>> from torchtext.data import Field, BucketIterator

>>> SRC = Field(tokenize = tokenize_de,
...             init_token = '<sos>',  
...             eos_token = '<eos>',
...             lower = True,
...             batch_first = True)

>>> TRG = Field(tokenize = tokenize_en,
...             init_token = '<sos>',
...             eos_token = '<eos>',
...             lower = True,
...             batch_first = True)
import torchtext.data
dir(torchtext.data)
dir(torchtext.data.functional)
import torchtext.data.BucketIterator
from torchtext.data import *
who
from torchtext.data import field
import torchtext
torchtext.__version__
from torchtext.datasets import field
from torchtext.datasets import Field
from torchtext.datasets import IWSLT2017
from torchtext.datasets import Multi30k
>> from torch import Tensor
>>> from typing import Optional, Any

>>> class CustomDecoderLayer(nn.TransformerDecoderLayer):
...     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
...                 memory_mask: Optional[Tensor] = None,
...                 tgt_key_padding_mask: Optional[Tensor] = None,
...                 mem_key_padding_mask: Optional[Tensor] = None) -> Tensor:
...         """Same as DecoderLayer but returns multi-head attention weights.
...         """
...         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
...                               key_padding_mask=tgt_key_padding_mask)[0]
...         tgt = tgt + self.dropout1(tgt2)
...         tgt = self.norm1(tgt)
...         tgt2, attention_weights = self.multihead_attn(tgt, memory, memory,  
...                                                       attn_mask=memory_mask,
...                                                       key_padding_mask=mem_key_padding_mask,
...                                                       need_weights=True)
...         tgt = tgt + self.dropout2(tgt2)
...         tgt = self.norm2(tgt)
...         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
...         tgt = tgt + self.dropout3(tgt2)
...         tgt = self.norm3(tgt)
...         return tgt, attention_weights
>>> from torch import Tensor
>>> from typing import Optional, Any

>>> class CustomDecoderLayer(nn.TransformerDecoderLayer):
...     def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
...                 memory_mask: Optional[Tensor] = None,
...                 tgt_key_padding_mask: Optional[Tensor] = None,
...                 mem_key_padding_mask: Optional[Tensor] = None) -> Tensor:
...         """Same as DecoderLayer but returns multi-head attention weights.
...         """
...         tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
...                               key_padding_mask=tgt_key_padding_mask)[0]
...         tgt = tgt + self.dropout1(tgt2)
...         tgt = self.norm1(tgt)
...         tgt2, attention_weights = self.multihead_attn(tgt, memory, memory,  
...                                                       attn_mask=memory_mask,
...                                                       key_padding_mask=mem_key_padding_mask,
...                                                       need_weights=True)
...         tgt = tgt + self.dropout2(tgt2)
...         tgt = self.norm2(tgt)
...         tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
...         tgt = tgt + self.dropout3(tgt2)
...         tgt = self.norm3(tgt)
...         return tgt, attention_weights
decoder = CustomDecoderLayer()
decoder = CustomDecoderLayer(nhead=3)
decoder = CustomDecoderLayer?
decoder = CustomDecoderLayer(50_000, 3)
decoder = CustomDecoderLayer(300, 100)
import torch
from torch import Tensor
import nump as np
import numpy as np
np.random.randn()
np.random.randn((300, 100))
np.random.randn(300, 100)
Tensor(np.random.randn(300, 100))
inp = Tensor(np.random.randn(300, 100))
decoder.forward(inp)
history
%hist -o -p -f ch12.hist.ipy
%hist -f ch12.hist.py
