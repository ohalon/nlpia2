# torch and PositionalEncoding
import math
import torch
from torch import nn
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)  # <1>
        self.d_model = d_model  # <2>
        self.max_len = max_len  # <3>
        pe = torch.zeros(max_len, d_model)  # <4>
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # <5>
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]  # <6>
        return self.dropout(x)

# dataset and device (GPU)
from datasets import load_dataset  # <1>
opus = load_dataset('opus_books', 'de-en')
opus
sents = opus['train'].train_test_split(test_size=.1)
sents
next(iter(sents['test']))  # <1>
DEVICE = torch.device(
    'cuda' if torch.cuda.is_available()
    else 'cpu')
SRC = 'en'  # <1>
TGT = 'de'  # <2>
SOS, EOS = '<s>', '</s>'
PAD, UNK, MASK = '<pad>', '<unk>', '<mask>'
SPECIAL_TOKS = [SOS, PAD, EOS, UNK, MASK]
VOCAB_SIZE = 10_000

# tokenizer
from tokenizers import ByteLevelBPETokenizer  # <3>
tokenize_src = ByteLevelBPETokenizer()
tokenize_src.train_from_iterator(
    [x[SRC] for x in sents['train']['translation']],
    vocab_size=10000, min_frequency=2,
    special_tokens=SPECIAL_TOKS)
PAD_IDX = tokenize_src.token_to_id(PAD)
tokenize_tgt = ByteLevelBPETokenizer()
tokenize_tgt.train_from_iterator(
    [x[TGT] for x in sents['train']['translation']],
    vocab_size=10000, min_frequency=2,
    special_tokens=SPECIAL_TOKS)
assert PAD_IDX == tokenize_tgt.token_to_id(PAD)

# decoder LAYER
from torch import Tensor
from typing import Optional
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None,
            memory_key_padding_mask: Optional[Tensor] = None
            ) -> Tensor:
        """Like decode but returns multi-head attention weights."""
        tgt2 = self.self_attn(
            tgt, tgt, tgt, attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attention_weights = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,  # <1>
            key_padding_mask=memory_key_padding_mask,
            need_weights=True,
            attn_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(
            self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_weights  # <2>

# instantiate DecoderLayer()
decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
decoder_layer

# class nn.TransformerDecoder
class CustomDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
        super().__init__(
            decoder_layer, num_layers, norm)

    def forward(self,
            tgt: Tensor, memory: Tensor,
            tgt_mask: Optional[Tensor] = None,
            memory_mask: Optional[Tensor] = None,
            tgt_key_padding_mask: Optional[Tensor] = None
            ) -> Tensor:
        """Like TransformerDecoder but cache multi-head attention"""
        self.attention_weights = []  # <1>
        output = tgt
        for mod in self.layers:
            output, attention = mod(
                output, memory, tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)
            self.attention_weights.append(attention) # <2>

        if self.norm is not None:
            output = self.norm(output)

        return output


# instantiate Decoder()
decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=5)
decoder

# einops to reshape tensors
from einops import rearrange  # <1>

# class Transformer
class TranslationTransformer(nn.Transformer):  # <2>
    def __init__(self,
            d_model: int = 512,
            nhead: int = 8,
            dim_feedforward: int = 2048,
            dropout: float = 0.1,
            activation: str = "relu",
            device=DEVICE,
            src_vocab_size: int = VOCAB_SIZE,
            src_pad_idx: int = PAD_IDX,
            tgt_vocab_size: int = VOCAB_SIZE,
            tgt_pad_idx: int = PAD_IDX,
            max_sequence_length: int = 100,
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6,
        ):

        decoder_layer = CustomDecoderLayer(
            d_model, nhead, dim_feedforward,  # <3>
            dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        decoder = CustomDecoder(
            decoder_layer, num_decoder_layers,
            decoder_norm)  # <4>

        super().__init__(
            d_model=d_model, nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout, custom_decoder=decoder)

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.device = device

        self.src_emb = nn.Embedding(
            src_vocab_size, d_model)  # <5>
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)

        self.pos_enc = PositionalEncoding(
            d_model, dropout, max_sequence_length)  # <6>
        self.linear = nn.Linear(
            d_model, tgt_vocab_size)  # <7>
    def _make_key_padding_mask(self, t, pad_idx):
        mask = (t == pad_idx).to(self.device)
        return mask

    def prepare_src(self, src, src_pad_idx):
        src_key_padding_mask = self._make_key_padding_mask(
            src, src_pad_idx)
        src = rearrange(src, 'N S -> S N')
        src = self.pos_enc(self.src_emb(src)
            * math.sqrt(self.d_model))
        return src, src_key_padding_mask
    def prepare_tgt(self, tgt, tgt_pad_idx):
        tgt_key_padding_mask = self._make_key_padding_mask(
            tgt, tgt_pad_idx)
        tgt = rearrange(tgt, 'N T -> T N')
        tgt_mask = self.generate_square_subsequent_mask(
            tgt.shape[0]).to(self.device)
        tgt = self.pos_enc(self.tgt_emb(tgt)
            * math.sqrt(self.d_model))
        return tgt, tgt_key_padding_mask, tgt_mask
    def forward(self, src, tgt):
        src, src_key_padding_mask = self.prepare_src(
            src, self.src_pad_idx)
        tgt, tgt_key_padding_mask, tgt_mask = self.prepare_tgt(
            tgt, self.tgt_pad_idx)
        memory_key_padding_mask = src_key_padding_mask.clone()
        output = super().forward(
            src, tgt, tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask)
        output = rearrange(output, 'T N E -> N T E')
        return self.linear(output)
    def init_weights(self):
        def _init_weights(m):
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight.data)
        self.apply(_init_weights);  # <1>


# # DUPLICATE!! class TranslationTransformer
# class TranslationTransformer(nn.Transformer):
#     def __init__(self,
#             device=DEVICE,
#             src_vocab_size: int = 10000,
#             src_pad_idx: int = PAD_IDX,
#             tgt_vocab_size: int  = 10000,
#             tgt_pad_idx: int = PAD_IDX,
#             max_sequence_length: int = 100,
#             d_model: int = 512,
#             nhead: int = 8,
#             num_encoder_layers: int = 6,
#             num_decoder_layers: int = 6,
#             dim_feedforward: int = 2048,
#             dropout: float = 0.1,
#             activation: str = "relu"
#             ):
#         decoder_layer = CustomDecoderLayer(
#             d_model, nhead, dim_feedforward,
#             dropout, activation)
#         decoder_norm = nn.LayerNorm(d_model)
#         decoder = CustomDecoder(
#             decoder_layer, num_decoder_layers, decoder_norm)

#         super().__init__(
#             d_model=d_model, nhead=nhead,
#             num_encoder_layers=num_encoder_layers,
#             num_decoder_layers=num_decoder_layers,
#             dim_feedforward=dim_feedforward,
#             dropout=dropout, custom_decoder=decoder)

#         self.src_pad_idx = src_pad_idx
#         self.tgt_pad_idx = tgt_pad_idx
#         self.device = device
#         self.src_emb = nn.Embedding(src_vocab_size, d_model)
#         self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
#         self.pos_enc = PositionalEncoding(
#             d_model, dropout, max_sequence_length)
#         self.linear = nn.Linear(d_model, tgt_vocab_size)

#     def init_weights(self):
#         def _init_weights(m):
#             if hasattr(m, 'weight') and m.weight.dim() > 1:
#                 nn.init.xavier_uniform_(m.weight.data)
#         self.apply(_init_weights);

#     def _make_key_padding_mask(self, t, pad_idx=PAD_IDX):
#         mask = (t == pad_idx).to(self.device)
#         return mask

#     def prepare_src(self, src, src_pad_idx):
#         src_key_padding_mask = self._make_key_padding_mask(
#             src, src_pad_idx)
#         src = rearrange(src, 'N S -> S N')
#         src = self.pos_enc(self.src_emb(src)
#             * math.sqrt(self.d_model))
#         return src, src_key_padding_mask

#     def prepare_tgt(self, tgt, tgt_pad_idx):
#         tgt_key_padding_mask = self._make_key_padding_mask(
#             tgt, tgt_pad_idx)
#         tgt = rearrange(tgt, 'N T -> T N')
#         tgt_mask = self.generate_square_subsequent_mask(
#             tgt.shape[0]).to(self.device)      # <1>
#         tgt = self.pos_enc(self.tgt_emb(tgt)
#             * math.sqrt(self.d_model))
#         return tgt, tgt_key_padding_mask, tgt_mask

#     def forward(self, src, tgt):
#         src, src_key_padding_mask = self.prepare_src(
#             src, self.src_pad_idx)
#         tgt, tgt_key_padding_mask, tgt_mask = self.prepare_tgt(
#             tgt, self.tgt_pad_idx)
#         memory_key_padding_mask = src_key_padding_mask.clone()
#         output = super().forward(
#             src, tgt, tgt_mask=tgt_mask,
#             src_key_padding_mask=src_key_padding_mask,
#             tgt_key_padding_mask=tgt_key_padding_mask,
#             memory_key_padding_mask = memory_key_padding_mask,
#             )
#         output = rearrange(output, 'T N E -> N T E')
#         return self.linear(output)

# # model = TranslationTransformer() 
model = TranslationTransformer(
    device=DEVICE,
    src_vocab_size=tokenize_src.get_vocab_size(),
    src_pad_idx=tokenize_src.token_to_id('<pad>'),
    tgt_vocab_size=tokenize_tgt.get_vocab_size(),
    tgt_pad_idx=tokenize_tgt.token_to_id('<pad>')
    ).to(DEVICE)
model.init_weights()
model

src = torch.randint(1, 100, (10, 5)).to(DEVICE)  # <1>
tgt = torch.randint(1, 100, (10, 7)).to(DEVICE)
with torch.no_grad():
    output = model(src, tgt)  # <2>
print(output.shape)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(
    ignore_index=tokenize_tgt.token_to_id('<pad>'))  # <1>


def train(model, iterator, optimizer, criterion, clip):

    model.train()  # <1>
    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg
        optimizer.zero_grad()
        output = model(src, trg[:,:-1])  # <2>
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)
def evaluate(model, iterator, criterion):
    model.eval()  # <1>
    epoch_loss = 0

    with torch.no_grad():  # <2>
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg
            output = model(src, trg[:,:-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs
N_EPOCHS = 15
CLIP = 1
BEST_MODEL_FILE = 'best_model.pytorch'
best_valid_loss = float('inf')
for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss = train(
        model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), BEST_MODEL_FILE)
    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    train_ppl = f'{math.exp(train_loss):7.3f}'
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {train_ppl}')
    valid_ppl = f'{math.exp(valid_loss):7.3f}'
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {valid_ppl}')
model.load_state_dict(torch.load(BEST_MODEL_FILE))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
    model.eval()
    if isinstance(sentence, str):
        nlp = spacy.load('de')
        tokens = [token.text.lower() for token in nlp(sentence)]
    else:
        tokens = [token.lower() for token in sentence]
    tokens = [src_field.init_token] + tokens + [src_field.eos_token]  # <1>
    src_indexes = [src_field.vocab.stoi[token] for token in tokens]
    src = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    src, src_key_padding_mask = model.prepare_src(src, SRC_PAD_IDX)
    with torch.no_grad():
        enc_src = model.encoder(src, src_key_padding_mask=src_key_padding_mask)
    trg_indexes = [trg_field.vocab.stoi[trg_field.init_token]]  # <2>
example_idx = 10
src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']
src
trg
translation, attention = translate_sentence(src, SRC, TRG, model, device)
print(f'translation = {translation}')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def display_attention(sentence, translation, attention_weights):
    n_attention = len(attention_weights)

    n_cols = 2
    n_rows = n_attention // n_cols + n_attention % n_cols

    fig = plt.figure(figsize=(15,25))

    for i in range(n_attention):

        attention = attention_weights[i].squeeze(0)
        attention = attention.cpu().detach().numpy()
        cax = ax.matshow(attention, cmap='gist_yarg')

        ax = fig.add_subplot(n_rows, n_cols, i+1)
        ax.tick_params(labelsize=12)
        ax.set_xticklabels([''] + ['<sos>'] + 
            [t.lower() for t in sentence]+['<eos>'],
            rotation=45)
        ax.set_yticklabels(['']+translation)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()
display_attention(src, translation, attention_weights)
example_idx = 25
src = vars(valid_data.examples[example_idx])['src']
trg = vars(valid_data.examples[example_idx])['trg']
print(f'src = {src}')
print(f'trg = {trg}')
translation, attention = translate_sentence(src, SRC, TRG, model, device)
print(f'translation = {translation}')
display_attention(src, translation, attention)
from torchtext.data.metrics import bleu_score
def calculate_bleu(data, src_field, trg_field, model, device, max_len = 50):
    trgs = []
    pred_trgs = []
    for datum in data:
        src = vars(datum)['src']
        trg = vars(datum)['trg']
        pred_trg, _ = translate_sentence(
            src, src_field, trg_field, model, device, max_len)
        # strip <eos> token
        pred_trg = pred_trg[:-1]
        pred_trgs.append(pred_trg)
        trgs.append([trg])

    return bleu_score(pred_trgs, trgs)
bleu_score = calculate_bleu(test_data, SRC, TRG, model, device)
print(f'BLEU score = {bleu_score*100:.2f}')
from transformers import BertModel
model = BertModel.from_pretrained('bert-base-uncased')
print(model)
import pandas as pd
df = pd.read_csv('data/train.csv')  # <1>
df.head()
df.shape
from sklearn.model_selection import train_test_split
random_state=42
labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
X = df[['comment_text']]
y = df[labels]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=random_state)  # <1>
def get_dataset(X, y):
    data = [[X.iloc[i][0], y.iloc[i].values.tolist()] for i in range(X.shape[0])]
    return pd.DataFrame(data, columns=['text', 'labels'])
train_df = get_dataset(X_train, y_train)
eval_df = get_dataset(X_test, y_test)
train_df.shape, eval_df.shape
train_df.head()  # <1>
import logging
logging.basicConfig(level=logging.INFO)  # <1>
model_type = 'bert'  # <2>
model_name = 'bert-base-cased'
output_dir = f'{model_type}-example1-outputs'
model_args = {
    'output_dir': output_dir, # where to save results
    'overwrite_output_dir': True, # allow re-run without having to manually clear output_dir
    'manual_seed': random_state, # <3>
    'no_cache': True,
}
from sklearn.metrics import roc_auc_score
from simpletransformers.classification import MultiLabelClassificationModel
model = MultiLabelClassificationModel(
   model_type, model_name, num_labels=len(labels),
   args=model_args)  # <1>
model.train_model(train_df=train_df)
result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=roc_auc_score) # <1>
result
from preprocessing.preprocessing import TextPreprocessor
tp = TextPreprocessor()
df = df.rename(columns={'comment_text':'original_text'})
df['comment_text'] = df['original_text'].apply(lambda x: tp.preprocess(x)) # <1>
pd.set_option('display.max_colwidth', 45)
df[['original_text', 'comment_text']].head()
model_type = 'bert'
model_name = 'bert-base-cased'
output_dir = f'{model_type}-example2-outputs'  # <1>
best_model_dir = f'{output_dir}/best_model'
model_args = {
    'output_dir': output_dir,
    'overwrite_output_dir': True,
    'manual_seed': random_state,
    'no_cache': True,

    'best_model_dir': best_model_dir,

    'max_seq_length': 300,
    'train_batch_size': 24,
    'eval_batch_size': 24,

    'gradient_accumulation_steps': 1,
    'learning_rate': 5e-5,

    'evaluate_during_training': True,
    'evaluate_during_training_steps': 1000,
    'save_eval_checkpoints': False,
    "save_model_every_epoch": False,
    'save_steps': -1,  # saving model unnecessarily takes time during training
    'reprocess_input_data': True,

    'num_train_epochs': 5,
    'use_early_stopping': True,
    'early_stopping_patience': 4,
    'early_stopping_delta': 0,
}
model = MultiLabelClassificationModel(
    model_type, model_name, num_labels=len(labels),
    args=model_args)
model.train_model(
    train_df=train_df, eval_df=eval_df, acc=roc_auc_score,
    show_running_loss=False, verbose=False)
best_model = MultiLabelClassificationModel(
    model_type, best_model_dir,
    num_labels=len(labels), args=model_args)
result, model_outputs, wrong_predictions = best_model.eval_model(
    eval_df, acc=roc_auc_score)
result
