class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
import spacy
spacy_de = spacy.load('de')
spacy_en = spacy.load('en')
def tokenize_de(text):
    return [tok.text for tok in spacy_de.tokenizer(text)]
def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]
import torchtext
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
SRC = Field(tokenize = tokenize_de,
            init_token = '<sos>',  #<1>
            eos_token = '<eos>',
            lower = True,
            batch_first = True)
TRG = Field(tokenize = tokenize_en,
            init_token = '<sos>',
            eos_token = '<eos>',
            lower = True,
            batch_first = True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  #<1>
train_data, valid_data, test_data = Multi30k.splits(exts = ('.de', '.en'),
                                                    fields = (SRC, TRG))
SRC.build_vocab(train_data, min_freq = 2)
TRG.build_vocab(train_data, min_freq = 2)
BATCH_SIZE = 128
train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
     batch_size = BATCH_SIZE,
     device = device)
from torch import Tensor
from typing import Optional, Any
class CustomDecoderLayer(nn.TransformerDecoderLayer):
    def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                mem_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        """Same as DecoderLayer but returns multi-head attention weights.
        """
        tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, attention_weights = self.multihead_attn(tgt, memory, memory,  #<1>
                                                      attn_mask=memory_mask,
                                                      key_padding_mask=mem_key_padding_mask,
                                                      need_weights=True)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, attention_weights  #<2>
class CustomDecoder(nn.TransformerDecoder):
    def __init__(self, decoder_layer, num_layers, norm=None):
         super(CustomDecoder, self).__init__(decoder_layer, num_layers, norm)
from einops import rearrange  #<1>
class TranslationTransformer(nn.Transformer):  #<2>
    def __init__(self, device: str, src_vocab_size: int, src_pad_idx: int,
                 tgt_vocab_size: int, tgt_pad_idx: int, max_sequence_length: int = 100,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
from einops import rearrange
class TranslationTransformer(nn.Transformer):
    def __init__(self, device: str, src_vocab_size: int, src_pad_idx: int,
                 tgt_vocab_size: int, tgt_pad_idx: int, max_sequence_length: int = 100,
                 d_model: int = 512, nhead: int = 8, num_encoder_layers: int = 6,
                 num_decoder_layers: int = 6, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "relu"):
SRC_PAD_IDX = SRC.vocab.stoi[SRC.pad_token]
TRG_PAD_IDX = TRG.vocab.stoi[TRG.pad_token]
model = TranslationTransformer(device=device,
                           src_vocab_size=len(SRC.vocab), src_pad_idx=SRC_PAD_IDX,
                           tgt_vocab_size=len(TRG.vocab), tgt_pad_idx=TRG_PAD_IDX).to(device)
model.init_weights()
src = torch.randint(1, 100, (10, 5)).to('cuda')  #<1>
tgt = torch.randint(1, 100, (10, 7)).to('cuda')
with torch.no_grad():
    output = model(src, tgt)  #<2>
print(output.shape)
LEARNING_RATE = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)  <1>
def train(model, iterator, optimizer, criterion, clip):
def evaluate(model, iterator, criterion):
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
model.load_state_dict(torch.load(BEST_MODEL_FILE))
test_loss = evaluate(model, test_iterator, criterion)
print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
def translate_sentence(sentence, src_field, trg_field, model, device, max_len = 50):
example_idx = 10
src = vars(test_data.examples[example_idx])['src']
trg = vars(test_data.examples[example_idx])['trg']
print(f'src = {src}')
print(f'trg = {trg}')
translation, attention = translate_sentence(src, SRC, TRG, model, device)
print(f'translation = {translation}')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
def display_attention(sentence, translation, attention_weights):
    n_attention = len(attention_weights)
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
model = MultiLabelClassificationModel(model_type, model_name, num_labels=len(labels),
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
model = MultiLabelClassificationModel(model_type, model_name, num_labels=len(labels),
                                      args=model_args)
model.train_model(train_df=train_df, eval_df=eval_df, acc=roc_auc_score,
                  show_running_loss=False, verbose=False)
best_model = MultiLabelClassificationModel(model_type, best_model_dir,
                                           num_labels=len(labels), args=model_args)
result, model_outputs, wrong_predictions = best_model.eval_model(eval_df, acc=roc_auc_score)
result
