from char_rnn_from_scratch_refactored import *
rnn3 = RNN(vocab_size=len(meta3['char2i']), n_hidden=128, n_categories=len(meta3['categories']))
load_model_meta(filepath2)
filepath2 = 'char_rnn_from_scratch_refactored-1_311-17min_28sec'
load_model_meta(filepath2)
meta3 = _
meta3
rnn3 = RNN(vocab_size=len(meta3['char2i']), n_hidden=128, n_categories=len(meta3['categories']))
RNN??
rnn3 = RNN(vocab_size=len(meta3['char2i']), n_hidden=128, n_categories=len(meta3['categories']))
rnn3.load_state_dict(meta3['state_dict'])
predict_category("Davletyarov", categories=meta3['categories'], char2i=meta3['char2i'], model=rnn3)
predict_category??
evaluate_tensor??
model.__call__
rnn3.__call__
predict_category("Bilal", categories=meta3['categories'], char2i=meta3['char2i'], model=rnn3)
predict_category("Turan", categories=meta3['categories'], char2i=meta3['char2i'], model=rnn3)
meta3['categories']
predict_category("Doestoevsky", categories=meta3['categories'], char2i=meta3['char2i'], model=rnn3)
predict_category("Nakamoto", categories=meta3['categories'], char2i=meta3['char2i'], model=rnn3)
dir(rnn3)
rnn3.i2o
rnn3.i2o.shape
rnn3.i2o.size
rnn3.i2o.size()
rnn3.i2o.weight
rnn3.i2o.weight.size()
len(char2i)
char2i = meta3['char2i']
len(char2i)
char2i
predict_category("Nakamoto", model=rnn3)
predict_category("O'Neal", model=rnn3)
predict_category("Pie're", model=rnn3)
predict_category("Piere", model=rnn3)
58 + 128
len(meta3['categories'])
rnn3??
    def forward(self, char_tens, hidden):  # <2> x = input = char_tens
        combined = torch.cat((char_tens, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden
forward
rnn3.forward = forward
predict_category("Piere", model=rnn3)
rnn3??
class RNN(nn.Module):
    def __init__(self, vocab_size, n_hidden, n_categories):
        super(RNN, self).__init__()

        self.n_hidden = n_hidden
        self.n_categories = n_categories  # <1> n_categories = n_outputs (one-hot)

        self.i2h = nn.Linear(vocab_size + n_hidden, n_hidden)
        self.i2o = nn.Linear(vocab_size + n_hidden, n_categories)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, char_tens, hidden):  # <2> x = input = char_tens
        combined = torch.cat((char_tens, hidden), 1)
        
        hidden = self.i2h(combined)
        print(f"hidden: {hidden.size()}")
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, self.n_hidden)
rnn3 = RNN(vocab_size=len(meta3['char2i']), n_hidden=128, n_categories=len(meta3['categories']))
rnn3.load_state_dict(meta3['state_dict'])
predict_category("Piere", model=rnn3)
hist -f working/test_piere_load_model_1_311.hist.py
pwd
hist -f test_piere_load_model_1_311.hist.py
