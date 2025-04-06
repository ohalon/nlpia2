"""
Generate text starting with word sampled from Wikitext-2 vocabulary (33278 words)

Based on github.com/pytorch/examples/ `word_language_model/generate.py`
"""
from pathlib import Path
import argparse
import torch

from preprocessing import Corpus
from model import RNNModel

DEVICE = torch.device('cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Language Model Generator')

    parser.add_argument('--data', type=str, default='./data/wikitext-2',
                        help='location of the data corpus')
    parser.add_argument('--checkpoint', type=str, default='./model.pt',
                        help='Model checkpoint file path to load (default: model.pt)')
    parser.add_argument('--outf', type=str, default='generated.txt',
                        help='Output file to write generated text to.')
    parser.add_argument('--words', type=int, default='1000',
                        help='Number of words to generate')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Temperature (randomness) of generator. Must be greater than 1e-3. Larger values will increase randomness of generated text.')
    parser.add_argument('--log-interval', type=int, default=100,
                        help='reporting interval')
    parser.add_argument('--prompt', type=str, default='',
                        help='Prompt token to seed text generation. Tokens must be separated by spaces. Default = randomly selected.')
    return parser.parse_args()


def generate_word_hidden(model, input_word, vocab=None, hidden_tens=None, temperature=1, device=DEVICE):
    if hidden_tens is None:
        hidden_tens = model.init_hidden()
    if vocab is None:
        vocab = model.vocab

    input_tens = torch.LongTensor(
        [[vocab.word2idx[input_word]]]).to(device)
    output_tens, hidden_tens = model(input_tens, hidden_tens)
    word_weights = output_tens.squeeze().div(temperature).exp().cpu()
    word_idx = torch.multinomial(word_weights, 1)[0]
    # if word_idx == IDX_UNK:
    #     continue
    input_tens.fill_(word_idx)
    return vocab.idx2word[word_idx], hidden_tens


def generate_words(
        model, prompt,
        vocab=None,
        eos_words='<eos>', skip_words='<unk>', max_words=1024,
        temperature=1, seed=None,
        device=DEVICE):
    # IDX_UNK = dictionary.word2idx['<unk>']
    if isinstance(model, (str, Path)):
        with open(model, 'rb') as fin:
            model = torch.load(fin, map_location=device)
            model.eval()  # switch model into evaluation mode instead of training mode

    if vocab is None:
        vocab = model.vocab
    prompt_words = prompt
    if isinstance(prompt_words, str):
        if ' ' in prompt_words:
            prompt_words = prompt_words.split()
        else:
            prompt_words = [prompt_words]
    output_words = []
    max_words = 1024
    hidden_tens = model.init_hidden(batch_size=1)

    for word in prompt_words:
        unused_prediction, hidden_tens = generate_word_hidden(
            model=model, vocab=vocab, input_word=word, hidden_tens=hidden_tens)
    while word and word not in eos_words and len(output_words) < max_words:
        word, hidden_tens = generate_word_hidden(
            model=model, vocab=vocab,
            input_word=word, hidden_tens=hidden_tens,
            temperature=temperature)
        if not skip_words or word not in skip_words:
            output_words.append(word)
    return output_words


def generate_text(model, vocab, prompt, seed=1111, eos_words='<eos>', skip_words='<unk>', max_words=1024):
    words = generate_words(
        model=model, vocab=vocab, prompt=prompt,
        seed=seed, eos_words=eos_words, max_words=max_words)
    return ' '.join([w for w in words if not skip_words or w not in skip_words])


def main():
    args = parse_args()
    corpus = Corpus(args.data)

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()  # switch model into evaluation mode instead of training mode

    is_transformer_model = getattr(model, 'model_type') == 'Transformer'
    if not is_transformer_model:
        hidden = model.init_hidden(1)
    inpt = torch.randint(len(corpus.vocab), (1, 1), dtype=torch.long).to(device)

    with open(args.outf, 'w') as outf:
        with torch.no_grad():  # don't compute or remember gradients
            for i in range(args.words):
                if is_transformer_model:
                    output = model(inpt, False)
                    word_weights = output[-1].squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    word_tensor = torch.Tensor([[word_idx]]).long().to(device)
                    inpt = torch.cat([inpt, word_tensor], 0)
                else:
                    output, hidden = model(inpt, hidden)
                    word_weights = output.squeeze().div(args.temperature).exp().cpu()
                    word_idx = torch.multinomial(word_weights, 1)[0]
                    inpt.fill_(word_idx)

                word = corpus.vocab.idx2word[word_idx]

                outf.write(word + ('\n' if i % 20 == 19 else ' '))

                if i % args.log_interval == 0:
                    print('| Generated {}/{} words'.format(i, args.words))


if __name__ == '__main__':
    corpus = Corpus('data/wikitext-2')
    model = RNNModel('GRU', vocab=corpus.vocab, num_layers=1)
    checkpoint = 'checkpoints/model_epochs_12_rnn_type_GRU_hidden_size_200_batch_size_20_bptt_35_num_layers_1.pt'
    checkpoint = torch.load(open(checkpoint, 'rb'), map_location='cpu')
    model.load_state_dict(checkpoint.state_dict())
# FIXME:
# File ~/code/tangibleai/nlpia2/src/nlpia2/ch08/rnn_word/generate.py:64, in generate_words(model, vocab, prompt, eos_words, skip_words, max_words, temperature, seed)
#      62 output_words = []
#      63 max_words = 1024
# ---> 64 hidden_tens = model.init_hidden()
#      66 for word in prompt_words:
#      67     unused_prediction, hidden_tens = generate_word_hidden(
#      68         model=model, vocab=vocab, input_word=word, hidden_tens=hidden_tens)

# File ~/code/tangibleai/nlpia2/src/nlpia2/ch08/rnn_word/model.py:190, in RNNModel.init_hidden(self, batch_size)
#     182     return (
#     183         weight.new_zeros(
#     184             self.num_layers, self.batch_size, self.hidden_size),
#     185         weight.new_zeros(
#     186             self.num_layers, self.batch_size, self.hidden_size)
#     187     )
#     188 else:
#     189     return weight.new_zeros(
# --> 190         self.num_layers, self.batch_size, self.hidden_size)

# File ~/code/tangibleai/nlpia2/venv/lib/python3.9/site-packages/torch/nn/modules/module.py:1207, in Module.__getattr__(self, name)
#    1205     if name in modules:
#    1206         return modules[name]
# -> 1207 raise AttributeError("'{}' object has no attribute '{}'".format(
#    1208     type(self).__name__, name))

# AttributeError: 'RNNModel' object has no attribute 'batch_size'
#     generate_words(model, vocab=corpus.vocab, prompt='He')
