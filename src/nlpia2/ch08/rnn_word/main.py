from pathlib import Path
import inspect
import sys
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from tqdm import tqdm

# DEVICE = device = torch.device('cpu')  # 'cuda', 'cuda:0', 'cuda:1'

try:
    from nlpia2 import torch_utils
except ImportError:
    __file__ = inspect.getfile(inspect.currentframe())
    NLPIA2_PACKAGE_PATH = str(Path(__file__).absolute().parent.parent.parent.parent)
    print(f'WARNING: Added {NLPIA2_PACKAGE_PATH} to PYTHONPATH')
    sys.path.append(NLPIA2_PACKAGE_PATH)

    from nlpia2 import torch_utils

from preprocessing import Corpus, batchify
import model as rnn_models


def try_exp(num):
    try:
        return math.exp(num)
    except Exception:
        return -1 * num


DEFAULT_HYPERPARAMS = dict(
    early_stop_fract=0.001,
    early_stop_count=2,
    no_improvement_count_max=1,
    batch_size=20,
    seqlen=35,
    clip=0.25,
    cuda=True,
    datapath='./data/wikitext-2',
    device='cuda',
    dropout=0.0,
    dry_run=False,
    emsize=200,
    epochs=1,
    log_interval=500,
    lr=3,
    rnn_type='RNN_TANH',
    nhead=2,
    hidden_size=200,
    num_layers=2,
    onnx_export='',
    save='model.pt',
    filename='model.pt',
    seed=1111,
    tied=False,
)


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM/GRU/Transformer Language Model')
    parser.add_argument('--datapath', type=str, default=DEFAULT_HYPERPARAMS['datapath'],
                        help='location of the data corpus')
    parser.add_argument('--rnn_type', type=str, default=DEFAULT_HYPERPARAMS['rnn_type'],
                        help='type of network (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=DEFAULT_HYPERPARAMS['emsize'],
                        help='size of word embeddings')
    parser.add_argument('--hidden_size', type=int, default=DEFAULT_HYPERPARAMS['hidden_size'],
                        help='number of hidden units per layer')
    parser.add_argument('--num_layers', type=int, default=DEFAULT_HYPERPARAMS['num_layers'],
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=DEFAULT_HYPERPARAMS['lr'],
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=DEFAULT_HYPERPARAMS['clip'],
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=DEFAULT_HYPERPARAMS['epochs'],
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_HYPERPARAMS['batch_size'], metavar='N',
                        help='split each document into this number of independently trained batches (columns)')
    parser.add_argument('--seqlen', type=int, default=DEFAULT_HYPERPARAMS['seqlen'],
                        help='Number of tokens in an individual document (phrase, sentence, paragraph)')
    parser.add_argument('--dropout', type=float, default=DEFAULT_HYPERPARAMS['dropout'],
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=DEFAULT_HYPERPARAMS['seed'],
                        help='random seed')
    parser.add_argument('--device', type=str, default=DEFAULT_HYPERPARAMS['device'],
                        help='device string to use in torch.device() call')
    parser.add_argument('--cuda', action='store_true', default=DEFAULT_HYPERPARAMS['cuda'],
                        help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=DEFAULT_HYPERPARAMS['log_interval'], metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default=DEFAULT_HYPERPARAMS['filename'],
                        help='path to save the final model')
    parser.add_argument('--onnx_export', type=str, default=DEFAULT_HYPERPARAMS['onnx_export'],
                        help='path to export the final model in onnx format')
    parser.add_argument('--nhead', type=int, default=DEFAULT_HYPERPARAMS['nhead'],
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--dry_run', action='store_true', default=DEFAULT_HYPERPARAMS['dry_run'],
                        help='verify the code and the model')

    parser.add_argument('--annealing_loss_improvement_pct', type=float, default=1.0,
                        help='For each epoch, if the loss is not smaller than this fraction of the previous best loss, the learning rate is reduced (default = 1.0).')
    parser.add_argument('--early_stop_fract', type=float,
                        default=DEFAULT_HYPERPARAMS['early_stop_fract'],
                        help='If the loss does not improve by this amount for no_improvement_count_max then stop training.',
                        )
    parser.add_argument('--early_stop_count', type=float,
                        default=DEFAULT_HYPERPARAMS['no_improvement_count_max'],
                        help='If the loss does not improve by stop_improvement_fraction amount for this number of epochs then stop training.',
                        )
    args = parser.parse_args()

    return args


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i):
    seq_len = min(kwargs['seqlen'], len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def evaluate(model, criterion, ntokens=None, eval_batch_size=None, data_source=None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    if kwargs['rnn_type'] != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, kwargs['seqlen']):
            data, targets = get_batch(data_source, i)
            if kwargs['rnn_type'] == 'Transformer':
                output = model(data)
                output = output.view(-1, ntokens)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            total_loss += len(data) * criterion(output, targets).item()
    return total_loss / (len(data_source) - 1)


def train_epoch(model, train_data, ntokens, criterion=nn.NLLLoss(), lr=2.0, **kwargs):
    # Training mode enables dropout layers
    model.train()
    total_loss = 0.
    start_time = time.time()
    log_interval = kwargs.get('log_interval', 500)

    if kwargs['rnn_type'] != 'Transformer':
        hidden = model.init_hidden(kwargs['batch_size'])
    for batch, i in tqdm(
            enumerate(range(0, train_data.size(0) - 1, kwargs['seqlen'])),
            total=len(train_data) // kwargs['seqlen']):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        model.zero_grad()
        if kwargs['rnn_type'] == 'Transformer':
            output = model(data)
            output = output.view(-1, ntokens)
        else:
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
        loss = criterion(output, targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs['clip'])
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()

        if batch and batch % kwargs['log_interval'] == 0:
            cur_loss = total_loss / kwargs['log_interval']
            elapsed = time.time() - start_time

            print((' | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                   'loss {:5.2f} | ppl {:8.2f}').format(
                batch, len(train_data) // kwargs['seqlen'], lr,
                elapsed * 1000 / kwargs['log_interval'],
                cur_loss,
                try_exp(cur_loss)))
            total_loss = 0
            start_time = time.time()
        if kwargs['dry_run']:
            break


def export_onnx(model, path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}.'.format(os.path.realpath(kwargs['onnx_export'])))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)


def main(
        stop_improvement_fraction=0.00001,
        no_improvement_count_max=5,
        **kwargs):
    default_kwargs = DEFAULT_HYPERPARAMS.copy()
    default_kwargs.update(vars(parse_args()))
    default_kwargs.update(kwargs)
    kwargs = default_kwargs
    corpus = Corpus(kwargs['datapath'])

    # Set the random seed manually for reproducibility.
    torch.manual_seed(kwargs['seed'])
    if torch.cuda.is_available():
        if not kwargs['cuda']:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda.")

    device = kwargs['device'] or ("cuda" if kwargs['cuda'] else "cpu")
    device = torch.device(device)

    model = rnn_models.RNNModel(vocab=corpus.vocab, **kwargs).to(device)

    ###############################################################################
    # Training

    batch_size = kwargs['batch_size']  # 10
    train_data = batchify(dataset=corpus.train, batch_size=batch_size, device=device)
    print(f'batchify(corpus.train, batch_size={batch_size}).size(): {train_data.size()}')
    val_data = batchify(dataset=corpus.valid, batch_size=batch_size, device=device)
    print(f'batchify(corpus.valid, batch_size={batch_size}).size(): {val_data.size()}')
    test_data = batchify(dataset=corpus.test, batch_size=batch_size, device=device)
    print(f'batchify(corpus.test, batch_size={batch_size}).size(): {test_data.size()}')
    checkpoint_filename = kwargs['filename']
    print(f'checkpoint_filename: {checkpoint_filename}')
    print(f'log_interval: {kwargs["log_interval"]}')

    # get_batch subdivides the source data into chunks of length kwargs['seqlen'].
    # If source is equal to the example output of the batchify function, with
    # a seqlen-limit of 2, we'd get the following two Variables for i = 0:
    # ┌ a g m s ┐ ┌ b h n t ┐
    # └ b h n t ┘ └ c i o u ┘
    # Note that despite the name of the function, the subdivison of data is not
    # done along the batch dimension (i.e. dimension 1), since that was handled
    # by the batchify function. The chunks are along dimension 0, corresponding
    # to the seq_len dimension in the LSTM.

    # Loop over epochs.
    lr = kwargs['lr']

    results = kwargs.copy()
    total_time = 0
    epoch_time = 0

    best_loss = 1e6
    no_improvement_count = 0

    # [ctrl]-C to break out of training early and retain the latest best checkpoint (model.pt)
    try:
        for epoch_num in tqdm(range(1, kwargs['epochs'] + 1)):
            epoch_start_time = time.time()

            train_epoch(
                model=model,
                criterion=nn.NLLLoss(),
                ntokens=len(corpus.vocab.idx2word),
                train_data=train_data,
                **kwargs)
            val_loss = evaluate(
                model=model,
                criterion=nn.NLLLoss(),
                ntokens=len(corpus.vocab.idx2word),
                data_source=val_data)
            epoch_time = time.time() - epoch_start_time
            total_time += epoch_time
            results.update(dict(
                best_loss=best_loss,
                epoch_num=epoch_num,
                epoch_time=epoch_time,
                total_time=total_time,
                val_loss=val_loss,
                val_perplexity=(val_loss)))
            print('-' * 89)
            print(('| epoch {epoch_num:3d} | time: {epoch_time:5.2f}s | total: {total_time:6.2f}s'
                   '| val loss {val_loss:5.2f}').format(**results))
            #       ' | valid ppl {val_perplexity:8.2f}').format(**results))
            print('-' * 89)

            improvement = best_loss - val_loss

            # Save the model if the validation loss is the best we've seen so far.
            if improvement > 0:
                with open(checkpoint_filename, 'wb') as f:
                    torch.save(model, f)
                best_loss = val_loss
                no_improvement_count = 0
                print(f'TRAINING best_loss: {best_loss:7.3f} is {improvement * 100. / best_loss:7.3f}% improvement')
            if improvement < stop_improvement_fraction * best_loss:
                no_improvement_count += 1
                # Reduce the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4
                print(f'TRAINING no improvement count: {no_improvement_count}, new lr: {lr:7.3f}')

            if no_improvement_count >= no_improvement_count_max:
                print(f'Stopping training early at best_loss: {best_loss:7.4f}.')
                break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(checkpoint_filename, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if kwargs['rnn_type'] in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    results['test_loss'] = evaluate(
        model=model,
        criterion=nn.NLLLoss(),
        ntokens=len(corpus.vocab.idx2word),
        data_source=test_data)
    results['test_perplexity'] = try_exp(results['test_loss'])
    print('=' * 89)
    print('| End of training | test loss {test_loss:5.2f} | test ppl {test_perplexity:8.2f}'.format(
        **results))
    results['learned_parameters'] = torch_utils.count_parameters(model)
    print(' {learned_parameters:6d} learned params for {rnn_type}'.format(**results))
    print('=' * 89)

    if kwargs['onnx_export']:
        onnx_batch_size = 1
        # Export the model in ONNX format.
        export_onnx(
            model,
            kwargs['onnx_export'],
            batch_size=onnx_batch_size,
            seq_len=kwargs['seqlen']
        )

    return dict(model=model, results=results)


if __name__ == '__main__':
    args = parse_args()
    kwargs = vars(args)
    trained_model, results = main(**kwargs).values()
