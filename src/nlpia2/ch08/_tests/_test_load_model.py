import sys
from pathlib import Path

import pandas as pd
from ch08_rnn_char_nationality import RNN, CATEGORIES, CHAR2I
from ch08_rnn_char_nationality import predict, predict_proba
from persistence import load_model_meta


def check_predictions(model=None, names=None, target='nationality', nationalities=None, char2i=CHAR2I, categories=CATEGORIES, min_correct=.4):
    if names is None or nationalities is None:
        names = 'Fyodor Dostoevsky Oluwasanmi Sanmi Koyejo Satoshi Nakamoto Rediet Abebe Silvio Micali'.split()
        nationalities = ['Russian'] * 2 + ['Nigerian'] * 3 + ['Japanese'] * 2 + ['Ethiopian'] * 2 + ['Italian'] * 2

    preds = []
    for text, cat_true in zip(names, nationalities):
        cat_pred = predict(
            model=model, text=text, char2i=char2i, categories=categories)
        proba = predict_proba(
            model=model, text=text, char2i=char2i, categories=categories)
        preds.append([text, cat_pred, proba, cat_true])

    preds = pd.DataFrame(preds, columns='name prediction probability truth'.split())

    num_correct = (preds['truth'] == preds['prediction']).sum()
    portion_correct = num_correct / len(preds)
    print(f'num_correct: {num_correct}  portion_correct: {portion_correct}')
    if 0 <= min_correct <= 1:
        assert portion_correct > min_correct

    return preds


if __name__ == '__main__':
    filebase = sys.argv[1] if len(sys.argv) > 1 else 'ch08_rnn_char_nationality'

    for fp in Path(filebase).parent.glob(f'{filebase}*.meta.json'):
        filepath3 = fp.with_suffix('')
        meta = load_model_meta(filepath3)
        model = meta.get('model', None)
        meta['n_hidden'] = meta.get('n_hidden', 128)
        char2i = meta['char2i']
        if model is None:
            model = RNN(
                vocab_size=len(meta['char2i']),
                n_hidden=meta['n_hidden'],
                n_categories=len(meta['categories']))
        df = check_predictions(
            model=model, target='nationality', categories=meta['categories'])
        print(df)
