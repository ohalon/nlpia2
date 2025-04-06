from char_rnn_from_scratch_refactored import *
cd nlpia-rnn/
from char_rnn_from_scratch_refactored import *
len(CHAR2I)
len(CATEGORIES)
186 / 58
META
META['model']
META['rnn']
META['model']
rnn
predict_category('Nakamoto')
predict_category("Abe")
predict_category("Abe'")
predict_category("Abe'")
predict_category("Satoshi")
predict_category("Chen")
predict_category("O'Leary")
predict_category("O'Callaghan")
rnn.state_dict
rnn.state_dict()
meta['state_dict']
META['state_dict']
META.keys()
filename = 'char_rnn_from_scratch_refactored-1_517-09min_46sec'
help(load_model)
load_model(filename)
rnn_old = rnn
rnn = RNN()
rnn = RNN(vocab_size=len(META['char2i']))
rnn = RNN(vocab_size=len(META['char2i']), n_hidden=128, n_categories=len(CATEGORIES))
rnn
META2 = load_model(filename)
rnn = RNN(vocab_size=len(META2['char2i']), n_hidden=128, n_categories=len(META2['categories']))
rnn
rnn.load_state_dict(META['state_dict'])
rnn.load_state_dict(META2['state_dict'])
predict_category("O'Callaghan")
predict_category("O'Leary")
predict_category("Nakamoto")
ls -hal
filename
print_dataset_samples(df)
df = load_dataset()
print_dataset_samples(df)
print_dataset_samples(df)
predict_category("Davletyarov")
print_predictions('Devorak')
    print(f"META['categories']: {META['categories']}")
    print(f'CATEGORIES: {CATEGORIES}')
    print()
    print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
    print_predictions(input_line='Fyodor', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Dostoevsky', n_predictions=3, categories=CATEGORIES)
    print()
    print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
    print_predictions(input_line='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Sanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Koyejo', n_predictions=3, categories=CATEGORIES)
    print()
    print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
    print_predictions(input_line='Satoshi', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Nakamoto', n_predictions=3, categories=CATEGORIES)
    print()
    print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
    print_predictions(input_line='Rediet', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Abebe', n_predictions=3, categories=CATEGORIES)
    print()
    print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
    print_predictions(input_line='Silvio', n_predictions=3, categories=CATEGORIES)
    print_predictions(input_line='Micali', n_predictions=3, categories=CATEGORIES)
    print(f"META['categories']: {META['categories']}")
    print(f'CATEGORIES: {CATEGORIES}')
    print()
    print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
    print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
    print()
    print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
    print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
    print()
    print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
    print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
    print()
    print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
    print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
    print()
    print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
    print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
print_example_tensor()
58 + 128
help(train)
ls -hal
df
results = train(df)
results
filename
min(results['losses'])
filename = 'char_rnn_from_scratch_refactored-1_311-17min_28sec'
save_model(filename, **results)
filepath2 = _
predict_categ
predict_category("Davletyarov")
    print(f"META['categories']: {META['categories']}")
    print(f'CATEGORIES: {CATEGORIES}')
    print()
    print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
    print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
    print()
    print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
    print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
    print()
    print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
    print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
    print()
    print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
    print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
    print()
    print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
    print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
meta2 = load_model(filepath2)
meta2
results
rnn.state_dict()
rnn.state_dict() == meta2['model'].state_dict()
meta2.keys()
rnn2 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
    print(f"META['categories']: {META['categories']}")
    print(f'CATEGORIES: {CATEGORIES}')
    print()
    print('Russia: https://en.wikipedia.org/wiki/Fyodor_Dostoevsky')
    print_predictions(text='Fyodor', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Dostoevsky', n_predictions=3, categories=CATEGORIES)
    print()
    print('Nigeria: https://en.wikipedia.org/wiki/Sanmi_Koyejo # Oluwasanmi')
    print_predictions(text='Oluwasanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Sanmi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Koyejo', n_predictions=3, categories=CATEGORIES)
    print()
    print('Japan: https://en.wikipedia.org/wiki/Satoshi_Nakamoto')
    print_predictions(text='Satoshi', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Nakamoto', n_predictions=3, categories=CATEGORIES)
    print()
    print('Etheopia: https://en.wikipedia.org/wiki/Rediet_Abebe')
    print_predictions(text='Rediet', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Abebe', n_predictions=3, categories=CATEGORIES)
    print()
    print('Italy: https://en.wikipedia.org/wiki/Silvio_Micali')
    print_predictions(text='Silvio', n_predictions=3, categories=CATEGORIES)
    print_predictions(text='Micali', n_predictions=3, categories=CATEGORIES)
rnn_old = rnn
rnn2 = rnn2.load_state_dict(meta2['state_dict'])
rnn2
def predict_category(name, categories=CATEGORIES, char2i=CHAR2I, model=rnn):
    tensor = encode_one_hot_seq(name, char2i=char2i)
    pred_i = evaluate_tensor(tensor, model=model).topk(1)[1][0].item()
    return categories[pred_i]
predict_category("Davletyarov", model=rnn)
predict_category("Davletyarov", model=rnn2)
RNN
RNN??
RNN??
dir(rnn2)
dir(rnn)
rnn2.load_state_dict(meta2['state_dict'])
rnn2 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
rnn2.load_state_dict(meta2['state_dict'])
filepath2
rnn2
predict_category("Davletyarov", model=rnn2)
predict_category("Davletyarov", model=rnn_old)

def load_model_meta(filepath, model=None):
    """ Return meta dict from filepath.meta.json & state_dict from filepath.state_dict.pickle """
    filepath = Path(filepath)
    with filepath.with_suffix('.meta.json').open('rt') as fin:
        meta = json.load(fin)
    with filepath.with_suffix('.state_dict.pickle').open('rb') as fin:
        state_dict = torch.load(fin)
    meta['state_dict'] = state_dict
    if model is not None:
        model.load_state_dict(state_dict)
    meta['model'] = model
    return meta
load_model_meta(filepath2, model=rnn3)
rnn3 = RNN(vocab_size=len(meta2['char2i']), n_hidden=128, n_categories=len(meta2['categories']))
predict_category("Davletyarov", categories=meta2['categories'], char2i=meta2['char2i'], model=rnn_old)
load_model_meta(filepath2, model=rnn3)
import json
meta3 = load_model_meta(filepath2, model=rnn3)
predict_category("Davletyarov", categories=meta3['categories'], char2i=meta3['char2i'], model=meta3['model'])
filepath2
ls
filepath2
ls working
hist -o -p -f working/test_train_model_1_311.md
hist -f working/test_train_model_1_311.py
