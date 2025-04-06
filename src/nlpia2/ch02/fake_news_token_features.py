import numpy as np
import pandas as pd
import re
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from pandas.api.types import is_numeric_dtype


THIS_DIR = Path(__file__).expanduser().resolve().absolute().parent
DATA_DIR = Path(THIS_DIR).parent / 'data'


def get_text_column_names(df):
    columns = []
    for c in df.columns:
        try:
            df.head(2)[c].str.len()
            columns.append(c)
        except (AttributeError, ValueError) as e:
            pass
            # print(c + ' is not a str: ' + str(e))
    return columns


def extract_len_features(
        df,
        columns=None,
        pattern=r'\w+(?:\'\w+)?|[^\w\s]',
        pattern_name=None):
    if columns is None:
        columns = get_text_column_names(df)
    for c in columns:
        df[f'{c}_len'] = df[c].str.len()
        df[f'{c}_num_token{pattern_name}s'] = df[c].str.findall(pattern).apply(len)
        df[f'{c}_ave_token{pattern_name}_len'] = (df[f'{c}_num_token{pattern_name}s'] / df[c + '_len'])
        df[f'{c}_ave_token{pattern_name}_len_isna'] = df[f'{c}_ave_token{pattern_name}_len'].isna()
        maximum = df[f'{c}_ave_token{pattern_name}_len'].max()
        if np.isnan(maximum):
            maximum = 1
        df[f'{c}_ave_token{pattern_name}_len'] = df[f'{c}_ave_token{pattern_name}_len'].fillna(maximum)
    return df.fillna(0)


def extract_keyword_features(
        df,
        columns=None,
        keywords='Republican|Trump|Donald|Donald Trump|Senat|CORRUPT|CLINTON|OBAMA|TRUMP'.split('|')):
    if columns is None:
        columns = get_text_column_names(df)
    for c in columns:
        for kw in keywords:
            df[c + '_haskw_' + kw] = df[c].str.contains(kw)
    return df


def train(df, model=LogisticRegression(max_iter=10000), feature_cols=None, target_col=None):
    df = df.sample(len(df))  # <1>
    # <1> Shuffle your data to improve convergence when your data is "stratified" by the target as it was here.

    feature_cols = feature_cols or df.columns.values[:-1]
    target_col = target_col or df.columns.values[-1]

    istestset = np.random.rand(len(df)) < .1  # <1>
    # <1> 10% test dataset for model validation

    df_train = df[~istestset]
    df_test = df[istestset]

    X_train = df_train[feature_cols]
    y_train = df_train[target_col]
    X_test = df_test[feature_cols]
    y_test = df_test[target_col]

    model.fit(X_train, y_train)

    return dict(
        model=model,
        train_score=model.score(X_train, y_train),
        test_score=model.score(X_test, y_test),
        coef=pd.Series(model.coef_.flatten(), index=feature_cols).sort_values())


def tokenize(s):
    pattern = r'\w+(?:\'\w+)?|[^\w\s]'
    return re.findall(pattern, s)


def tokenize_words(s):
    wordpat = r'\w+(?:\'\w+)?'
    return re.findall(wordpat, s)


def filter_nonwords(tokens):
    return [t for t in tokens if t and t[0]]


def _wordlen(tokens):
    return tokenize_words()


def extract_kitchen_sink_features(df, ignore=['subject'], columns=['text', 'title', 'date', 'date_isna', 'isfake']):
    df = df[[c for c in columns if c in df.columns]].copy()
    for i in ignore:
        if i in df.columns:
            del df[i]
    for i, p in enumerate([r'\w+(?:\'\w+)?|[^\w\s]', r'[^\w\s]+', r'[A-Z0-9_]+']):
        df = extract_len_features(df, pattern=p, pattern_name=i)
    # print(f'df.shape before kw features: {df.shape}')
    df = extract_keyword_features(df)
    # print(f'df.shape after kw features: {df.shape}')

    numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    # delete uniform (constant) numeric columns
    numeric_cols = [c for c in numeric_cols if df[c].std() > 0]

    target_col = 'isfake'
    feature_cols = [c for c in numeric_cols if c != target_col]
    # print(f'len(numeric_cols): {len(numeric_cols)}')
    # print(f'len(feature_cols): {len(feature_cols)}')
    # print(f'target_col: {target_col}')

    df = df[numeric_cols]
    target = df[target_col].copy()
    del df[target_col]
    df[target_col] = target
    df.to_csv(DATA_DIR / 'numeric.csv.gz', compression='gzip')
    return df


def kitchen_sink_model(df=None, target_col=None):
    if df is None:
        df_filepath = Path(DATA_DIR) / 'all.csv.gz'
        df = pd.read_csv(df_filepath, index_col=None)
        print(filepath)
    # print(f'kitchen sink df.shape: {df.shape}')

    df = extract_kitchen_sink_features(df)
    return train(df, target_col=target_col)


def allcaps_tokens(tokens):
    return [tok for tok in tokens if tok.upper() == tok]


def cheap_trick_model(df, target_col='isfake'):
    feature_names = []
    for col in ['text', 'title']:
        feature_names.append(f'{col}_len')
        df[feature_names[-1]] = df[col].apply(len)

        feature_names.append(f'{col}_allcap_token_len')
        df[feature_names[-1]] = df[col].apply(tokenize).apply(allcaps_tokens).apply(len)

        feature_names.append(f'{col}_allcap_ratio')
        df[feature_names[-1]] = df[feature_names[-2]] / df[feature_names[-3]].replace(0, 1)

    return train(df, feature_cols=feature_names, target_col='isfake')


if __name__ == '__main__':
    filepath = Path(DATA_DIR) / 'all.csv.gz'
    df = pd.read_csv(filepath, index_col=None)

    results = []
    results.append(cheap_trick_model(df, target_col='isfake'))
    results[-1]['description'] = '## 6 ALLCAPS token features'
    print()
    print(results[-1]['description'])
    print('Test Accuracy: ', results[-1]['test_score'])
    print('### Top 5 isfake indicators:')
    print(results[-1]['coef'].sort_values()[-5:])
    print('### Top 5 NOT fake indicators (negative coefficients):')
    print(results[-1]['coef'].sort_values()[:5])
    print()
    results.append(kitchen_sink_model(df, target_col='isfake'))
    results[-1]['description'] = '## kitchen sink features (keywords and token stats)'
    print()
    print(results[-1]['description'])
    print('Test Accuracy: ', results[-1]['test_score'])
    print('### Top 5 isfake indicators:')
    print(results[-1]['coef'].sort_values()[-5:])
    print('### Top 5 NOT fake indicators (negative coefficients):')
    print(results[-1]['coef'].sort_values()[:5])
    print()
    print('# Hyperparameter Table')
    print(pd.DataFrame(results)['model train_score test_score description'.split()])
