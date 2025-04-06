import pandas as pd
from pathlib import Path

THIS_DIR = Path(__file__).expanduser().resolve().absolute().parent
DATA_DIR = Path(THIS_DIR).parent / 'data'

if __name__ == '__main__':
    df_fake = pd.read_csv(Path(DATA_DIR) / 'Fake.csv', index_col=None)
    print(df_fake.shape)
    df_true = pd.read_csv(Path(DATA_DIR) / 'True.csv', index_col=None)
    print(df_true.shape)
    df_fake['isfake'] = 1
    df_true['isfake'] = 0
    df = pd.concat([df_true, df_fake], ignore_index=True)
    """
    >>> import pandas as pd
    ... from pathlib import Path
    ... DATA_DIR = Path('~/code/tangibleai/fake-news/data').expanduser().resolve().absolute()
    ... df_fake = pd.read_csv(Path(DATA_DIR) / 'Fake.csv', index_col=None)
    ... df_true = pd.read_csv(Path(DATA_DIR) / 'True.csv', index_col=None)
    ... df_fake['isfake'] = 1
    ... df_true['isfake'] = 0
    ... df = pd.concat([df_true, df_fake], ignore_index=True)
    >>> df.describe(include='all')
                                                        title   text       subject                date        isfake
    count                                               44898  44898         44898               44898  44898.000000
    unique                                              38729  38646             8                2397           NaN
    top     Factbox: Trump fills top jobs for his administ...         politicsNews  December 20, 2017            NaN
    freq                                                   14    627         11272                 182           NaN
    mean                                                  NaN    NaN           NaN                 NaN      0.522985
    std                                                   NaN    NaN           NaN                 NaN      0.499477
    min                                                   NaN    NaN           NaN                 NaN      0.000000
    25%                                                   NaN    NaN           NaN                 NaN      0.000000
    50%                                                   NaN    NaN           NaN                 NaN      1.000000
    75%                                                   NaN    NaN           NaN                 NaN      1.000000
    max                                                   NaN    NaN           NaN                 NaN      1.000000
    """

    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    """
    >>> df['date'].isna().sum()
    10
    """

    df['date_isna'] = df['date'].isna()
    df['date'] = df['date'].fillna('mean')
    """
    >>> df['date'].isna().sum()
    0
    """
    filepath = Path(DATA_DIR) / 'all.csv.gz'
    df.to_csv(filepath, compression='gzip', index=False)
    print(filepath)
