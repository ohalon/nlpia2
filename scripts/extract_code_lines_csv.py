#!/usr/bin/env python3
import pandas as pd
from nlpia2.text_processing.extractors import extract_tagged_code_lines
from nlpia2.constants import OFFICIAL_ADOC_DIR, MANUSCRIPT_DIR

if __name__ == '__main__':
    dfs = []
    for f in OFFICIAL_ADOC_DIR.glob('*.adoc'):
        df = pd.DataFrame(extract_tagged_code_lines(f))
        df['filepath'] = str(f)
        df['filename'] = f.name
        dfs.append(df)

    datadir = MANUSCRIPT_DIR / 'csv'
    datadir.mkdir(exist_ok=True)
    df = pd.concat(dfs)
    df.to_csv(datadir / 'code_lines_all.csv')
