#!/usr/bin/env python

import pandas as pd
df = pd.read_csv('../nlpia-manuscript/manuscript/csv/code_lines_all.csv', index_col=0)
islong = df['line'].str.len() > 72
# df[['filename', 'lineno', 'line', 'prompted_line']][islong].to_csv('code_lines_all_toolong.csv')
iscomment = ~df['comment'].isna()
islongcomment = iscomment & (df['line'].str.len() > 50)
df['is_comment'] = iscomment
df['is_long'] = islong
df['is_too_long'] = islongcomment | islong
df['is_long_comment'] = islongcomment
df[[c for c in df.columns if c.startswith('is')]].sum()
df.to_csv('code_lines_all.csv')
df[['filename', 'lineno', 'line', 'prompted_line'] + [c for c in df.columns if c.startswith('is')]][df['is_too_long']].to_csv('code_lines_all_toolong.csv')
