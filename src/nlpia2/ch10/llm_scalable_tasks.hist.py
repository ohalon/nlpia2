url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/src/nlpia2/data/llm/llm-emmergence-table-other-big-bench-tasks.csv'
df = pd.read_csv(url)
import pandas as pd
df = pd.read_csv(url)
df
df = pd.read_csv(url, index_col=0)
df
df['Emergence'].apply(lambda x: 'linear' in x or 'flat' in x or not x)
df['Scales'] = df['Emergence'].apply(lambda x: 'linear' in x or 'flat' in x or not x)
df.columns
df['Scaling'] = df['Emergence'].apply(lambda x: 'linear' in x or 'flat' in x or not x)
df[['Task', 'Scaling']]
df[['Task', 'Emergence']]['Scaling']
df[['Task', 'Emergence']][df['Scaling']]
scalable_tasks = df[['Task', 'Emergence']][df['Scaling']]
scalable_tasks.columns = ['Task', 'Scaling']
hist -o -p -f llm_scalable_tasks.hist.ipy
hist -f llm_scalable_tasks.hist.py
scalable_tasks
hist -o -p -f llm_scalable_tasks.hist.ipy
hist -f llm_scalable_tasks.hist.py
