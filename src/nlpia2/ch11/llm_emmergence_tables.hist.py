import pandas as pd
pd.read_csv('llm-emmergence-table.csv')
pd.read_csv('llm-emmergence-table.csv')
df = _
df
df = pd.read_csv('llm-emmergence-table.csv')
df = pd.read_csv('llm-emmergence-table.csv')
df
df['Reference'] = df['Author'].str + df['Year'].str
df.columns
df = pd.read_csv('llm-emmergence-table.csv')
df.columns
df = pd.read_csv('llm-emmergence-table.csv')
df.columns
df['Reference'] = df['Author'].str + df['Year'].str
df['Reference'] = df['Author'].str() + df['Year'].str()
df['Reference'] = df['Author'] + df['Year']
df.Reference
df.to_csv('llm-emmergence-table-cleaned.csv', index=False)
df.to_csv('llm-emmergence-table-cleaned.csv')
df = pd.read_csv('llm-emmergence-table-cleaned.csv')
df
df = pd.read_csv('llm-emmergence-table.csv')
df['Reference'] = df['Author'] + df['Year']
df.columns
df
df['Prompt Type'] = df.index.values
df.index = range(len(df))
df.columns
df.head()
df.head().T
df.iloc[0].T
columns = list(df.columns)
df = pd.read_csv('llm-emmergence-table.csv')
df['Reference'] = df['Author'] + df['Year']
df['Year'] = df['Year'].str.strip('(').str.strip(')')
columns = list(df.columns)
df
df['Year'] = df['Year'].str.strip('(')
df['Year'] = df['Year'].str.strip().str.strip('(').str.strip(')')
df
palm = '''analytic entailment, codenames, common morpheme, fact checker, figure of speech detection, gender inclusive
sentences german, hindu knowledge, international phonetic alphabet transliterate, irony identification, logical
args, logical deduction, misconceptions, modified arithmetic, phrase relatedness, physical intuition, question
answer creation, repeat copy logic, self evaluation tutoring, social iqa, sports understanding, strange stories,
strategyqa, swahili english proverbs, word sorting, word unscrambling'''.split(',')
palm = [p.strip() for p in palm]
palm
gpt3 = LaMDA = palm
palm = [s.strip() for s in '''anachronisms, analogical similarity, ascii word recognition, auto debugging, causal judgment, code line
description, conceptual combinations, crass ai, cryptonite, cs algorithms, disambiguation qa, elementary
math qa, emoji movie, english proverbs, english russian proverbs, geometric shapes, goal step wikihow,
gre reading comprehension, hinglish toxicity, hyperbaton, identify odd metaphor, international phonetic
alphabet nli, language identification, linguistics puzzles, logic grid puzzle, logical fallacy detection, logical
sequence, metaphor boolean, metaphor understanding, movie dialog same or different, odd one out, parsinlu
qa, parsinlu reading comprehension, physics questions, question selection, snarks, sufficient information,
temporal sequences, timedial, understanding fables, unit interpretation, vitaminc fact verification'''.split(',')]
flat = [s.strip() for s in '''abstraction and reasoning corpus, authorship verification, checkmate in one, chinese remainder theorem, cifar10
classification, color, com2sense, cycled letters, discourse marker prediction, formal fallacies syllogisms negation,
hhh alignment, kanji ascii, kannada, key value maps, language games, mathematical induction, minute
mysteries qa, misconceptions russian, mnist ascii, multistep arithmetic, navigate, paragraph segmentation,
play dialog same or different, presuppositions as nli, program synthesis, python programming challenge, real
or fake text, roots optimization and games, salient translation error detection, self awareness, semantic parsing
in context sparc, semantic parsing spider, simple text editing, sudoku, symbol interpretation, talkdown, tense,
text navigation game, topical chat, tracking shuffled objects, twenty questions, web of lies, which wiki edit,
winowhy, word problems on sets and graphs'''.split(',')]
smooth = '''abstract narrative understanding, auto categorization, bbq lite json, cause and effect, chess state tracking, con-
lang translation, context definition alignment, contextual parametric knowledge conflicts, coqa conversational
question answering, cryobiology spanish, date understanding, emojis emotion prediction, empirical judgments,
entailed polarity, evaluating information essentiality, forecasting subquestions, gem, general knowledge, hindi
question answering, human organs senses, implicatures, implicit relations, intent recognition, linguistic
mappings, list functions, matrixshapes, mult data wrangling, multiemo, natural instructions, nonsense words
grammar, object counting, operators, penguins in a table, physics, polish sequence labeling, qa wikidata,
reasoning about colored objects, rephrase, riddle sense, sentence ambiguity, similarities abstraction, simp
turing concept, simple arithmetic, simple arithmetic json, simple arithmetic json multiple choice, simple
arithmetic json subtasks, simple arithmetic multiple targets json, simple ethical questions, squad shifts,
subject verb agreement, swedish to german proverbs, undo permutation, unit conversion, unnatural in context
learning, bridging anaphora resolution barqa, disfl qa, novel concepts, periodic elements'''
smooth = [p.split(',').strip() for p in smooth]
smooth
smooth = [s.strip() for s in smooth.split(',')]
df_scale = pd.DataFrame({'scaling': ['smooth']*len(smooth)})
df_scale
df_scale['Task'] = smooth
df_scale2 = pd.concat([df_scale, pd.DataFrame({'scaling': ['PaLM']*len(smooth), 'task': palm})], ignore_index=True)
df_scale
df_scale2 = pd.concat([df_scale, pd.DataFrame({'scaling': ['PaLM']*len(palm), 'task': palm})], ignore_index=True)
df_scale2
df_scale2 = pd.concat([df_scale, pd.DataFrame({'scaling': ['PaLM']*len(palm), 'Task': palm})], ignore_index=True)
who
df_scale3 = pd.concat([df_scale2, pd.DataFrame({'scaling': ['flat']*len(flat), 'Task': flat})], ignore_index=True)
df_scale4 = pd.concat([df_scale3, pd.DataFrame({'scaling': ['GPT-3']*len(flat), 'Task': gpt3})], ignore_index=True)
df_scale4 = pd.concat([df_scale3, pd.DataFrame({'scaling': ['GPT-3']*len(gpt3), 'Task': gpt3})], ignore_index=True)
hist
palm
other = [s.strip() for s in other.split(',')]
other = '''boolean expressions, crash blossom, dynamic counting,
entailed polarity hindi, epistemic reasoning, factuality of summary, fantasy reasoning, gender sensitivity
chinese, gender sensitivity english, high low game, identify math theorems, intersect geometry, muslim violence
bias, persian idioms, protein interacting sites, scientific press release, self evaluation courtroom, social support,
spelling bee, taboo, training on test set, truthful qa, yes no black white, dark humor detection, dyck languages,
moral permissibility, ruin names'''
other = [s.strip() for s in other.split(',')]
df_scale5 = pd.concat([df_scale4, pd.DataFrame({'scaling': ['sublinear']*len(other), 'Task': other})], ignore_index=True)
df_scale5
df_scale5['scaling'].unique()
df_scale5['scaling'].replace('smooth', 'linear scaling')
df_scale5['Scaling'] = df_scale5['scaling'].replace('smooth', 'linear scaling')
df_scale5['Scaling'] = df_scale5['Scaling'].replace('sublinear', 'sublinear scaling')
df_scale5.columns
df_scale5.drop(columns=['scaling'])
dfscale = df_scale5.drop(columns=['scaling'])
dfscale.columns = 'Task Emergence'.split()
dfscale['Emergence'].unique()
palm62 = '''nachronisms, ascii word recognition, conceptual combinations, cryptonite, disam-
biguation qa, emoji movie, goal step wikihow, gre reading comprehension, linguistics puzzles, logic grid puzzle,
metaphor boolean, metaphor understanding, odd one out, parsinlu qa.'''
palm62 = [s.strip() for s in palm62.split(',')]
dfscale5 = pd.concat([dfscale, pd.DataFrame({'Emergence': ['PaLM-62B']*len(palm62), 'Task': palm62})], ignore_index=True)
dfscale5
dfscale5['Emergence'] = dfscale5['Emergence'].replace('GPT-3', 'GPT-3/LaMDA')
dfscale5.sample(20)
dfscale5['Emergence'].unique()
dfscale5.to_csv('llm-emmergence-table-other-big-bench-tasks.csv')
dfother = pd.read_csv('llm-emmergence-table-other-big-bench-tasks.csv')
dfother
dfother = pd.read_csv('llm-emmergence-table-other-big-bench-tasks.csv', index_col=0)
dfother
dfscale5['Emergence'].unique()
df
df.drop(index=[9])
df = df.drop(index=[9])
df['Prompt-Task'] = df['Prompt'].str + ' - '
df['Prompt-Task'] = df['Prompt'].str + [' - ']*len(df)
df['Prompt-Task'] = df['Prompt'].str + pd.Series([' - ']*len(df), index=df.index)
df['Prompt-Task'] = df['Prompt'] + pd.Series([' - ']*len(df), index=df.index)
df['Prompt-Task'] = df['Prompt-Task'] + df['Task']
df
df['Prompt-Task'] = df['Prompt'].str.strip() + pd.Series([': ']*len(df), index=df.index)
df['Prompt-Task'] = df['Prompt-Task'] + df['Task'].str.strip()
df[c] = df[c].str.strip()
for c in df.columns: df[c] = df[c].str.strip()
df
df.columns
df.to_csv('llm-emmergence-table-cleaned.csv')
dfscale5
dfscale
dfscale=dfscale5
dfscale['Prompt-Task'] = dfscale['Task']
pd.concat([dfscale, df[['Prompt-Task', 'Task', 'Model']]], ignore_index=True)
df.columns
df['Emergence'] = df['Model']
pd.concat([dfscale, df[['Prompt-Task', 'Task', 'Emergence']]], ignore_index=True)
pd.concat([dfscale, df[['Prompt', 'Task', 'Prompt-Task', 'Emergence']]], ignore_index=True)
pd.concat([dfscale, df[['Prompt', 'Task', 'Prompt-Task', 'Emergence']]], ignore_index=True).fillna('')
dfscale = pd.concat([dfscale, df[['Prompt', 'Task', 'Prompt-Task', 'Emergence']]], ignore_index=True).fillna('')
dfscale.to_csv('llm-emmergence-table-combined-tasks.csv')
hist -o -p -f llm_emmergence_tables.hist.ipy
hist -f llm_emmergence_tables.hist.py
