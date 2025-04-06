from nlpia2.spacy_language_model import nlp
from nlpia2.nlpia_wikipedia import wikipedia as wiki
import nlpia2_wikipedia.wikipedia as wiki
page = wiki.page('Timnit_Gebru')
from nlpia2_wikipedia.wikipedia.parsers paragraph_dataframe
from nlpia2_wikipedia.wikipedia.parsers import paragraph_dataframe
from nlpia2_wikipedia.parsers import paragraphs_dataframe
paragraph_dataframe(page)
paragraphs_dataframe(page)
df = paragraphs_dataframe(page)
df[0]
df.iloc[0]
df.iloc[1]
df.iloc[2]
df.iloc[3]
df.iloc[4]
df.iloc[5]
df.iloc[5].text
p = df.ilocdf.iloc[5]
row = df.iloc[5]
row.text
row.index
row.iloc[1:]
row.iloc[1:].join(': ')
list(row.iloc[1:]).join(': ')
': '.join(row.iloc[1:])
': '.join(row.iloc[2:])
': '.join((h for h in row.iloc[2:] if h))
': '.join(h for h in row.iloc[2:] if h)
': '.join(h for h in row.iloc[2:] if h) + ':\n'
text = ': '.join(h for h in row.iloc[2:] if h) + ':\n' + row.text
doc = nlp(text)
doc.noun_chunks
list(doc.noun_chunks)
    df['all_headings'] = [': '.join(h for h in row.iloc[2:] if h) for i, row in df.iterrows()]
    df['all_headings'] = [': '.join(h for h in row.loc['h0':] if h) for i, row in df.iterrows()]
df = df.fillna('')
    df['all_headings'] = [': '.join(h for h in row.loc['h0':] if h) for i, row in df.iterrows()]
df.head()
df
df.columns
hist
def paragraphs_dataframe(page):
    """ Split wikitext into paragraphs and return a dataframe with columns for headings (title, h1, h2, h3, ...)

    TODO: create a method or property within a wikipedia.Page class with this function

    >>> from nlpia2_wikipedia.wikipedia import wikipedia as wiki
    >>> page = wiki.page('Large language model')
    >>> df = paragraphs_dataframe(page)
    >>> df.head(2)

    """
    paragraphs = [p for p in page.content.split('\n\n')]
    headings = [page.title]
    df_paragraphs = []
    for p in paragraphs:
        p = p.strip()
        p_headings = re.findall(RE_HEADING, p)
        # TODO strip headings from front of p
        # TODO use match instead of findall (only need 1)
        while p_headings:
                h = p_headings[0]
                p = p[len(h):].strip()
                h = h.strip()
                level = len([c for c in h if c == '=']) + 1
                h = h.strip('=').strip()
                headings = headings[:level]
                if len(headings) <= level:
                    headings = headings + [''] * (level - len(headings))
                    headings[level - 1] = h
                p_headings = re.findall(RE_HEADING, p)
        if p:
            p_record = dict(text=p, title=page.title)
            p_record.update({f'h{i}': h for (i, h) in enumerate(headings)}) 
            df_paragraphs.append(p_record)
    
    df = pd.DataFrame(df_paragraphs).fillna('')
    df['h_all'] = [
        ': '.join(h for h in row.loc['h0':] if h) for i, row in df.iterrows()]
    return df
df = paragraphs_dataframe(page)
import re
df = paragraphs_dataframe(page)
import pandas as pd

RE_HEADING = r'^\s*[=]+ [^=]+ [=]+\s*'
df = paragraphs_dataframe(page)
df['h_all']
texts = df[['h_all', 'text']].str.join('\n')
texts = df['h_all'] + df['text']
texts
texts = df['h_all'] + '\n' + df['text']
texts = df['h_all'] + ':\n' + df['text']
texts = (df['h_all'] + ':\n' + df['text']).values
texts = list(df['h_all'] + ':\n' + df['text'])
texts[5]
len(tests[5].splitlines())
len(texts[5].splitlines())
docs = [nlp(t) for t in texts]
list(docs[5].noun_chunks)
df['num_nouns'] = [len(list(d.noun_chunks)) for d in docs]
df.sort_values('num_nouns')
df.sort_values('num_nouns')[-1]
df.sort_values('num_nouns').tail(1)
df.sort_values('num_nouns').tail(1).iloc[0]
row = df.sort_values('num_nouns').tail(1).iloc[0]
text = row.h_all + '\n:' + row.text
docs[0]
doc = docs[4]
doc
doc = docs[5]
doc
doc.noun_chunks
nouns = list(doc.noun_chunks)
entities = list(doc.noun_chunks)
entities
entities[0].pos_
entities[0].ents
entities[1].ents
entities
doc.sents
[len(s) for s in doc.sents]
sorted((len(s), s) for s in doc.sents)
sorted((len(s.noun_chunks), s) for s in doc.sents)
sorted((len(list(s.noun_chunks)), s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s), s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.5, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.2, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.25, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.15, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.1, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.05, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.01, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.01, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.02, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.03, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.025, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.027, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**1.03, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s), s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**2, s) for s in doc.sents)
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for s in enumerate(doc.sents))
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for i, s in enumerate(doc.sents))
docs[13]
docs[5].sents[13]
doc.sents[13]
list(doc.sents)[13]
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for i, s in enumerate(doc.sents))
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for i, s in enumerate(doc.sents) if 'Gebru' in s.text)
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for i, s in enumerate(doc.sents) if 'she' in s.text.lower())
len(df)
doc
doc.text
page.content
page.content.find('1 February 2021')
loc = page.content.find('1 February 2021')
page.content[loc-10:loc+10]
page.content[loc-10:loc+100]
doc.text
sorted((len(list(s.noun_chunks)) / len(s)**2, i, s) for i, s in enumerate(doc.sents) if 'she' in s.text.lower())
sents = list(docs.sents)
sents = list(doc.sents)
sents[0]
sents[1]
hist
pwd
hist -o -p -f src/nlpia2/ch11/timnit-gebru-spacy-ents.hist.ipy
hist -f src/nlpia2/ch11/timnit-gebru-spacy-ents.hist.py
