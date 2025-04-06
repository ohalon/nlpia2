#!/usr/bin/env python
# coding: utf-8

# #### [`Chapter-11_Information-extraction-and-knowledge-graphs-grounding`](/home/hobs/code/hobs/nlpia-manuscript/manuscript/adoc/Chapter-11_Information-extraction-and-knowledge-graphs-grounding.adoc)

# #### 

# In[ ]:


re.split(r'[!.?]+[\s$]+',
    "Hello World.... Are you there?!?! I'm going to Mars!")


# #### 

# In[ ]:


re.split(
   r'[!.?]+[\s$]+',
   "The author wrote \"'It isn't conscious.' Turing said.\"")


# #### 

# In[ ]:


re.split(r'(?<!\d)\.|\.(?!\d)', "I went to GT.You?")


# #### 

# In[ ]:


import spacy
nlp = spacy.load('en_core_web_md')
doc = nlp("Are you an M.D. Dr. Gebru? either way you are brilliant.")
[s.text for s in doc.sents]


# #### 

# In[ ]:


nlp.pipeline


# #### 

# In[ ]:


nlp = spacy.load("en_core_web_md", exclude=[
   'tok2vec', 'parser', 'lemmatizer',  # <1>
   'ner', 'tagger', 'attribute_ruler'])
nlp.pipeline  # <2>


# #### 

# In[ ]:


nlp.enable_pipe('senter')
nlp.pipeline


# #### 

# In[ ]:


t0 = time.time(); lines2 = extract_lines(nlp=nlp); t1=time.time()
t1 - t0


# #### 

# In[ ]:


df_md = pd.DataFrame(lines)  # <1>
df_fast = pd.DataFrame(lines2)  # <2>
(df_md['sents_spacy'][df_md.is_body]
 == df_fast['sents_spacy'][df_fast.is_body]
 ).sum() / df_md.is_body.sum()


# #### 

# In[ ]:


df_md['sents_spacy'][df_md.is_body]


# #### 

# In[ ]:


df_fast['sents_spacy'][df_fast.is_body]


# #### 

# In[ ]:


import pandas as pd
url = 'https://gitlab.com/tangibleai/nlpia2/-/raw/main/'
url += 'src/nlpia2/data/nlpia_lines.csv'  # <1>
df = pd.read_csv(url, index_col=0)
df9 = df[df.chapter == 9].copy()
df9.shape


# #### 

# In[ ]:


pd.options.display.max_colwidth=25
df9[['text', 'is_title', 'is_body', 'is_bullet']]


# #### 

# In[ ]:


texts = df9.text[df9.is_body]
texts.shape


# #### 

# In[ ]:


from sentence_transformers import SentenceTransformer
minibert = SentenceTransformer('all-MiniLM-L12-v2')
vecs = minibert.encode(list(texts))
vecs.shape


# #### 

# In[ ]:


from numpy.linalg import norm
dfe = pd.DataFrame([list(v / norm(v)) for v in vecs])
cos_sim = dfe.values.dot(dfe.values.T)
cos_sim.shape


# #### 

# In[ ]:


import seaborn as sns
from matplotlib import pyplot as plt
sns.heatmap(cos_sim)


# #### 

# In[ ]:


plt.xticks(rotation=-35, ha='left')
plt.show(block=False)


# #### 

# In[ ]:


from nlpia2 import wikipedia as wiki
page = wiki.page('Timnit Gebru')
text = page.content


# #### 

# In[ ]:


i1 = text.index('Stochastic')
text[i1:i1+51]


# #### 

# In[ ]:


import re
lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'
lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'
sep = r'[,/ ]{1,3}'
re_gps = re.compile(lat + sep + lon)
re_gps.findall('http://...maps/@34.0551066,-118.2496763...')


# #### 

# In[ ]:


doc = nlp(text)
doc.ents[:6]  # <1>


# #### 

# In[ ]:


first_sentence = list(doc.sents)[0]
' '.join(['{}_{}'.format(tok, tok.pos_) for tok in first_sentence])


# #### 

# In[ ]:


spacy.explain('CCONJ')


# #### 

# In[ ]:


' '.join(['{}_{}'.format(tok, tok.tag_) for tok in first_sentence])


# #### 

# In[ ]:


spacy.explain('VBZ')


# #### 

# In[ ]:


import pandas as pd
def token_dict(token):
   return dict(TOK=token.text,
       POS=token.pos_, TAG=token.tag_,


# #### 

# In[ ]:


def doc2df(doc):
   return pd.DataFrame([token_dict(tok) for tok in doc])
pd.options.display.max_colwidth=20
doc2df(doc)


# #### 

# In[ ]:


nlp = spacy.load('en_core_web_lg')
doc = nlp(text)
doc2df(doc)


# #### 

# In[ ]:


i0 = text.index('In a six')
text_gebru = text[i0:i0+308]
text_gebru


# #### 

# In[ ]:


get_ipython().system("python -m spacy.cli download 'en_core_web_trf'  # <1>")
import spacy, coreferee  # <2>
nlptrf = spacy.load('en_core_web_trf')
nlptrf.add_pipe('coreferee')


# #### 

# In[ ]:


doc_gebru = nlptrf(text_gebru)
doc_gebru._.coref_chains


# #### 

# In[ ]:


doc_gebru._.coref_chains.print()


# #### 

# In[ ]:


text = "Gebru was unethically fired from her Ethical AI team."
doc = nlp(text)
doc2df(doc)


# #### 

# In[ ]:


def token_dict2(token):
   d = token_dict(token)
   d['children'] = list(token.children)  # <1>
   return d
token_dict2(doc[0])


# #### 

# In[ ]:


def doc2df(doc):
    df = pd.DataFrame([token_dict2(t) for t in doc])
    return df.set_index('TOK')
doc2df(doc)


# #### 

# In[ ]:


doc2df(doc)['children']['fired']


# #### 

# In[ ]:


from spacy.displacy import render
sentence = "In 1541 Desoto wrote in his journal about the Pascagoula."
parsed_sent = nlp(sentence)
with open('pascagoula.html', 'w') as f:
    f.write(render(docs=parsed_sent, page=True, options=dict(compact=True)))


# #### 

# In[ ]:


import benepar
benepar.download('benepar_en3')


# #### 

# In[ ]:


import spacy
nlp = spacy.load("en_core_web_md")
if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:


# #### 

# In[ ]:


doc_dataframe(nlp("In 1541 Desoto met the Pascagoula."))


# #### 

# In[ ]:


pattern = [
    {'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'},


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


from spacy.matcher import Matcher
doc = nlp("In 1541 Desoto met the Pascagoula.")
matcher = Matcher(nlp.vocab)
matcher.add(
    key='met',
    patterns=[pattern])
matches = matcher(doc)
matches


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


start = matches[0][1]
stop = matches[0][2]
doc[start:stop]  # <2>


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


doc = nlp("October 24: Lewis and Clark met their" \
    "first Mandan Chief, Big White.")
m = matcher(doc)[0]
m


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


doc[m[1]:m[2]]


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


doc = nlp("On 11 October 1986, Gorbachev and Reagan met at Höfði house")
matcher(doc)


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


doc = nlp(
    "On 11 October 1986, Gorbachev and Reagan met at Hofoi house"
    )
pattern = [
    {'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'},
    {'LEMMA': 'and'},
    {'POS': {'IN': ['NOUN', 'PROPN']}, 'OP': '+'},
    {'IS_ALPHA': True, 'OP': '*'},
    {'LEMMA': 'meet'}
    ]
matcher.add('met', None, pattern)  # <1>
matches = matcher(doc)
pd.DataFrame(matches, columns=)


# #### .Creating a POS pattern matcher with spaCy

# In[ ]:


doc[m[-1][1]:m[-1][2]]  # <3>


# #### 

# In[ ]:


import pandas as pd
pd.options.display.max_colwidth = 20
from nlpia2.nell import read_nell_tsv, simplify_names
df = read_nell_tsv(nrows=1000)
df[df.columns[:4]].head()


# #### 

# In[ ]:


pd.options.display.max_colwidth = 40
df['entity'].str.split(':').str[1:].str.join(':')


# #### 

# In[ ]:


df['entity'].str.split(':').str[-1]


# #### 

# In[ ]:


df = simplify_names(df)  # <1>
df[df.columns[[0, 1, 2, 4]]].head()


# #### 

# In[ ]:


islatlon = df['relation'] == 'latlon'
df[islatlon].head()


# #### 

# In[ ]:


def get_wikidata_qid(wikiarticle, wikisite="enwiki"):
    WIKIDATA_URL='https://www.wikidata.org/w/api.php'
    resp = requests.get(WIKIDATA_URL, timeout=5, params={
        'action': 'wbgetentities',
        'titles': wikiarticle,
        'sites': wikisite,
        'props': '',
        'format': 'json'
    }).json()
    return list(resp['entities'])[0]
tg_qid = get_wikidata_qid('Timnit Gebru')
tg_qid


# #### 

# In[ ]:


NOTABLE_WORK_PID = 'P800'     # <1>
INSTANCE_OF_PID = 'P31'       # <2>
SCH_ARTICLE_QID= 'Q13442814'  # <3>
query = f"""
    SELECT ?article WHERE {{


# #### 

# In[ ]:


from SPARQLWrapper import SPARQLWrapper, JSON
endpoint_url = "https://query.wikidata.org/sparql"
sparql = SPARQLWrapper(endpoint_url)
sparql.setReturnFormat(JSON)  # <1>


# #### 

# In[ ]:


sparql.setQuery(query)
result = sparql.queryAndConvert()
result


# #### 

# In[ ]:


import re
uri = result['results']['bindings'][0]['article']['value']
match_id = re.search(r'entity/(Q\d+)', uri)
article_qid = match_id.group(1)
AUTHOR_PID = 'P50'
query = f"""
     SELECT ?author ?authorLabel WHERE {{
     wd:{article_qid} wdt:{AUTHOR_PID} ?author.
     SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
     }}
     """
sparql.setQuery(query)
result = sparql.queryAndConvert()['results']['bindings']
authors = [record['authorLabel']['value'] for record in result]
authors


# #### 

# In[ ]:


query = """
SELECT ?author ?authorLabel WHERE {
    {
    SELECT ?article WHERE {
        wd:Q59753117 wdt:P800 ?article.
        ?article wdt:P31 wd:Q13442814.
        }
    }
    ?article wdt:P50 ?author.
    SERVICE wikibase:label {

