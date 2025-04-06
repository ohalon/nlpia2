from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
# import re   

import warnings
warnings.filterwarnings("ignore")

## for data

## for plotting
import matplotlib.pyplot as plt  #3.3.2
import seaborn as sns  #0.11.1

from nlpia2_wikipedia import wikipedia as wiki

import spacy  #3.5.0
from spacy import displacy
import textacy  #0.12.0

## for graph
import networkx as nx  # 3.0 (also pygraphviz==1.10)

from grounder.spacy_language_model import nlp


# import plotly.graph_objs as go  # 5.1.0

## for timeline
# import dateparser # 1.1.7


wiki.set_lang('en')
page = wiki.page("Cory Doctorow")

# Stop after natural language text sections
text = page.content
text = text[:text.find("== Bibliography ==")]
print(text)
'''
Compute n-grams frequency with nltk tokenizer.
:parameter
    :param text: str
    :param ngrams: int or list - 1 for unigrams, 2 for bigrams, [1,2] for both
    :param top: num - plot the top frequent words
:return
    dtf_count: dtf with word frequency
'''

def count_ngrams(text, ngrams=[1, 2, 3]):
    dfs = []
    tokens = [tok.text for tok in nlp(text)]
    for n in ngrams:
        dic_words_freq = Counter(zip(*[tokens[i:] for i in range(n)]))
        dtf_n = pd.DataFrame(dic_words_freq.most_common(), columns=["word","freq"])
        dtf_n["ngrams"] = n
        dfs.append(dtf_n)
    return pd.concat(dfs)


print(count_ngrams(text))


def get_tfidf(docs, ngram_range=(1, 3)):
    vectorizer = TfidfVectorizer(ngram_range=ngram_range) 
    df = vectorizer.fit_transform(docs)
    df = pd.DataFrame.sparse.from_spmatrix(df)
    df.columns = vectorizer.get_feature_names_out()
    return df


docs = (nlp(line) for line in text.split('\n') if not line.lstrip()[:2] == '==')
sentences = []
for d in docs:
    sentences += [s for s in d.sents]


ngram_counts = count_ngrams(text=text)
print(ngram_counts.head(30))


def plot_ngram_counts(df):
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x="freq", y="word", hue="ngrams", dodge=False, ax=ax,
                data=df.groupby('ngrams')["ngrams","freq","word"])
    ax.set(xlabel=None, ylabel=None, title="Most frequent words")
    ax.grid(axis="x")
    plt.show()


plot_ngram_counts()

docs = (nlp(line) for line in text.split('\n') if not line.lstrip()[:2] == '==')
sentences = []
for d in docs:
    sentences += [s for s in d.sents]


def tag_sentence(sentence, tags='pos_ dep_'.split()):
    if isinstance(sentence, str):
        sentence = nlp(sentence)
    tokens = [t for t in sentence]
    if not tags or isinstance(tags, str):
        if tags in ('all', '*', '', None):
            tags = [
                label for label in dir(tokens[0]) 
                if (label.endswith('_') and not label[0] =='_') or label == label.strip('_')
                ]
        else:
            tags = tags.split()
    tags = ['text'] + list(tags)
    return pd.DataFrame(
        [[getattr(tok, tag) for tag in tags] for tok in tokens],
        columns=tags)



html = displacy.render(sentences[3], style="dep", page=True)
open('sent.html', 'w').write(html)
print('!firefox sent.html')

sent = sentences[3]
for entity in sent.ents:
    print(entity.text, f"({entity.label_})")


html = displacy.render(sent, style="ent", page=True)
open('sent_ents.html', 'wt').write(html)
# !firefox sent_ents.html



def extract_entities_manually(doc):
    a, b, prev_dep, prev_txt, prefix, modifier = "", "", "", "", "", ""
    for token in doc:
        if token.dep_ != "punct":
            ## prexif --> prev_compound + compound
            if token.dep_ == "compound":
                prefix = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
            
            ## modifier --> prev_compound + %mod
            if token.dep_.endswith("mod") == True:
                modifier = prev_txt +" "+ token.text if prev_dep == "compound" else token.text
            
            ## subject --> modifier + prefix + %subj
            if token.dep_.find("subj") == True:
                a = modifier +" "+ prefix + " "+ token.text
                prefix, modifier, prev_dep, prev_txt = "", "", "", ""
            
            ## if object --> modifier + prefix + %obj
            if token.dep_.find("obj") == True:
                b = modifier +" "+ prefix +" "+ token.text
            
            prev_dep, prev_txt = token.dep_, token.text

    a = " ".join([i for i in a.split()])
    b = " ".join([i for i in b.split()])
    return (a.strip(), b.strip())


entities_manual = [extract_entities_manually(d) for d in docs]
entities_manual


def extract_relation_manually(doc, nlp):
    matcher = spacy.matcher.Matcher(nlp.vocab)
    pattern = [{'DEP':'ROOT'}, 
          {'DEP': 'prep', 'OP':"?"},
          {'DEP': 'agent', 'OP':"?"},
          {'POS': 'ADJ', 'OP':"?"}] 
    matcher.add(key="root_prep_agent_adj", patterns=[pattern]) 
    matches = matcher(doc)
    k = len(matches) - 1
    span = doc[matches[k][1]:matches[k][2]] 
    return span.text


relations_manual = [extract_relation_manually(doc=d, nlp=nlp) for d in lst_docs]
relations_manual


def extract_entities_spacy(docs, label=None):
    ents = []
    for i, d in enumerate(docs):
        for e in d.ents:
            if not label or e.label_ == label:
                ents += [[i, e.text, e.label]]
    return pd.DataFrame(ents, columns='sentence entity label'.split())


entities_spacy = extract_entities_spacy(docs, "DATE")
entities_spacy


def extract_triples(sentences):
    """ Extract [subject verb object] triples (subject-predicate-object) using textacy"""
    triples = []
    for i, sent in enumerate(sentences):
        sent_triples = textacy.extract.subject_verb_object_triples(sent)  
        for t in sent_triples:
            triples.append(t._asdict())

    return pd.DataFrame(triples)


df_triples = extract_triples(docs=sentences)


def extract_date_attrs(docs=docs, name="DATE"):
    ## extract attributes

    dic = {"id":[], "text":[], name:[]}

    for n,sentence in enumerate(docs):
        lst = list(textacy.extract.entities(sentence, include_types={name}))
        if len(lst) > 0:
            for attr in lst:
                dic["id"].append(n)
                dic["text"].append(sentence.text)
                dic[name].append(str(attr))
        else:
            dic["id"].append(n)
            dic["text"].append(sentence.text)
            dic[name].append('')

    dtf_att = pd.DataFrame(dic)
    dtf_att = dtf_att[dtf_att[name].str.len() > 0]
    return dtf_att

dtf_att = extract_date_attrs()
dtf_att[dtf_att["id"]==3]


def create_subgraph(dtf, entity='Cory Efram Doctorow'):
    entity = entity.strip().replace(' ', '_').lower() if entity else None
    if not entity or str(entity).lower().strip() == 'all':
        entity = None
    if entity:
        tmp = dtf[(dtf["entity"].str.strip().str.lower()==entity) | (dtf["object"].str.strip().str.lower()==entity)]
    else:
        tmp = dtf[(dtf["entity"].str.len() > 0) | (dtf["object"].str.len() > 0)]

    G = nx.from_pandas_edgelist(
        tmp, source="entity", target="object", 
        edge_attr="relation", 
        create_using=nx.DiGraph())
    return G

G = create_subgraph(dtf_att, entity=None)


def plot_subgraph(G, show=True, entity=None):
    """ FIXME
    File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/pygraphviz/agraph.py:336, in AGraph.add_node(self, n, **attr)
    334 except KeyError:
    335     nh = gv.agnode(self.handle, n, _Action.create)
    --> 336 node = Node(self, nh=nh)
        337 node.attr.update(**attr)

    File ~/code/tangibleai/nlpia2/.venv/lib/python3.9/site-packages/pygraphviz/agraph.py:1857, in Node.__new__(self, graph, name, nh)
       1855 def __new__(self, graph, name=None, nh=None):
       1856     if nh is not None:
    -> 1857         n = super().__new__(self, gv.agnameof(nh), graph.encoding)
       1858     else:
       1859         n = super().__new__(self, name)

    TypeError: decoding to str: need a bytes-like object, NoneType found
"""
    fig = plt.figure(figsize=(15,10))

    positions = nx.spring_layout(G, k=1)
    positions = nx.nx_agraph.graphviz_layout(G, prog="neato")

    node_color = ["red" if node==entity else "skyblue" for node in G.nodes]
    edge_color = ["red" if edge[0]==entity else "black" for edge in G.edges]

    nx.draw(G, pos=positions, with_labels=True, node_color=node_color, 
            edge_color=edge_color, cmap=plt.cm.Dark2, 
            node_size=2000, node_shape="o", connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edge_labels(G, pos=positions, label_pos=0.5, 
                            edge_labels=nx.get_edge_attributes(G,'relation'),
                            font_size=12, font_color='black', alpha=0.6)
    if show:
        plt.show()

    return fig

fig = plot_subgraph(G)
