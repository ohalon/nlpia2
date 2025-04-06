from nlpia2 import wikipedia as wiki

page = wiki.page('Timnit Gebru')

text = page.content

text[:66]

i1 = text.index('Stochastic')

text[i1:i1+51]

from nlpia2.spacy_pipes import nlp_df

nlpsm = load('en_core_web_sm')

df = nlp_df(text, nlp=nlpsm)

from nlpia2.spacy_pipes import nlp_df

nlp = load('en_core_web_lg'

df = nlp_df(text, nlp=nlp)

df

i0 = text.index('Gebru had')

text[i0:i0+171]

import spacy, coreferee

nlptrf = spacy.load('en_coreference_web_trf')

gebru_she = text[i0:i1]

text_gebru = text[i0:i1]

doc_gebru = nlp(text_gebru)

doc_gebru

doc_gebru.spans

doc = nlp(text)

doc.ents[:6]  # <1>

from nlpia2.spacy_pipes import nlp_df, load

df.loc['On':'?']

tags = []

for tok in doc:
    tags.append(dict(token=tok.text, pos=tok.pos_, dep=tok.dep_))
    tags[-1].update({f'child{i}': c.text for (i, c) in enumerate(tok.children)})

df = pd.DataFrame(tags).set_index('token').fillna('')

df.head()

df.tail(11)

doc.ents

doc

from nlpia2.spacy_language_model import nlp

import pandas as pd

text = "Gebru was unethically fired from her Ethical AI team."

doc = nlp(text)

tags = []
for tok in doc:
    tags.append(dict(text=tok.text, pos=tok.pos_, dep=tok.dep_))
    tags[-1].update({f'child_{i}': c.text for (i, c) in enumerate(tok.children)})

df =

df = pd.DataFrame(tags)

df

import pandas as pd

pd.options.display.max_colwidth = 20

from nlpia2.nell import read_nell_tsv, simplify_names

df = read_nell_tsv(nrows=1000)

df[df.columns[:4]].head()

pd.options.display.max_colwidth = 40

df['entity'].str.split(':').str[1:].str.join(':')

df['entity'].str.split(':').str[-1]

df = simplify_names(df)  # <1>

df[df.columns[[0, 1, 2, 4]]].head()

islatlon = df['relation'] == 'latlon'

df[islatlon].head()

import spacy

nlp = spacy.load("en_core_web_sm")

sentence = "We will be learning NLP today!"

print ("{:<15} | {:<8} | {:<15} | {:<30} | {:<20}".format('Token','Relation','Head', 'Children', 'Meaning'))

print ("-" * 115)

for token in doc:
    # Print the token, dependency nature, head, all dependents of the token, and meaning of the dependency
    print ("{:<15} | {:<8} | {:<15} | {:<30} | {:<20}"
            .format(str(token.text), str(token.dep_), str(token.head.text), str([child for child in token.children]) , str(spacy.explain(token.dep_))[:17] ))

import benepar

benepar.download('benepar_en3')

import spacy

nlp = spacy.load("en_core_web_md")

if spacy.__version__.startswith('2'):
    nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
else:
    nlp.add_pipe("benepar", config={"model": "benepar_en3"})

doc = nlp("Johnson was compelled to ask the EU for an extension of the deadline, which was granted")

sent = list(doc.sents)[0]

print(sent._.parse_string)

import spacy

nlp = spacy.load("en_core_web_md")

sent = "John Smith works at Tangible AI"

doc = nlp(sent)

entities = []

for ent in doc.ents:
    sent = sent.replace(ent.text, "^/" + ent.label_ + "/" + ent.text + "^")

print(sent)

import spacy

nlp = spacy.load('en_core_web_md')

import neuralcoref

neuralcoref.add_to_pipe(nlp)

doc = nlp(u'My sister has a dog. She loves him.')

doc._.coref_clusters

from allennlp.predictors.predictor import Predictor

import allennlp_models.tagging

predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

predictor.predict(

def find_greeting(s):
    """ Return greeting str (Hi, etc) if greeting pattern matches """
    if s[0] == 'H':
        if s[:3] in ['Hi', 'Hi ', 'Hi,', 'Hi!']:
            return s[:2]
        elif s[:6] in ['Hello', 'Hello ', 'Hello,', 'Hello!']:
            return s[:5]
    elif s[0] == 'Y':
        if s[1] == 'o' and s[:3] in ['Yo', 'Yo,', 'Yo ', 'Yo!']:
            return s[:2]
    return None

find_greeting('Hi Mr. Turing!')

find_greeting('Hello, Rosa.')

find_greeting("Yo, what's up?")

find_greeting("Hello")

print(find_greeting("hello"))

print(find_greeting("HelloWorld"))

import re

lat = r'([-]?[0-9]?[0-9][.][0-9]{2,10})'

lon = r'([-]?1?[0-9]?[0-9][.][0-9]{2,10})'

sep = r'[,/ ]{1,3}'

re_gps = re.compile(lat + sep + lon)

re_gps.findall('http://...maps/@34.0551066,-118.2496763...')

re_gps.findall("https://www.openstreetmap.org/#map=10/5.9666/116.0566")

re_gps.findall("Zig Zag Cafe is at 45.344, -121.9431 on my GPS.")

us = r'((([01]?\d)[-/]([0123]?\d))([-/]([0123]\d)\d\d)?)'

mdy = re.findall(us, 'Santa came 12/25/2017. An elf appeared 12/12.')

mdy

dates = [{'mdy': x[0], 'my': x[1], 'm': int(x[2]), 'd': int(x[3]),
    'y': int(x[4].lstrip('/') or 0), 'c': int(x[5] or 0)} for x in mdy]

dates

for i, d in enumerate(dates):
    for k, v in d.items():
        if not v:
            d[k] = dates[max(i - 1, 0)][k]  # <1>

dates

from datetime import date

datetimes = [date(d['y'], d['m'], d['d']) for d in dates]

datetimes

eu = r'((([0123]?\d)[-/]([01]?\d))([-/]([0123]\d)?\d\d)?)'

dmy = re.findall(eu, 'Alan Mathison Turing OBE FRS (23/6/1912-7/6/1954) \
    was an English computer scientist.')

dmy

dmy = re.findall(eu, 'Alan Mathison Turing OBE FRS (23/6/12-7/6/54) \
    was an English computer scientist.')

dmy

yr_19xx = (
    r'\b(?P<yr_19xx>' +
    '|'.join('{}'.format(i) for i in range(30, 100)) +
    r')\b'
    )  # <1>

yr_20xx = (
    r'\b(?P<yr_20xx>' +
    '|'.join('{:02d}'.format(i) for i in range(10)) + '|' +
    '|'.join('{}'.format(i) for i in range(10, 30)) +
    r')\b'
    )  # <2>

yr_cent = r'\b(?P<yr_cent>' + '|'.join(
    '{}'.format(i) for i in range(1, 40)) + r')'  # <3>

yr_ccxx = r'(?P<yr_ccxx>' + '|'.join(
    '{:02d}'.format(i) for i in range(0, 100)) + r')\b'  # <4>

yr_xxxx = r'\b(?P<yr_xxxx>(' + yr_cent + ')(' + yr_ccxx + r'))\b'

yr = (
    r'\b(?P<yr>' +
    yr_19xx + '|' + yr_20xx + '|' + yr_xxxx +
    r')\b'
    )

groups = list(re.finditer(
    yr, "0, 2000, 01, '08, 99, 1984, 2030/1970 85 47 `66"))

full_years = [g['yr'] for g in groups]

full_years

mon_words = 'January February March April May June July ' \
    'August September October November December'

mon = (r'\b(' + '|'.join('{}|{}|{}|{}|{:02d}'.format(
    m, m[:4], m[:3], i + 1, i + 1) for i, m in enumerate(mon_words.split())) +
    r')\b')

re.findall(mon, 'January has 31 days, February the 2nd month of 12, has 28, except in a Leap Year.')

day = r'|'.join('{:02d}|{}'.format(i, i) for i in range(1, 32))

eu = (r'\b(' + day + r')\b[-,/ ]{0,2}\b(' +
    mon + r')\b[-,/ ]{0,2}\b(' + yr.replace('<yr', '<eu_yr') + r')\b')

us = (r'\b(' + mon + r')\b[-,/ ]{0,2}\b(' +
    day + r')\b[-,/ ]{0,2}\b(' + yr.replace('<yr', '<us_yr') + r')\b')

date_pattern = r'\b(' + eu + '|' + us + r')\b'

list(re.finditer(date_pattern, '31 Oct, 1970 25/12/2017'))

import datetime

dates = []

for g in groups:
    month_num = (g['us_mon'] or g['eu_mon']).strip()
    try:
        month_num = int(month_num)
    except ValueError:
        month_num = [w[:len(month_num)]
            for w in mon_words].index(month_num) + 1
    date = datetime.date(
        int(g['us_yr'] or g['eu_yr']),
        month_num,
        int(g['us_day'] or g['eu_day']))
    dates.append(date)

dates

import spacy

en_model = spacy.load('en_core_web_md')

sentence = ("In 1541 Desoto wrote in his journal that the Pascagoula people " +
    "ranged as far north as the confluence of the Leaf and Chickasawhay rivers at 30.4, -88.5.")

parsed_sent = en_model(sentence)

parsed_sent.ents

' '.join(['{}_{}'.format(tok, tok.tag_) for tok in parsed_sent])

from spacy.displacy import render

sentence = "In 1541 Desoto wrote in his journal about the Pascagoula."

parsed_sent = en_model(sentence)

with open('pascagoula.html', 'w') as f:
    f.write(render(docs=parsed_sent, page=True, options=dict(compact=True)))

import pandas as pd

from collections import OrderedDict

def token_dict(token):
    return OrderedDict(ORTH=token.orth_, LEMMA=token.lemma_,
        POS=token.pos_, TAG=token.tag_, DEP=token.dep_)

def doc_dataframe(doc):
    return pd.DataFrame([token_dict(tok) for tok in doc])

doc_dataframe(en_model("In 1541 Desoto met the Pascagoula."))

pattern = [{'TAG': 'NNP', 'OP': '+'}, {'IS_ALPHA': True, 'OP': '*'},
           {'LEMMA': 'meet'},
           {'IS_ALPHA': True, 'OP': '*'}, {'TAG': 'NNP', 'OP': '+'}]

from spacy.matcher import Matcher

doc = en_model("In 1541 Desoto met the Pascagoula.")

matcher = Matcher(en_model.vocab)

matcher.add('met', None, pattern)

m = matcher(doc)

m

doc[m[0][1]:m[0][2]]

doc = en_model("October 24: Lewis and Clark met their" \
               "first Mandan Chief, Big White.")

m = matcher(doc)[0]

m

doc[m[1]:m[2]]

doc = en_model("On 11 October 1986, Gorbachev and Reagan met at Höfði house")

matcher(doc)

doc = en_model("On 11 October 1986, Gorbachev and Reagan met at Hofoi house")

pattern = [{'TAG': 'NNP', 'OP': '+'}, {'LEMMA': 'and'},
           {'TAG': 'NNP', 'OP': '+'},
           {'IS_ALPHA': True, 'OP': '*'}, {'LEMMA': 'meet'}]

matcher.add('met', None, pattern)  # <1>

m = matcher(doc)

m

doc[m[-1][1]:m[-1][2]]  # <3>

re.split(r'[!.?]+[ $]', "Hello World.... Are you there?!?! I'm going to Mars!")

re.split(r'[!.?] ', "The author wrote \"'I don't think it's conscious.' Turing said.\"")

re.split(r'[!.?] ', "The author wrote \"'I don't think it's conscious.' Turing said.\" But I stopped reading.")

re.split(r'(?<!\d)\.|\.(?!\d)', "I went to GT.You?")

from nlpia.data.loaders import get_data

regex = re.compile(r'((?<!\d)\.|\.(?!\d))|([!.?]+)[ $]+')

examples = get_data('sentences-tm-town')

wrong = []

for i, (challenge, text, sents) in enumerate(examples):
    if tuple(regex.split(text)) != tuple(sents):
        print('wrong {}: {}{}'.format(i, text[:50], '...' if len(text) > 50 else ''))
        wrong += [i]

len(wrong), len(examples)
