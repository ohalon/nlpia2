import spacy
import neuralcoref

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)
doc1 = nlp('My sister has a dog. She loves him.')
print(doc1._.coref_clusters)

doc2 = nlp('Angela lives in Boston. She is quite happy in that city.')
for ent in doc2.ents:
    print(ent._.coref_cluster)




import coreferee, spacy
nlp = spacy.load('en_core_web_trf')
nlp.add_pipe('coreferee')
doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
doc._.coref_chains.print()
nlp.add_pipe('coreferee')
doc = nlp('Although he was very busy with his work, Peter had had enough of it. He and his wife decided they needed a holiday. They travelled to Spain because they loved the country very much.')
doc._.coref_chains
doc._.coref_chains
doc._.coref_chains?
dir(doc._.coref_chains)
doc._.coref_chains.pretty_representation()
doc._.coref_chains.pretty_representation
doc
doc._.coref_chains.print()
doc[1]
doc[1]._.coref_chains.print()

i = 0
headers = []
sents = []
for sent in doc.sents:
    headers.append('')
    sents.append('')
    for t in sent:
        tok = t.text + t.whitespace_
        idx = str(i)
        if len(idx) >= len(tok):
            idx = ''
        else:
            idx += ' ' * (len(tok) - len(idx))
        headers[-1] += idx
        sents[-1] += tok
        i += 1
    print(headers[-1])
    print(sents[-1])
    print()


