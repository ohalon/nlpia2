# spacy_pipes.py
from nlpia2.spacy_language_model import load
import pandas as pd
# from spacy import Doc, Token
from copy import deepcopy


DEFAULT_COLUMNS = {
    'text': 'Token',
    'pos_': 'POS',
    'dep_': 'Dependent',
    'ent_type_': 'Entity',
    'ent_kb_id_': 'KB_ID',
    }

DETAILED_COLUMNS = {
    'ent_iob_': 'EntityPosition', 
    'is_oov': 'OOV',
    }


def nlp_df(text, nlp=None, add_columns=None, remove_columns=None,
           columns=DEFAULT_COLUMNS, index_col=None):
    """ Create DataFrame of token (rows) attributes (columns) using spacy, nlp pipeline """
    if isinstance(nlp, (type(None), str)):
        nlp = nlp_df.nlp
    if nlp_df.nlp is None or isinstance(nlp_df.nlp, str):
        nlp_df.nlp = nlp = load(nlp_df.nlp)
    if isinstance(text, str):
        doc = nlp(text)
    else:
        doc = text
    columns = dict(deepcopy(columns))
    # FIXME: allow columns, add_columns, and remove_columns to be lists/tuples (identity mapping)
    if add_columns is not None:
        columns.update(dict(add_columns))
    if remove_columns is not None:
        for c in remove_columns:
            del columns[c]

    table = []
    for t in doc:
        d = dict()
        for attr, name in columns.items():
            d[name] = getattr(t, attr, None)
        table.append(d)

    df = pd.DataFrame(table)
    if index_col is not None:
        if isinstance(index_col, int):
            index_col = df.columns[index_col]
        df = df.set_index(index_col)
    return df


nlp_df.nlp = None


if __name__ == '__main__':
    lines = [
        'Gebru had determined that publishing research papers was more effective at bringing forth the ethical change she was focused on than pressing her superiors in the company.',
        'She and five others coauthored a research paper: "On the Dangers of Stochastic Parrots: Can Language Models Be Too Big?"'
        ]
    text = lines[0]
    nlp_coref = load('en_coreference_web_trf')
    doc_coref = nlp_coref(text)
    print(doc_coref.spans)
    
    nlp = load('en_core_web_lg')
    doc = nlp(text)

    print(nlp_df(text, nlp))
    print(list(doc.sents))