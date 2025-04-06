from tqdm import tqdm
import gzip
import pandas as pd

"""
NELL Knowledge Graph schema:

1. entity: Canonical name of the entity part of the 'entity->relation->value' triple. NOT the literal string of NL seen by NELL in the text
2. relation: The canonical relation name between the entity and value. Category relations are named "generalizations".
3. value: Canonical name of the object or value in the 'entity->relation->value' triple. For category relations, this is the name of the category, otherwise it's an entity (noun phrase).
4. iteration: The point in NELL's life at which this category or relation instance was promoted to one that NELL beleives to be true. This is a non-negative integer indicating the number of iterations of bootstrapping NELL had gone through.
5. prob: A probabilistic confidence score for the belief.
6. source: A summary of the provenance for the belief indicating the set of learning subcomponents (CPL, SEAL, etc.) that had submitted this belief as being potentially true.
7. entities: The set of text strings that NELL has read that it believes can refer to the concept indicated in the Entity column.
8. values: For relations, the set of text strings that NELL has read that it believes can refer to the concept indicated in the Value column. For categories, this should be empty but may contain something spurious.
9. best_entities: Of the entity text strings, which str can best be used to describe the concept.
10. best_values: Of the value text strings, which str can best be used to describe the concept.
11. entity_categories: The full set of categories (which may be empty) to which NELL belives the concept indicated in the Entity column to belong.
12. value_categories: For relations, the full set of categories (which may be empty) to which NELL believes the concept indicated in the Value column to belong. For categories, this should be empty but may contain something spurious.
14. candidate_source: A free-form amalgamation of more specific provenance information describing the justification(s) NELL has for possibly believing this category or relation instance. 
"""

from .constants import BIGDATA_DIR
NELL_DIR = BIGDATA_DIR / 'nell'
NELL_DIR.mkdir(exist_ok=True, parents=True)
NELL_NUM_RELATIONS = 2_766_079
DEFAULT_PATH = NELL_DIR / 'NELL.08m.1115.esv.csv.gz'
DEFAULT_LAYOUT = 'spring'
DEFAULT_TOTAL = 3_000_000  # default number of rows expected


def read_nell_tsv(path=DEFAULT_PATH, total=DEFAULT_TOTAL, header=[0], skiprows=None, nrows=None, **kwargs):
    """ Read 13-column TSV containing facts/knowledge for a NELL triple, return DataFrame

    entity -> relation -> value(object)

    This will sometimes work (slowly, invisibly):    
    df = pd.read_csv(
        'http://rtw.ml.cmu.edu/resources/results/08m/NELL.08m.1115.esv.csv.gz',
        encoding='latin', sep='\t')
    """
    if isinstance(header, (list, tuple)):
        header = max([int(x) for x in header])
    if header is None or header is False or not isinstance(header, int):
        header = -1
    header = int(header) + 1
    lines = []
    total = nrows = min([total, nrows or total])
    with gzip.open(path) as fin:    
        for i, line in enumerate(tqdm(fin, total=total)):
            if i < header:
                continue
            line = line.decode('latin')
            # print(i, len(line.split('\t')))
            lines.append(line.split('\t'))
            if i > nrows:
                break
    return pd.DataFrame(lines[skiprows:nrows],
        columns=('entity relation value iteration prob source entities values '
            'best_entity_str best_value_str entity_categories value_categories '
            'candidate_source').split(), **kwargs)



RELATION_REPLACE = {
    'companyceo': 'CEO',
    'generalizations': 'is_a',
    'haswikipediaurl': 'Wiki_url',
    'acquired': 'acquired',
    'agentcollaborateswithagent': 'collaborates_with',
    'agentcontrols': 'controls',
    'controlledbyagent': 'is_controlled_by',
    'mutualproxyfor': 'mutual_proxy',
    r'organization(\w+)': r'\1',
    r'company(\w+)': r'\1',
    r'\w*alsoknownas': 'AKA',
    'proxyof': 'proxy_of',
    'subpartof': 'part_of',
    'synonymfor': 'synonym_for',
    'worker': 'has_worker',
    'economicsector': 'industry',
    'agentworkedondrug': 'developed_drug',
    'latitudelongitude': 'latlon',
    }


ENTITY_REPLACE = {
    'search_for_common_ground': 'SFCG',
    r'\w+inc': '',  # not good idea (other words end in inc)
    r'\w+corp': '',
    r'\w+llc': '',
    }

VALUE_REPLACE = {
    'en.wikipedia.org/wiki/': '',
    '%20': '_',
    }

def simplify_names(df, columns=None, sep=':', depth=1,
        relation_replace=RELATION_REPLACE,
        entity_replace=ENTITY_REPLACE,
        value_replace=VALUE_REPLACE): 
    """ Simplify the entity, relation, and value names 

    FIXME: split this into simplify_entities/relations/values
    """ 
    entity_replace = {} if not entity_replace else entity_replace
    relation_replace = {} if not relation_replace else relation_replace
    value_replace = {} if not value_replace else value_replace
    # TODO: broke:
    # erv = 'entity relation value'.split()
    # for name in erv:
    #     varname = f'{name}_replace'
    #     if locals()[varname] is None:
    #         locals()[varname] = locals()['entity_replace'] 
    #     elif not locals()[varname]:
    #         locals()[varname] = {}
    columns = list(df.columns[:3]) if columns is None else columns
    columns = columns or []
    for c in columns:
        df[c] = df[c].str.split(sep).apply(lambda x: x[-depth])
    # TODO: use `erv` and locals() to DRY this up 
    for k, v in tqdm(entity_replace.items()):
        c = columns[0]
        df[c] = df[c].str.replace(k, v, regex=True)
    for k, v in tqdm(relation_replace.items()):
        c = columns[1]
        df[c] = df[c].str.replace(k, v, regex=True)
    for k, v in tqdm(value_replace.items()):
        c = columns[2]
        df[c] = df[c].str.replace(k, v, regex=True)
    for c in columns:
        df[c] = df[c].str.strip('/')
        df[c] = df[c].str.split(',').apply(lambda l: ','.join([x.strip().rstrip('0') for x in l]))
        # df[c] = df[c].str.split(';').apply(lambda l: ';'.join([x.strip().rstrip('0') for x in l]))
        # df[c] = df[c].str.split('|').apply(lambda l: '|'.join([x.strip().rstrip('0') for x in l]))
    df['prob'] = df['prob'].astype(float).round(3)
    return df


def count_ents_rels(df=DEFAULT_PATH, nrows=3_000_000):
    if isinstance(df, pd.DataFrame):
        dfall = df
    else:
        dfall = read_nell_tsv(df)
    dfall = simplify_names(dfall)
    rels = dfall['relation'].unique()
    ents = dfall['entity'].unique()
    print(len(ents), len(rels))
    return dict(ents=ents, rels=rels)




if __name__ == '__main__':
    df = read_nell_tsv(total=NELL_NUM_RELATIONS)  # total=2_76X_XXX
