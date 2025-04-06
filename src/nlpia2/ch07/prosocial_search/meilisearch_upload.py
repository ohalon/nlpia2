import json
import os
from pathlib import Path

import meilisearch
import spacy
from nlpia2.env import get_env


spacy_en = spacy.load("en_core_web_md")
ENV = get_env()

MEILI_MASTER_KEY = ENV.get('MEILI_MASTER_KEY', os.environ['MEILI_MASTER_KEY'])
MEILI_URL = 'https://search.qary.ai'


def get_api_keys(url=MEILI_URL, master_key=MEILI_MASTER_KEY):
    client = meilisearch.Client(f'{url}', f'{master_key}')
    response = client.get_keys()
    return [k['key'] for k in response['results']]


API_KEYS = get_api_keys()
MEILI_API_KEY = API_KEYS[-1]


def get_meili_client(url=MEILI_URL, api_key=MEILI_API_KEY):
    return meilisearch.Client(f"{url}", api_key)


CLIENT = get_meili_client()


def get_indexes():
    return dict([(i.uid, i) for i in CLIENT.get_indexes()])


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


def upload_json(filepath='movies.json',
                api_key=MEILI_API_KEY,
                host=ENV.get('MEILI_URL', MEILI_URL),
                index_name=None,
                ):
    """ upload a json file to a new index on melisearch (search.qary.ai)

    curl -H "X-Meili-API-Key: $MEILI_API_KEY" -X POST $MEILI_URL'/indexes/movies/documents' --data @movies.json
    """

    json_file = open(filepath)
    data = json.load(json_file)
    if index_name is None:
        index_name = Path(filepath).with_suffix('').name
    print(f'index_name={index_name}')
    CLIENT.index(index_name).add_documents(data)


def upload_dicts(list_of_dicts, index_name='default'):
    CLIENT.index(index_name).add_documents(list_of_dicts)
