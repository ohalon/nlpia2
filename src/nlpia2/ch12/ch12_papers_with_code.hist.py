# pip install paperswithcode-client
from paperswithcode import PapersWithCodeClient
from tqdm import tqdm
import pandas as pd

import logging
log = logging.getLogger(__name__)


def get_df(name, client=None, max_pages=1000):
    client = client or getattr(get_df, 'client', None) or PapersWithCodeClient()
    get_df.client = client
    list_getter = None
    names = f'{name}_list {name[:-1]}_list {name[:-2]}_list {name}'.split()
    for n in names:
        list_getter = getattr(client, f'{name[:-1]}_list', None)
        if list_getter:
            break
    if not list_getter:
        raise ValueError(f"No endpoint named {'|'.join(names)} found!")
    page = 1
    items_per_page = 100
    response = list_getter(page=page, items_per_page=items_per_page)
    total_count = response.count
    num_pages = total_count // items_per_page + 1
    print(f'num_pages: {num_pages}, total_count: {total_count}, items_per_page: {items_per_page}')
    print(f'page: {page}, next_page: {response.next_page}')

    results = []
    for page in tqdm(range(1, num_pages + 1)):
        results += [r.dict() for r in response.results]
        num_pages = total_count // items_per_page + 1
        log.debug(f'num_pages: {num_pages}, total_count: {total_count}, items_per_page: {items_per_page}')
        log.debug(f'page: {page}, next_page: {response.next_page}')
        print(f'page: {page}, next_page: {response.next_page}, '
              + f'count: {response.count} len(results): {len(results)}')
        if not response.next_page:
            break
        page += 1
        response = list_getter(page=response.next_page)
    df = pd.DataFrame(results)
    if count != len(df):
        log.warning(f'Only retrieved {len(df)} rows, should be {count}')
    # df.to_csv(f'~/code/tangibleai/community/nlpia2/src/nlpia2/data/pwc/{name}_list.csv', index=0)
    return df
