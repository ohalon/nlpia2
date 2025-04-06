# pip install paperswithcode-client
import sys
import logging
from pathlib import Path

from paperswithcode import PapersWithCodeClient
from tqdm import tqdm
import pandas as pd

from nlpia2.constants import SRC_DATA_DIR

log = logging.getLogger(__name__)

DATA_DIR = SRC_DATA_DIR
DEFAULT_NAME = 'tasks'


def auto_csv_path(name, *args):
    fname = f'pwc_{name}'
    if len(args):
        csv_path = DATA_DIR / f'{fname}{"_".join(args)}.csv'
    csv_path = DATA_DIR / f'{fname}.csv'
    return csv_path


def get_df(name, *args, client=None, csv_path=None):
    # client = client or getattr(get_df, 'client', None) or PapersWithCodeClient()
    client = PapersWithCodeClient()
    get_df.client = client
    list_getter = None
    names = f'{name}_list {name[:-1]}_list {name[:-2]}_list {name}'.split()
    for n in names:
        list_getter = getattr(client, n, None)
        if list_getter:
            name = n
            break
    log.debug(f'Retrieving {name}(*{args}) from paperswithcode.com/api/')
    if not list_getter:
        raise ValueError(f"No endpoint named {'|'.join(names)} found!")
    page = 1
    items_per_page = 100
    response = list_getter(page=page, items_per_page=items_per_page)
    total_count = response.count
    num_pages = total_count // items_per_page + 1
    log.debug(f'num_pages: {num_pages}, total_count: {total_count}, items_per_page: {items_per_page}')
    log.debug(f'page: {page}, next_page: {response.next_page}')

    all_results = []
    for page in tqdm(range(1, num_pages + 1)):
        all_results += [r.dict() for r in response.results]
        num_pages = total_count // items_per_page + 1
        log.debug(f'num_pages: {num_pages}, total_count: {total_count}, items_per_page: {items_per_page}')
        log.debug(f'page: {page}, next_page: {response.next_page}, '
                  + f'count: {response.count}, len(all_results): {len(all_results)}, '
                  + f"len(response.results): {len(response.results)}")
        if not response.next_page:
            break
        page += 1
        response = list_getter(page=response.next_page, items_per_page=items_per_page)
    df = pd.DataFrame(all_results)
    df.index.name = name + '_id'
    if total_count != len(df):
        log.error(f'Only retrieved {len(df)} rows, should be {total_count}')
    if csv_path == '':
        csv_path = DATA_DIR
    if csv_path:
        path = Path(csv_path)
        if path.is_dir():
            csv_path = auto_csv_path(name, *args)
        df.to_csv(csv_path, index=0)
    return df


if __name__ == '__main__':
    name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_NAME
    args = list(sys.argv[2:]) if len(sys.argv) > 2 else []
    if len(args):
        csv_path = DATA_DIR / f'{name}{"_".join(args)}.csv'
    globals()[f'df_{name}'] = df = get_df(name, *args, csv_path=csv_path)
