""" Breadth first exploration (crawling) of a graph of web pages (Wikipedia Articles) """
import logging
import time
from collections import abc

from tqdm import tqdm
import wikipedia as wiki

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def get_page(title):
    try:
        return wiki.page(title, auto_suggest=False)
    except (wiki.DisambiguationError, wiki.PageError) as e:
        log.warning(f'auto_suggest=False: {e}')
    try:
        return wiki.page(title, auto_suggest=True)
    except (wiki.DisambiguationError, wiki.PageError) as e:
        log.warning(f'auto_suggest=True: {e}')
    return False


def walk_wikipedia(pages, depth=1, delay=0.1):
    depth_goal = depth
    depth = 0
    if isinstance(pages, str):
        pages = {t: None for t in pages.split(',')}
    if not isinstance(pages, abc.Mapping):
        pages = {t: None for t in pages.split(',')}
    queue = list(pages.keys())
    while depth < depth_goal and len(queue):
        log.info(f"depth={depth}, len(nextqueue)={len(queue)}, queue[0]={queue[0]}")
        pages_this_level = len(queue)
        for i in tqdm(range(pages_this_level)):
            title = queue.pop(0)
            page_tuple = pages.get(title)
            # if title from front of queue has not already been retrieved, then retrieve it
            if page_tuple is None:  # if False then retrieval has been attempted and failed
                time.sleep(delay)
                page = get_page(title)
                log.debug(page)
                # FIXME: add depth_id (depth) and link_id (i) integers as attributes to page object (no tuple for values)
                pages[title] = (depth, i, page)
                if page:  # page is False when get_page() failed to find a valid page
                    queue.extend(list(page.links))
            elif page_tuple and page_tuple[-1]:
                queue.extend(list(page_tuple[-1].links))
        depth += 1
    return pages
