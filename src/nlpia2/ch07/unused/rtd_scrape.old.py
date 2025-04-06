import requests
# ! pip instal html5lib_to_markdown
# from html5lib_to_markdown.transformer import to_markdown  # , Transformer

resp = requests.get('https://readthedocs.org/search/?q=javascript')
# bhtml = resp.content
# text = resp.text

from bs4 import BeautifulSoup
soup = BeautifulSoup(resp.content)
text = soup.get_text()


from markdownify import markdownify as html2md

md = html2md(resp.text)
