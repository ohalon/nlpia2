import requests
from beautifulsoup4 import BeautifulSoup

sections = "Dialogue,Taglines,Cast,External links".split(',')

url = 'https://en.wikiquote.org/wiki/WarGames'
soup = BeautifulSoup(requests.get(url).content)
in_dialog = False
dialog_lines = []
for line in soup.text.splitlines():
    if line.startswith(sections[0] or in_dialog):
        in_dialog = True
