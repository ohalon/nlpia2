from pathlib import Path

import bs4
import pandas as pd

DATA_DIR = Path('~/.nlpia2-data').expanduser().resolve().absolute()
ADOC_DIR = DATA_DIR / 'manuscript' / 'adoc'
HTML_DIR = DATA_DIR / 'manuscript' / 'html'

ADOC_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)


if __name__ == '__main__':
    paths = list(HTML_DIR.glob('Chapter*.html'))
    lines = []
    for p in paths:
        chapter = bs4.BeautifulSoup(p.open()).get_text()
        chapter = [line if line else ' ' for line in chapter.splitlines()]
        lines.extend(
            list(
                zip(
                    [p.name] * len(chapter),
                    range(len(chapter)),
                    chapter
                )
            )
        )
    df = pd.DataFrame(lines, columns='filename chapter text'.split())
    # df = df.dropna()
    # print(df.head())

    csv_path = DATA_DIR / 'manuscript' / 'manuscript.adoc.html.text.csv'
    df.to_csv(csv_path, index=False)
    df = pd.read_csv(csv_path)
    print(df.head())
    # print(df.sample(10))
    # print(df.tail())
