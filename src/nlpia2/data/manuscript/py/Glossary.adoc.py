from nlpia.book_parser import write_glossary

from nlpia.constants import DATA_PATH

print(write_glossary(
    os.path.join(DATA_PATH, 'book')))  # <1>
