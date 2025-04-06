[source, python]
----
>> > from collections import Counter
>> > import spacy
>> > import requests

>> > DATA_DIR = ('https://gitlab.com/tangibleai/nlpia/'
...             '-/raw/master/src/nlpia/data')

>> > url = DATA_DIR + '/bias_intro.txt'
>> > bias_intro = requests.get(url).content.decode()  # <1>
>> > nlp = spacy.load("en_core_web_sm")
>> > tokens = [token.text for token in nlp(bias_intro.lower())]
>> > token_counts = Counter(tokens)
>> > token_counts
Counter({',': 35, 'of': 15, '.': 15, 'to': 15, 'the': 15...
----
< 1 > `requests.get` method returns an objec with a byte string in its content field. We use `decode()` to transform it into a regular string.
