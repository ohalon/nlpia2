import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/master'
FAQ_DIR = 'src/qary/data/faq'
FAQ_FILENAME = 'short-faqs.csv'
DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])

df = pd.read_csv(DS_FAQ_URL)

vectorizer = TfidfVectorizer()
vectorizer.fit(df['question'])
# vectorize all the questions/answers in qa_dataset
tfidfvectors_sparse = vectorizer.transform(df['question'])
tfidfvectors = tfidfvectors_sparse.todense()

question = "What's overfitting a model?"
question_vector = vectorizer.transform([question]).todense()
idx = question_vector.dot(tfidfvectors.T).argmax()
# 51

print(
    f"Your question:\n  {question}\n\n"
    f"Most similar FAQ question:\n  {df['question'][idx]}\n\n"
    f"Answer to that FAQ question:\n  {df['answer'][idx]}\n\n"
)
# Your question:
#   What's overfitting a model?

# Most similar FAQ question:
#   What is overfitting?

# Answer to that FAQ question:
#   When your test set accuracy is significantly lower than your training set accuracy?


question = 'What causes a model to overfit?'
question_vector = vectorizer.transform([question]).todense()
idx = question_vector.dot(tfidfvectors.T).argmax()
# 35
print(
    f"Your question:\n  {question}\n\n"
    f"Most similar FAQ question:\n  {df['question'][idx]}\n\n"
    f"Answer to that FAQ question:\n  {df['answer'][idx]}\n\n"
)
# Your question:
#   What causes a model to overfit?

# Most similar FAQ question:
#   Is a `LogisticRegression` more likely or less likely to overfit to your data when compared to a `DecisionTree`?

# Answer to that FAQ question:
#   A `LogisticRegression` will be less likely to overfit than a `DecisionTree` in most situations.

# To improve recall, a stemmer should be added to the TfidfVectorizer


def tokenizer(text, nlp=None):
    """
    >>> spacy.cli.download('en_core_web_sm')
    >>> nlp = spacy.load('en_core_web_sm')
    >>> tokenizer("What causes overfit?", nlp)
    ['what', 'causes', 'overfit', '?']
    >>> tokenizer("What causes overfitting?", nlp)
    ['what', 'cause', 'overfitte', '?']

    >>> spacy.cli.download('en_core_web_md')
    >>> nlp = spacy.load('en_core_web_md')
    >>> tokenizer("What causes overfitting?", nlp)
    ['what', 'cause', 'overfitting', '?']
    >>> tokenizer("What causes overfit?", nlp)
    ['what', 'cause', 'overfit', '?']
    >>> tokenizer("What causes running?", nlp)
    ['what', 'causes', 'run', '?']

    Lemmatization will handle causes/cause, but not hitting/hit,
    even if you use the large model.

    >>> # spacy.cli.download('en_core_web_lg')
    >>> nlp = spacy.load('en_core_web_lg')
    >>> tokenizer("What causes overhitting?", nlp=nlp)
    ['what', 'cause', 'overhitting', '?']
    # >>> spacy.cli.download('en_core_web_md')
    >>> tokenizer('What causes overfitting?', nlp=nlp)
    ['what', 'cause', 'overfitting', '?']
    >>> tokenizer('What causes over-fitting?', nlp=nlp)
    ['what', 'cause', 'over', '-', 'fitting', '?']
    >>> tokenizer('What causes over fitting?', nlp=nlp)
    ['what', 'cause', 'over', 'fitting', '?']
    >>> tokenizer('What causes over hitting?', nlp=nlp)
    ['what', 'cause', 'over', 'hit', '?']
    >>> tokenizer('What causes over-hitting?', nlp=nlp)
    ['what', 'cause', 'over', '-', 'hit', '?']
    >>> tokenizer('What causes overhitting?', nlp=nlp)
    ['what', 'cause', 'overhitting', '?']
    """
    return [text.lemma_ for text in nlp(txt)]
