"""
If you add unimportant (frequent) words,
 the tf-idf vector search will still find the right text:

>>> question = 'What is overfitting a model?'
>>> question_vector = vectorizer.transform([question]).todense()
>>> np.array(question_vector).reshape(1,-1).dot(df_vecs.values.T).argmax()
51
>>> df['question'][51]
'What is overfitting?'

If you change unimportant (frequent) words,
 the tf-idf vector search will still find the right text:

>>> question = 'Who is overfitting a model?'
>>> question_vector = vectorizer.transform([question]).todense()
>>> question_vector.dot(df_vecs.values.T).argmax()
51

If you delete unimportant (frequent) words,
 the tf-idf vector search will still find the right text:

>>> question = 'overfitting'
>>> question_vector = vectorizer.transform([question]).todense()
>>> question_vector.dot(df_vecs.values.T).argmax()
51

>>> question = 'What is overfitting a model?'
>>> question_vector = vectorizer.transform([question]).todense()
>>> question_vector.dot(df_vecs.values.T).argmax()
51

If you change an important (infrequent) word, such as "overfit"
 it will find other questions that use that form of the word:

>>> question = 'What causes a model to overfit?'
>>> question_vector = vectorizer.transform([question]).todense()
>>> question_vector.dot(df_vecs.values.T).argmax()
35
>>> df['question'][35]
'Is a `LogisticRegression` more likely or less likely to overfit '
' to your data when compared to a `DecisionTree`?'

What was in this question_vector to cause it to match this FAQ question?

>>> question_vector[question_vector > 0]
matrix([[0.49665284, 0.73814899, 0.36844366, 0.26966885]])
>>> question_vector = pd.Series(np.array(question_vector)[0],
...     index=vectorizer.get_feature_names())
>>> question_vector[question_vector > 0]
model      0.496653
overfit    0.738149
to         0.368444
what       0.269669

>>> %run ch03_bot_sparse.py "how are you doing?"
Your question:
  how are you doing?
Most similar FAQ:
  What is a good package for doing topic analysis or natural language analytics?
Answer to that FAQ:
  The ldaviz package has good visualizations in addition to the Latent Dirichlet Allocation topic vector analysis.
"""
import sys
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# https://gitlab.com/tangibleai/qary/-/raw/main/src/qary/data/faq/short-faqs.csv
REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/main'
FAQ_DIR = 'src/qary/data/faq'
FAQ_FILENAME = 'short-faqs.csv'
DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])


class FAQ():
    def __init__(self, df, **kwargs):
        self.df = df
        self.vectorizer = TfidfVectorizer(**kwargs)
        self.vectorizer.fit(self.df['question'])
        self.q_vectors = self.vectorizer.transform(self.df['question'])
        # self.q_vectors = pd.DataFrame.sparse.from_spmatrix(
        #    self.q_vectors,
        #    columns=vectorizer.get_feature_names())
        # self.df = pd.concat([df, self.q_vectors], axis=1)

    def find_question_index(self, question):
        q_vector = self.vectorizer.transform([question])
        q_similarities = q_vector.dot(self.q_vectors.T)
        return q_similarities.argmax()

    def reply(self, question):
        idx = self.find_question_index(question)
        self.df['question'][idx]
        response = (
            f"Your question:\n  {question}\n"
            f"Most similar FAQ:\n  {self.df['question'][idx]}\n"
            f"Answer to that FAQ:\n  {self.df['answer'][idx]}\n"
        )
        return response


def run_bot(faq=None):
    if faq is None:
        df = pd.read_csv(DS_FAQ_URL)
        faq = FAQ(df)

    # Ask user for questions/queries to look up in the DB
    question = ""
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])

    qa_log = []
    while True:
        if question:
            if question.lower().strip().startswith('exit'):
                break
            answer = faq.reply(question)
            qa_log.append([question, answer])
            print(answer)
        question = input('Ask me anything: ')
    return qa_log


if __name__ == '__main__':
    # Create an FAQ database of questions and answers
    df = pd.read_csv(DS_FAQ_URL)
    faq = FAQ(df)
    qa_log = run_bot(faq)
