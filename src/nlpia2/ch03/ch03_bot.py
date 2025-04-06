import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

REPO_URL = 'https://gitlab.com/tangibleai/qary/-/raw/main'
FAQ_DIR = 'src/qary/data/faq'
FAQ_FILENAME = 'short-faqs.csv'
DS_FAQ_URL = '/'.join([REPO_URL, FAQ_DIR, FAQ_FILENAME])

df = pd.read_csv(DS_FAQ_URL)

vectorizer = TfidfVectorizer()
vectorizer.fit(df['question'])
# vectorize all the questions/answers in qa_dataset
tfidfvectors_sparse = vectorizer.transform(df['question'])
tfidfvectors = tfidfvectors_sparse.todense()

question = "How do I decrease overfitting for Logistic Regression?"
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


question = 'How can I'

def bot_reply(question):
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
