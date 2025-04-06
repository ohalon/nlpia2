
reviews = [
    "I really love your chocolate ice cream.",
    "The vanilla ice cream was delicious.",
    "I like mint chocolate chip ice cream.",
    "Your chocolate ice cream is sick.",
    "I'm lactose intollerant. Ice cream makes me ill."
]

# [source,python]
# ----
good_words = [
    "good", "delicious", "great", "like", "love",
    "dank", "dope", "sick"]
ok_words = [
    "ok", "decent", "ice cream", "chocolate", "vanilla"]
bad_words = [
    "bad", "ill", "sick", "messy", "bland", "nasty"]
vocab = good_words + ok_words + bad_words
# ----


def count_keywords(sentence, keywords):
    keyword_counts = dict(zip(keywords, [0] * len(keywords)))
    for kw in keywords:
        if kw in sentence:
            keyword_counts[kw] += 1
    return keyword_counts


count_keywords(reviews[0], vocab)
