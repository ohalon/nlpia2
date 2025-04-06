# conv1d_rightly_timed_pause
# SEE jupyter notebook


quote = "The right word may be effective, but no word was ever as effective as a rightly timed pause."
tags = 'ADV ADJ VERB NOUN'.split()
tagged_words = {
    f'{tok.i}:{tok.text}': [int(tok.pos_ == tag) for tag in tags]  # <1>
    for i, tok in nlp(quote)}                                      # <2>

df = pd.DataFrame(tagged_words, index=tags)
print(df)
#     The  right  word  may   be  ...    a  rightly  timed  pause    .
# ADV    0.0    0.0   0.0  0.0  0.0  ...  0.0      1.0    0.0    0.0  0.0
# ADJ    0.0    1.0   0.0  0.0  0.0  ...  0.0      0.0    0.0    0.0  0.0
# VERB   0.0    0.0   0.0  0.0  0.0  ...  0.0      0.0    1.0    0.0  0.0
# NOUN   0.0    0.0   1.0  0.0  0.0  ...  0.0      0.0    0.0    1.0  0.0
# match  1.0    0.0   0.0  0.0  0.0  ...  0.0      3.0    0.0    NaN  NaN
