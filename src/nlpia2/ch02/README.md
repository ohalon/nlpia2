# Chapter 2 -- Tokenizer

## Fake News Feature Engineering

1. To clean the dataset: [%run fake_news_clean.py](fake_news_clean.py) or `df = read_csv('../../data/all.csv.gz')`
2. To understand TFIDF vectors: [fake_news_manual_counter_vectorizer_and_tfidf.ipy](fake_news_manual_counter_vectorizer_and_tfidf.ipy)
3. Example modeling and feature engineering approaches: [fake_news_token_features.py](fake_news_token_features.py)
4. Chunking and IncrementalPCA example: [fake_news_big_data_techniques.ipy](fake_news_big_data_techniques.ipy)
5. To discover the "leakage" or "cheating" that makes your model so accurate, examine the top 5 TF-IDF features (1-gram tokens) based on the coefficients of a linear regression.

Video of Tangible AI interns in their weekly ["improv coding" session](https://tan.sfo2.digitaloceanspaces.com/videos/howto/howto-improv-coding--big-data-tfidf-vectorizer-counter-todense-hanna-camille-martha-maria-john-2021-07-14.mp4)

