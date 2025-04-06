[![Substack][substack-shield]][substack-url]
[![Nitter][nitter-shield]][nitter-url]
[![Mastodon][mastodon-shield]][mastodon-url]
[![Linkedin][linkedin-shield]][linkedin-url]

# Corrected Text Classification with CNNs

## Text Classification with CNNs in PyTorch 

Re-implementation, simplification of [Fernando Lopez][linkedin-fernando]'s ([@FernandoLpz](https://github.com/FernandoLpz)) [Text Classification CNN](https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch) which he based on the paper "[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/pdf/1408.5882.pdf)."
He has some nice diagrams in his Medium (PAYWALL) blog post [Text Classification with CNNs in PyTorch ](https://12ft.io/proxy?q=https%3A%2F%2Ftowardsdatascience.com%2Ftext-classification-with-cnns-in-pytorch-1113df31e79f).


## Preprocessing

For the book we improved the Fernando's pipeline to make it more readable, pythonic, and accurate.
Just as in previous chapters (and in general), you do not want to use case folding or stopword removal for your first model.
Give your model all the information you can for the first attempt at training it.
You only need to filter stopwords and lowercase your text if you have a dimensionality problem (information overload).
You can tune your model later to deal with overfitting or slow convergence.
Usually there are more effective approaches to dealing with overfitting.

The preprocessing pipeline here is the same one we used for chapters 1-6 in [NLPiA](proai.org/nlpia-book).
You can see where we've changed the implementation suggested by Fernando.

1. Simplified: load the CSV

`df = pd.read_csv(filepath, usecols=['text', 'target'])`

2. NOPE: case folding:

`texts = df['texts'].lower()`

3. NOPE: remove non-letters (nonalpha):

`re.sub(r'[^A-Za-z]', text, ' ')`

4. NOPE: remove stopwords

5. Simplified: tokenize with regex

`tokenized_texts = map(re.compile(r'\w+').findall, texts)`

6. Simplified: filter infrequent words

`counts = Counter(chain(*tokenized_texts))`
`id2tok = vocab = list(counts.most_common(200))`

7. Simplified: compute reverse index

`tok2id = dict(zip(idx2tok, range(len(id2tok))))`

8. Simplified: transform token sequences to integer id sequences

`id_sequences = [[tok2id[tok] for tok in tokens] for tokens in tokenized_texts]`

9. Simplified: pad token id sequences


10. Simplified: train_test_split

x_train, y_train, x_test, y_test = train_test_split(x=texts, y=df['targets'])

This is common in PyTorch implementations, but the integers are never used as the input to a Deep Learning model because the numerical values of word indices do not contain any information about the words themselves other than their position in a one-hot vector.
The index values are arbitrary and destroy all the information contained in a sentence.
The numerical values of word indices contain no information about the meaning of the words. Word indices are fundamentally a categorical variable, not an ordinal or numerical value.
Fernando's code effectively randomizes the 1-D numerical representation of a word thus the  sequence of indices passed into the CNN is meaningless.

This can be seen in the poor accuracy on the test set compared to the training set. 
The test set accuracy is little better than random guessing or always guessing the majority class.
In pyTorch a word embeddings layers is typically utilized to reconstruct a meaningful vector.

This repository attempts to correct this fundamental error by replacing word indices with progressively more informative numerical representations of words:

- removing the CNN entirely
- 2-D CNN on one-hot vectors
- 2-D CNN on TF-IDF vectors
- 2-D CNN on LSA vectors (PCA on TF-IDF vectors)
- 2-D CNN on word vectors

In addition, a baseline LinearModel (a single-layer perceptron, or fully-connected feed-forward layer) is trained on the average of the word vectors for a sentence or tweet (`spacy.Doc.vector`).

## Links


[substack-shield]: https://proai.substack.com
[substack-url]: https://proai.substack.com

[mastodon-shield]: https://img.shields.io/mastodon/follow/001019390?style=social
[mastodon-url]: https://mastodon.social/users/hobson

[nitter-url]: https://nitter.net/hobsonlane
[nitter-shield]: https://nitter.net/hobsonlane

[linkedin-shield]: https://img.shields.io/badge/linkedin-%230077B5.svg?&style=for-the-badge&logo=linkedin&logoColor=white
[linkedin-url]: https://www.linkedin.com/in/hobsonlane/

[linkedin-fernando]: https://www.linkedin.com/in/fernando-lopezvelasco/