# Resources
:chapter: 14
:part: BM
:imagesdir: .
:xrefstyle: short
:figure-caption: Figure {chapter}.
:listing-caption: Listing {chapter}.
:table-caption: Table {chapter}.
:stem: latexmath

In writing this book we pulled from numerous resources.
Here are some of our favorites.

In an ideal world, you could find these resources yourself simply by entering the heading text into a semantic search engine like Duck Duck Go (http://duckduckgo.com), Gigablast (http://gigablast.com/search?c=main&q=open+source+search+engine), or Qwant (https://www.qwant.com/web).
But until Jimmy Wales takes another shot at Wikia Search (https://en.wikipedia.org/wiki/Wikia_Search) or Google shares their NLP technology, we have to rely on 1990s-style lists of links like this.
Check out the <<search_engines_section>> section if your contribution to saving the world includes helping open source projects that index the web.

## Applications and project ideas

Here are some applications to inspire your own NLP projects.

* Retrain or specialize a sentence encoder (embedding model) on a variety of tasks including those used for the Universal Sentence Encoder (https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a).
* Compute an "information content" or complexity score for an NLP corpus by finding the minimum LSTM memory vector (embedding) dimensionality (number of encoder output neurons) required to perform a translation or autoencoder or word prediction task and near-state-of the art accuracye. Use the InferSent (https://arxiv.org/abs/1705.02364) paper and code (https://github.com/facebookresearch/InferSent) to evaluate performance on the same set of tasks that Facebook uses to evaluate their language models (embedders).
* Guessing passwords from social network profiles (http://www.sciencemag.org/news/2017/09/artificial-intelligence-just-made-guessing-your-password-whole-lot-easier)
* Predict censorship trends: Perform NLP/NLU of historical archive of twitter feeds and other news articles to generate a timeseries of NLU embedding vectors and other NLP features to predict censorship topic, geographic region, keywords and activity magnitude. These target variables can be obtained from raw Censored Planet logs (http://censoredplanet.org/data/rawand) using their visualizations (http://censoredplanet.org/data/visualizations) to reproduce their censorship metrics for historical time periods.
* "Chatbot lawyer overturns 160,000 parking tickets in London and New York" (www.theguardian.com/technology/2016/jun/28/chatbot-ai-lawyer-donotpay-parking-tickets-london-new-york)
* "GitHub - craigboman/gutenberg: Librarian working with project gutenberg data, for NLP and machine learning purposes." (https://github.com/craigboman/gutenberg)
* _Longitudial Detection of Dementia Through Lexical and Syntactic Changes in Writing_ (ftp://ftp.cs.toronto.edu/dist/gh/Le-MSc-2010.pdf) -- Masters thesis by Xian Le on psychology diagnosis with NLP
* Time Series Matching (https://www.cs.nyu.edu/web/Research/Theses/wang_zhihua.pdf) -- Songs, audio clips, and other time series can be discretized and searched with dynamic programming algorithms analogous to Levenshtein distance.
* NELL, Never Ending Language Learning (http://rtw.ml.cmu.edu/rtw/publications) -- CMU's constantly evolving knowledge base that learns by scraping natural language text
* How the NSA identified Satoshi Nakamoto (https://medium.com/cryptomuse/how-the-nsa-caught-satoshi-nakamoto-868affcef595) -- Wired Magazine and the NSA identified Satoshi Nakamoto using NLP, or "stylometry".
* Stylometry (https://en.wikipedia.org/wiki/Stylometry) and Natural Language Forensics (http://www.parkjonghyuk.net/lecture/2017-2nd-lecture/forensic/s8.pdf) -- Style/pattern matching and clustering of natural language text (also music and artwork) for authorship and attribution
* Online dictionaries like Your Dictionary (http://examples.yourdictionary.com/) can be scraped for grammatically correct sentences with POS labels which can be used to train your own Parsey McParseface (https://ai.googleblog.com/2016/05/announcing-syntaxnet-worlds-most.html) syntax tree and POS tagger.
* "Identifying 'Fake News' with NLP" (https://nycdatascience.com/blog/student-works/identifying-fake-news-nlp/) by Julia Goldstein and Mike Ghoul at NYC Data Science Academy.
* A simpleNumericalFactChecker (https://github.com/uclmr/simpleNumericalFactChecker) by Andreas Vlachos (https://github.com/andreasvlachos) and information extraction (see chapter 11) could be used to rank publishers, authors, and reporters for truthfulness. Might be combined with Julia Goldstein's "fake news" predictor.
* The artificial-adversary (https://github.com/airbnb/artificial-adversary) package by Jack Dai, an intern at Airbnb -- Obfuscates natural language text (turning phrases lke 'you are great' into 'ur gr8'). You could train a machine learning classifier to detect and "translate" English into obfuscated English or L33T (https://sites.google.com/site/inhainternetlanguage/different-internet-languages/l33t). You could also train a stemmer (an autoencoder with the obfuscator generating character features) to decipher obfuscated words so your NLP pipeline can handle obfuscated text without retraining. Thank you Aleck.

## Courses and tutorials

Here are some good tutorials, demonstrations, and even courseware from renowned university programs, many of which include Python examples.

* _Speech and Language Processing_ (https://web.stanford.edu/\~jurafsky/slp3/ed3book.pdf) by David Jurafsky and James H. Martin -- The next book you should read if you're serious about NLP. Jurafsky and Martin are more thorough and rigorous in their explanation of NLP concepts. They have whole chapters on topics that we largely ignore, like finite state transducers (FSTs), hidden Markhov models (HMMs), part-of-speech (POS) tagging, syntactic parsing, discourse coherence, machine translation, summarization, and dialog systems.
* MIT Artificial General Intelligence course 6.S099 (https://agi.mit.edu) led by Lex Fridman Feb 2018 -- MIT's free, interactive (public competition!) AGI course. It's probably the most thorough and rigorous free course on artificial intelligence engineering you can find.
* "Textacy: NLP, before and after spaCy" (https://github.com/chartbeat-labs/textacy) -- Topic modeling wrapper for SpaCy
* MIT Natural Language and the Computer Representation of Knowledge course 6-863j lecture notes (https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-863j-natural-language-and-the-computer-representation-of-knowledge-spring-2003/lecture-notes/) for Spring 2003
* "Linear Algebra tutorial: Singular value decomposition (SVD)" (http://people.revoledu.com/kardi/tutorial/LinearAlgebra/SVD.html) by Kardi Teknomo, PhD
* _An Introduction to Information Retrieval_ (https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf) by Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze -- _An Introduction to Information Retrieval_, April 1, Online Edition, by Christopher Manning, Prabhakar Raghavan, and Hinrich Schutze at Stanford (creators of the Stanford CoreNLP library)

## Tools and Packages

* nlpia (http://github.com/totalgood/nlpia) -- NLP datasets, tools, and example scripts from this book.
* `OpenFST` (http://openfst.org/twiki/bin/view/FST/WebHome) by Tom Bagby, Dan Bikel, Kyle Gorman, Mehryar Mohri et al. -- Open Source C++ Finite State Transducer implementation.
* `pyfst` (https://github.com/vchahun/pyfst) by Victor Chahuneau -- A python interface to OpenFST
* Stanford `CoreNLP` (https://stanfordnlp.github.io/CoreNLP/) by Christopher D. Manning et al -- Java library with state of the art sentence segmentation, datetime extraction, POS tagging, grammar checker, etc.
* `stanford-corenlp` (https://pypi.org/project/stanford-corenlp/) -- Python interface to Stanford `CoreNLP`
* `keras` (https://blog.keras.io/) -- High level API for constructing both TensorFlow and Theano computational graphs (neural nets)

== Research papers and talks

One of the best way to gain a deep understanding of a topic is to try to repeat the experiments of researchers and then modify them in some way.
That's how the best professors and mentors "teach" their students, by just encouraging them to try to duplicate the results of other researchers they are interested in.
You can't help but tweak an approach if you spend enough time trying to get it to work for you.

### Vector space models and semantic search

* Semantic Vector Encoding and Similarity Search Using Full Text Search Engines (https://arxiv.org/pdf/1706.00957.pdf) -- Jan Rygl et. al were able to use a conventional inverted index to implement efficient semantic search for all of Wikipedia.
* Learning Low-Dimensional Metrics (https://papers.nips.cc/paper/7002-learning-low-dimensional-metrics.pdf) -- Lalit Jain, et al, were able to incorporate human judgement into pairwise distance metrics, which can be used for better decision-making and unsupervised clustering of word vectors and topic vectors. For example, recruiters can use this to steer a content-based recommendation engine that matches resumes with job descriptions.
* _RAND-WALK: A latent variable model approach to word embeddings_ (https://arxiv.org/pdf/1502.03520.pdf) by Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, Andrej Risteski -- Explains the latest (2016) understanding of the "vector-oriented reasoning" of Word2vec and other word vector space models, particularly analogy questions
* "Efficient Estimation of Word Representations in Vector Space" (https://arxiv.org/pdf/1301.3781.pdf) by Tomas Mikolov, Greg Corrado, Kai Chen, Jeffrey Dean at Google, Sep 2013 -- First publication of the Word2vec model, including an implementation in C++ and pretrained models using a Google News corpus
* "Distributed Representations of Words and Phrases and their Compositionality" (https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) by Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg Corrado, Jeffrey Dean at Google -- Describes refinements to the Word2vec model that improved its accuracy, including subsampling and negative sampling
* _From Distributional to Semantic Similarity_ (https://www.era.lib.ed.ac.uk/bitstream/handle/1842/563/IP030023.pdf) 2003 Ph.D. Thesis by James Richard Curran -- Lots of classic information retrieval (full-text search) research, including TF-IDF normalization and page rank techniques for web search
* Universal Sentence Encoder (https://alpha.tfhub.dev/google/universal-sentence-encoder/2) -- TensorFlow Hub has a pretrained TensorFlow model for sentence embedding. It is similar to doc2vec in gensim, only much more accurate, because this model takes into account word order and grammar using multiple LSTM layers.

### Finance

* "Predicting Stock Returns by Automatically Analyzing Company News Announcements" (http://www.stagirit.org/sites/default/files/articles/a_0275_ssrn-id2684558.pdf) -- Bella Dubrov used gensim's Doc2vec to predict stock prices based on company announcements with excellent explanations of `Word2vec` and `Doc2vec`.
* _Building a Quantitative Trading Strategy to Beat the S&P 500_ (https://www.youtube.com/watch?v=ll6Tq-wTXXw) -- At PyCon 2016, Karen Rubin explained how she discovered that female CEOs are predictive of rising stock prices, though not as strongly as she initially thought.

### Question answering systems

* _Keras-based LSTM/CNN models for Visual Question Answering_ (https://github.com/avisingh599/visual-qa) by Avi Singh
* _Open Domain Question Answering: Techniques, Resources and Systems_ (http://lml.bas.bg/ranlp2005/tutorials/magnini.ppt) by Bernardo Magnini
* _2003 EACL tutorial by Lin Katz, University of Waterloo, Canada_ (https://cs.uwaterloo.ca/~jimmylin/publications/Lin_Katz_EACL2003_tutorial.pdf)
* _NLP-Question-Answer-System_ (https://github.com/raoariel/NLP-Question-Answer-System/blob/master/simpleQueryAnswering.py) -- built from scratch using `corenlp` and `nltk` for sentence segmenting and POS tagging
* _PiQASso: Pisa Question Answering System_ (http://trec.nist.gov/pubs/trec10/papers/piqasso.pdf) by Attardi, et al, 2001 -- Uses traditional information retrieval (IR) NLP.
* _Knowledge Based AI: Cognitive Systems_ (https://www.udacity.com/course/knowledge-based-ai-cognitive-systems--ud409) -- Georgia Tech course on Udacity where the example project is an AI system that can extract visual knowledge (semantics) from Raven visual analogy matrix questions (IQ tests) and answer them as well as a human.

### Deep learning

* "Understanding LSTM Networks" (https://colah.github.io/posts/2015-08-Understanding-LSTMs) by Christopher Olah -- A clear and correct explanation of LSTMs
* "Learning Phrase Representations using RNN Encoder–Decoder for Statistical Machine Translation" (https://arxiv.org/pdf/1406.1078.pdf) by Kyunghyun Cho, et al, 2014 -- Paper that first introduced gated recurring units making LSTMs more efficient for NLP

### LSTMs and RNNs

We had a lot of difficulty understanding the terminology and architecture of LSTMs. This is a gathering of the most cited references so you can let the authors "vote" on the right way to talk about LSTMs. The state of the Wikipedia page (and Talk page discussion) on LSTMs is a pretty good indication of the lack of consensus about what LSTM means.

* "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation" (https://arxiv.org/pdf/1406.1078.pdf) by Cho, Bengio, et al -- Explains how the contents of the memory cells in an LSTM layer can be used as an embedding that can encode variable length sequences and then decode them to a new variable length sequence with a potentially different length, translating or transcoding on sequence into another
* "Reinforcement Learning with Long Short-Term Memory" (https://papers.nips.cc/paper/1953-reinforcement-learning-with-long-short-term-memory.pdf) by Bram Bakker -- Application of LSTMs to planning and anticipation cognition with demonstrations of a network that can solve the T-maze navigation problem and an advanced pole-balancing (inverted pendulum) problem
* "Supervised Sequence Labelling with Recurrent Neural Networks" (https://mediatum.ub.tum.de/doc/673554/file.pdf) -- Thesis by Alex Graves with advisor B. Brugge; a detailed explanation of the mathematics for the exact gradient for LSTMs as first proposed by Hochreiter and Schmidhuber in 1997. But Gaves' first language may not be English, and he fails to define terms like CEC or LSTM _block_/_cell_ rigorously.
* Theano LSTM documentation (http://deeplearning.net/tutorial/lstm.html) by Pierre Luc Carrier and Kyunghyun Cho -- Diagram and discussion to explain the LSTM implementation in Theano and Keras
* "Learning to forget: Continual prediction with LSTM. Neural computation" (https://www.researchgate.net/profile/Felix_Gers/publication/12292425_Learning_to_Forget_Continual_Prediction_with_LSTM/links/5759414608ae9a9c954e84c5/Learning-to-Forget-Continual-Prediction-with-LSTM.pdf) by F. A., Schmidhuber, J., & Cummins -- Uses nonstandard notation for layer inputs (**y**^in^) and outputs (**y**^out^) and internal hidden state (**h**). All math and diagrams are "vectorized."
* "Sequence to Sequence Learning with Neural Networks" (http://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) by Ilya Sutskever, Oriol Vinyals, and Quoc V. Le at Google
* "Understanding LSTM Networks" (http://colah.github.io/posts/2015-08-Understanding-LSTMs) 2015 blog by Charles Olah -- lots of good diagrams and discussion/feedback from readers
* "Long Short-Term Memory" (http://www.bioinf.jku.at/publications/older/2604.pdf) by Sepp Hochreiter and Jurgen Shmichuber, 1997 -- Original paper on LSTMs with outdated terminology and inefficient implementation, but detailed mathematical derivation

## Competitions and awards

* "Large Text Compression Benchmark" (http://mattmahoney.net/dc/text.html) -- Some researchers believe that compression of natural language text is equivalent to artificial general intelligence (AGI)
* "The Hutter Prize" (https://en.wikipedia.org/wiki/Hutter_Prize) -- Annual competition to compress a 100 MB archive of Wikipedia natural language text. Alexander Rhatushnyak won in 2017.
* "Open Knowledge Extraction Challenge" (https://svn.aksw.org/papers/2017/ESWC_Challenge_OKE/public.pdf)

## Datasets

Natural language data is everywhere you look.
Language is the superpower of the human race, and your pipeline should take advantage of it.

* "Datasets for Natural Language Processing" (https://proai.org/nlp-datasets-tds) by Jason Brownlee, Towards Data Science
* "15 Best Chatbot Datasets for Machine Learning" (https://gengo.ai/datasets/15-best-chatbot-datasets-for-machine-learning/) -- Paired natural language strings for training a chatbot: 4 question-answer, 9 dialog, and 2 translation corpora to train or test your next chatbot with.
* Google's Dataset Search (http://toolbox.google.com/datasetsearch) -- a search engine similar to Google Scholar (http://scholar.google.com), but for data.
* "Stanford Datasets" (https://nlp.stanford.edu/data/) -- Pretrained `word2vec` and GloVE models, multilingual language models and datasets, multilingual dictionaries, lexica, and corpora.
* "Pretrained word vector models" (https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-model) -- The README for a word vector web API provides links to several word vector models, including the 300-D Wikipedia GloVE model.
* "A list of datasets/corpora for NLP tasks, in reverse chronological order" (https://github.com/karthikncode/nlp-datasets) by Karthik Narasimhan
* "Alphabetical list of free/public domain datasets with text data for use in Natural Language Processing (NLP)" (https://github.com/niderhoff/nlp-datasets)
* "Datasets and tools for basic natural language processing" (https://github.com/googlei18n/language-resources) -- Google's International tools for i18n
* `nlpia` (https://github.com/totalgood/nlpia) --  Python package with data loaders (`nlpia.loaders`) and preprocessors for all the NLP data you will ever need... until you finish this book ;)

## Search engines

Search (information retrieval) is a big part of NLP.
And it's extremely important that we get it right so that our AI (and corporate) overlords can't manipulate us through the information they feed our brains.
If you want to learn how to retrieve your own information, build your own search engines, these are some resources that will help.

### Search algorithms

* "GPU-enhanced BidMACH" (https://arxiv.org/pdf/1702.08734.pdf) -- BidMACH is a high-dimensional vector indexing and KNN search implementation, similar to the `annoy` python package. This paper explains an enhancement for GPUs that is 8x faster than the original implementation.
* Spotify's `Annoy` Package (https://erikbern.com/2017/11/26/annoy-1.10-released-with-hamming-distance-and-windows-support.html) by Erik Bernhardsson's -- A K nearest neighbors algorithm used at Spotify to find "similar" songs.
* "Erik Bernhardsson's ANN Comparison" (https://erikbern.com/2018/02/15/new-benchmarks-for-approximate-nearest-neighbors.html) -- Approximate nearest neighbor algorithms are the key to scalable semantic search, and Erik keeps tabs on the state of the art.

### Open source search engines

* "BeeSeek" (https://launchpad.net/~beeseek-devs) -- Open source distributed web indexing and private search (hive search); no longer maintained
* "WebSphinx" (https://www.cs.cmu.edu/~rcm/websphinx/) -- Web GUI for building a web crawler

### Open source full-text indexers

Efficient indexing is critical to any natural language search application.
Here are a few open source full-text indexing options.
However, these "search engines" do not crawl the web, so you need to provide them with the corpus you want them to index and search.

* "Elastic Search" (https://github.com/elastic/elasticsearch) -- "Open Source, Distributed, RESTful Search Engine".
* "Apache Lucern + Solr" (https://github.com/apache/lucene-solr)
* "Sphinx Search" (https://github.com/sphinxsearch/sphinx)
* "Kronuz/Xapiand: Xapiand: A RESTful Search Engine" (https://github.com/Kronuz/Xapiand) -- There are packages for Ubuntu that will let you search your local hard drive (like Google Desktop used to do).
* "Lemur Project Components: Indri" (http://www.lemurproject.org/indri.php) -- Semantic search with a Python interface (https://github.com/cvangysel/pyndri) but it isn't actively maintained.
* "Gigablast" (https://github.com/gigablast/open-source-search-engine) -- Open source web crawler and natural language indexer in C++.
* "Zettair" (http://www.seg.rmit.edu.au/zettair) -- Open source HTML and TREC indexer (no crawler or live example); last updated 2009.
* "OpenFTS: Open Source Full Text Search Engine" (http://openfts.sourceforge.net) -- Full text search indexer for PyFTS using PostgreSQL with a Python API (http://rhodesmill.org/brandon/projects/pyfts.html).

### Manipulative search engines

The search engines most of us use are not optimized solely to help you find what you need, but rather to ensure that you click  links that generate revenue for the company that built it.
Google's innovative second-price sealed-bid auction ensures that advertisers don't overpay for their ads,footnote:[Cornell University Networks Course case study, "Google Adwords Auction" (https://blogs.cornell.edu/info2040/2012/10/27/google-adwords-auction-a-second-price-sealed-bid-auction)] but it doesn't prevent search users from  overpaying when they click disguised advertisements.
This manipulative search is not unique to Google, but any search engine that ranks results according to any other "objective function" other than your satisfaction with the search results.
But here they are, if you want to compare and experiment.

* Google
* Bing
* Baidu

### Less manipulative search engines

To determine how "commercial" and manipulative a search engine was, I queried several engines with things like "open source search engine".
I then counted the number of ad-words purchasers and click-bait sites were among the search results in the top 10.
The following sites kept that count below one or two.
And the top search results were often the most objective and useful sites, such as Wikipedia, Stack Exchange, or reputable news articles and blogs.

* Alternatives to Google (https://www.lifehack.org/374487/try-these-15-search-engines-instead-google-for-better-search-results) footnote:[See the web page titled "Search Engines Instead of Google For Better Search Results" (https://www.lifehack.org/374487/try-these-15-search-engines-instead-google-for-better-search-results).]
* Yandex (https://yandex.com/search/?text=open%20source%20search%20engine&lr=21754) -- Surprisingly, the most popular Russian search engine (60% of Russian searches) seemed less manipulative than the top US search engines.
* DuckDuckGo (https://duckduckgo.com)
* `Watson` Semantic Web Search (http://watson.kmi.open.ac.uk/WatsonWUI) -- No longer in development, and not really a full text web search, but it is an interesting way to explore the semantic web (at least what it was years ago before `watson` was frozen)

### Distributed search engines

Distributed search engines footnote:[See the web page titled "Distributed search engine - Wikipedia" (https://en.wikipedia.org/wiki/Distributed_search_engine).] footnote:[See the web page titled "Distributed Search Engines - P2P Foundation" (https://wiki.p2pfoundation.net/Distributed_Search_Engines).] are perhaps the least manipulative and most "objective" because they have no central server to influence the ranking of the search results.
However, current distributed search implementations rely on TF-IDF word frequencies to rank pages, because of the difficulty in scaling and distributing semantic search NLP algorithms.
However, distribution of semantic indexing approaches such as latent semantic analysis (LSA) and locality sensitive hashing have been successfully distributed with nearly linear scaling (as good as you can get).
It's just a matter of time before someone decides to contribute code for semantic search into an open source project like Yacy or builds a new distributed search engine capable of LSA.

* Nutch (https://nutch.apache.org/) -- Nutch spawned Hadoop and itself became less of a distributed search engine and more of a distributed HPC system over time.
* Yacy (https://www.yacy.net/en/index.html) -- One of the few open source (https://github.com/yacy/yacy_search_server) decentralized, or federated, search engines and web crawlers still actively in use. Preconfigured clients for Mac, Linux, and Windows are available.

