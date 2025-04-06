 # 8387  more LICENSE.txt
 # 8391  source scripts/install_editable.sh
 # 8392  poetry name
 # 8393  poetry --help
 # 8394  poetry check
 # 8395  poetry config --help
 # 8396  poetry --help
 # 8397  basename $(pwd)
 # 8398  source scripts/install_editable.sh
 # 8399  mv scripts/install_editable.sh .
 # 8400  source install_editable.sh
 # 8401  git add install_editable.sh 
 # 8402  git status
 # 8403  cd src/
 # 8404  cd nlpia2
 # 8405  ls
 # 8406  ls -hal data
 # 8407  poetry build
 # 8408  ls -hal dist
 # 8409  cd ../../
 # 8410  ls -hal dist
 # 8411  poetry publish
 # 8412  poetry publish -f sdist
 # 8413  poetry publish --help
 # 8414  rm dist/nlpia2-0.0.17-py3-none-any.whl 
 # 8415  poetry publish --help
 # 8416  poetry -f sdist publish
 # 8417  poetry publish
 # 8418  nano .gitignore
 # 8419  git status
 # 8420  git commit -am 'install_editable.sh tested and published'
 # 8421  python src/nlpia2/ch07/cnn/main.py
 # 8422  python src/nlpia2/ch07/cnn/main.py
 # 8423  python src/nlpia2/ch07/cnn/main.py
 # 8424  python src/nlpia2/ch07/cnn/main.py
 # 8425  python src/nlpia2/ch07/cnn/main.py
 # 8426  ipython
 # 8427  python src/nlpia2/ch07/cnn/main.py
 # 8428  python src/nlpia2/ch07/cnn/main.py
 # 8429  python src/nlpia2/ch07/cnn/main.py
 # 8430  python src/nlpia2/ch07/cnn/main.py
 # 8431  python main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8432  python main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8433  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8434  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8435  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8436  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --embedding_size=256 --dropout_ratio=0
 # 8437  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --test_size=50 --embedding_size=256 --dropout_ratio=0
 # 8438  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --vocab_size=16000 --epochs=12 --test_size=50 --embedding_size=256 --dropout_ratio=0.5
 # 8439  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --kernel_lengths=[2,3,4,5,6] --vocab_size=15000 --epochs=15 --test_size=100 --embedding_size=128 --dropout_ratio=0.4
 # 8440  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --kernel_lengths=[2,3,4,5,6] --vocab_size=15000 --epochs=15 --test_size=100 --embedding_size=128 --dropout_ratio=0.4
 # 8441  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --stride=1 --kernel_lengths=[2,3,4,5] --vocab_size=5000 --epochs=15 --test_size=100 --embedding_size=64 --dropout_ratio=0.4
 # 8442  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --stride=1 --strides=[1,1,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=5000 --epochs=15 --test_size=100 --embedding_size=64 --dropout_ratio=0.3
 # 8443  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --num_stopwords=10 --stride=1 --strides=[1,1,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=5000 --epochs=12 --test_size=100 --embedding_size=64 --dropout_ratio=0.25
 # 8444  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --num_stopwords=10 --stride=1 --strides=[1,1,1,1] --kernel_lengths=[2,3,4] --vocab_size=5000 --epochs=12 --test_size=100 --embedding_size=32 --dropout_ratio=0.15
 # 8445  workon nlpia2
 # 8446  pip install nlpia2
 # 8447  pip uninstall nlpia2
 # 8448  pip uninstall nlpia2
 # 8449  pip install -e .
 # 8450  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --num_stopwords=0 --stride=1 --strides=[1,1,1] --kernel_lengths=[2,3,4] --vocab_size=10000 --epochs=12 --test_size=2 --embedding_size=300 --dropout_ratio=0
 # 8451  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=0 --num_stopwords=50 --stride=1 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=12 --test_size=.05 --embedding_size=63 --dropout_ratio=.2
 # 8452  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=50 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=12 --test_size=.05 --embedding_size=63 --dropout_portion=.2
 # 8453  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=50 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=63 --dropout_portion=.3
 # 8454  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=50 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8455  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=10 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8456  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=0 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=5000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8457  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --case_sensitive=1 '--re_sub=[^-_%#@&*[]{};,/=+A-Za-z0-9:().?!]+' --num_stopwords=15 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8458  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --case_sensitive=1 '--re_sub=[^-_%#@&*[]{};,/+A-Za-z0-9:().?!]+' --num_stopwords=15 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8459  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_spacy --case_sensitive=1 '--re_sub=[^-_%@&*[]{};,/+A-Za-z0-9:().?!]+' --num_stopwords=15 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=18 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8460  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=50 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=18 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8461  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=10 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=18 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8462  python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=10 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
python src/nlpia2/ch07/cnn/main.py --tokenizer=tokenize_re --case_sensitive=1 --num_stopwords=10 --strides=[2,2,2,2] --kernel_lengths=[2,3,4,5] --vocab_size=4000 --epochs=14 --test_size=.05 --embedding_size=64 --dropout_portion=.1
 # 8464  history | tail -n 100  > cnn_text_classification_disaster_tweets_77pct_accuracy.hist.sh
