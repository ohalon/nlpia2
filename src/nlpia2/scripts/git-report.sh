for filename in "Chapter 02 -- Build Your Vocabulary (Word Tokenization).adoc" ""
do
    git blame --since=90.days -- $filename
done
