#!/usr/bin/env bash

CODE_DIR=./code
CODE_DIR=../nlpia2/src/nlpia2

if [[ -d "$CODE_DIR" ]] ; then
    python $CODE_DIR/ch02/book_thief_sentence_split_graphviz.py
    python $CODE_DIR/ch01/NLU_NLG_graphviz.py
    python $CODE_DIR/scripts/render_html.py
else
    echo "Unable to find $CODE_DIR"
fi
firefox $(pwd)/manuscript/html
