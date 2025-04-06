# if [ -f '.venv/bin/activate' ] ; then
#     echo "Found existing venv in .venv/ so now running `source .venv/bin/activate`"
# else
#     python3.7 -m venv .venv
# fi
# source .venv/bin/activate

if [ -z `which aptitude` ] ; then
    echo "You don't appear to be on an Ubuntu machine, so you'll need to install your binaries."
else
    sudo aptitude install libpython3-dev build-essential git automake xdotool autoconf intltool autopoint
    sudo apt install asciidoctor
fi
conda create -n nlpia2 install -y --upgrade 'python==3.11.4' pip
conda activate nlpia2
conda install jupyter pandas scikit-learn scipy nltk spacy -c pytorch pytorch torchtext -c spacy spacy-lookups-data graphviz dot
conda install -c conda-forge git-bash
# pip install -y -r requirements.txt
python -m spacy download en_core_web_md
# pip install spacy-lookups-data  # for lemmatization
