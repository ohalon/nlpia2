# install nlpia2 dependencies in a conda environment named `nlpia2`
# NOTE: does not install Anaconda3 nor miniconda

# create a new environment named "nlpia2" if one doesn't already exist:
conda activate nlpia2 || CONDA_DEFAULT_ENV=''

if [ "$CONDA_DEFAULT_ENV" != "nlpia2" ] ; then
    conda create -n nlpia2 -y 'python==3.8.8'
    conda activate nlpia2
    # install all of `nlpia2`'s dependences if they aren't already installed:
    conda install -c huggingface -c pytorch -c conda-forge -c defaults -y \
        asciidoctor \
        emoji \
        ffmpeg \
        glcontext \
        graphviz \
        huggingface_hub \
        jupyter \
        lxml \
        manimpango \
        nltk \
        pandoc \
        pyglet \
        pylatex \
        pyrr \
        pyopengl \
        pytablewriter \
        pytest \
        pytorch \
        regex \
        seaborn \
        scipy \
        scikit-learn \
        sentence-transformers \
        statsmodels \
        spacy \
        torchtext \
        transformers \
        wikipedia \
        xmltodict

python -m pip install manim manimgl

    # python ./conda_install_spacy_en_core_web_md.py || python ./scripts/conda_install_spacy_en_core_web_md.py

    # python -c "import spacy; import os; from pathlib import Path; nlp=spacy.load('en_core_web_sm'); modeldir=Path(nlp._path).parent.parent; files = os.listdir(modeldir); assert(any(f.startswith('en_core_web_sm') for f in files))" || python -m spacy download en_core_web_sm
    # python -c "import spacy; import os; from pathlib import Path; nlp=spacy.load('en_core_web_sm'); modeldir=Path(nlp._path).parent.parent; files = os.listdir(modeldir); assert(any(f.startswith('en_core_web_md') for f in files))" || python -m spacy download en_core_web_md
fi
