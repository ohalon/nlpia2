FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime
# FIXME: apt cannot install `git` which is required by `pip freeze`

# FROM jupyter/base-notebook
# FROM condaforge/miniforge3

ENV TRANSFORMERS_CACHE=/tmp/.cache
ENV TOKENIZERS_PARALLELISM=true
ENV PYTHONUNBUFFERED 1

WORKDIR /nlpia2

ADD requirements.txt .

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN python -m spacy download en_core_web_sm

RUN python -m spacy download en_core_web_md

RUN python -m spacy download en_core_web_lg

RUN python -c 'from sentence_transformers import SentenceTransformer; sbert = SentenceTransformer("paraphrase-MiniLM-L6-v2")'

ADD . .

RUN pip install -e .
# WARNING: `EXPOSE` does nothing, it's basically a reminder to `run --publish 8888:8888` 
EXPOSE 8888


CMD ["./scripts/entrypoint.sh"]