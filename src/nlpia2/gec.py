""" Load a pretrained GEC (Grammar Error Corrector) for English

based on Open NMT (Neural Machine Translation) model
References:
- huggingface.co/models/jordimas/gec-opennmt-english
"""

from pathlib import Path
import ctranslate2
import pyonmttok
from huggingface_hub import snapshot_download
import difflib

class Corrector:
    def __init__(self,
            repo_id="jordimas/gec-opennmt-english",
            model_dir=None,
            revision="main"):
        if model_dir is None:
            self.model_dir = snapshot_download(
                repo_id=repo_id, revision=revision)
        else:
            self.model_dir = Path(model_dir)
        self.tokenizer = pyonmttok.Tokenizer(
            mode="none",
            sp_model_path=self.model_dir + "/sp_m.model")
        self.translator = ctranslate2.Translator(self.model_dir)

    def correct(self, src):
        self.src = src
        self.src_tokens = self.tokenizer.tokenize(self.src)[0]

        # translator can return multiple hypotheses
        self.translated_batch = self.translator.translate_batch(
            [self.src_tokens])
        self.tgt_tokens = self.translated_batch[0][0]['tokens']
        self.tgt = self.tokenizer.detokenize(self.tgt_tokens)
        return self.tgt

    def __call__(self, src):
        return self.correct(src)


def highlight_diff(src_tokens, tgt_tokens):
    differ = difflib.Differ()
    tagged_tokens = list(differ.compare(src_tokens, tgt_tokens))
    print(tagged_tokens)
    return tagged_tokens
    
    for (i_s, s), (i_t, t) in zip(enumerate(src_tokens), enumerate(tgt_tokens)):
        print (i_s, s), (i_t, t)


corrector = Corrector()


if __name__ == '__main__':
    src = input("Enter an example paragraph: ")
    src = (src or
        "The water arent hot. My friends are going to be late. Today mine mother is in Barcelona."
        )    
    print(f"Attempting to correct:\n{src}\n")
    tgt = corrector.correct(src)
    delta = highlight_diff(corrector.src_tokens, corrector.tgt_tokens)

