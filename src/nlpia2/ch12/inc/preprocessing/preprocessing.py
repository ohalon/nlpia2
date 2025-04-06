#----------------------------------------------------------------------------------
# preprocessing.py: Utils for preprocessing text
#----------------------------------------------------------------------------------

import glob
import json
import os
import re

from flashtext.keywords import KeywordProcessor


class TextPreprocessor(object):
    
    spaces = ['\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\x10', '\x7f', 
                  '\x9d', '\xad', '\xa0']
    
    
    punct = [',', '.', '"', ':', ')', '(', '-', '!', '?', '|', ';', "'", '$', '&', '/', '[', ']', '>', 
             '%', '=', '#', '*', '+', '\\', '•', '~', '@', '£', '·', '_', '{', '}', '©', '^', '®', '`', 
             '<', '→', '°', '€', '™', '›', '♥', '←', '×', '§', '″', '′', 'Â', '█', '½', 'à', '…', '“', '★', 
             '”', '–', '●', 'â', '►', '−', '¢', '²', '¬', '░', '¶', '↑', '±', '¿', '▾', '═', '¦', '║', '―',
             '¥', '▓', '—', '‹', '─', '▒', '：', '¼', '⊕', '▼', '▪', '†', '■', '’', '▀', '¨', '▄', '♫', '☆',
             'é', '¯', '♦', '¤', '▲', 'è', '¸', '¾', 'Ã', '⋅', '‘', '∞', '∙', '）', '↓', '、', '│', '（', '»',
             '，', '♪', '╩', '╚', '³', '・', '╦', '╣', '╔', '╗', '▬', '❤', 'ï', 'Ø', '¹', '≤', '‡', '√', 'θ',
             '₹', 'º', 'β', '«', '´', '₤', '¡', '÷', '∅', 'α', 'π']

    
    def __init__(self):
        # Load json resources in this module (preprocessing)
        # into a flashtext KeywordProcessor instance. 
        self.kwp = self._load_resources()
        
        self.kw_replace_errors = []
        
    def _load_resources(self):
        text_replacement_dicts = []
        searchpath = os.path.join(os.path.dirname(__file__), 'json/*.json')
        files = glob.glob(searchpath)
        for filename in files:
            try:
                with open(filename) as f:  
                    data = json.load(f)
                    text_replacement_dicts.append(data)
                    print('loaded', filename)
            except:
                print('error: could not load', filename)
                   
        kwp = KeywordProcessor()
        for d in text_replacement_dicts:
            for key, value in d.items():
                kwp.add_keyword(key, value)
        
        return kwp
    
    def _replace_keywords(self, text):
        try:
            return self.kwp.replace_keywords(text)
        except:
            # flashtext hits list index out of bounds error sometimes.
            # save the text to try figure what went wrong
            self.kw_replace_errors.append(text)
        
        return text

    def preprocess(self, text):
        text = self._remove_space(text)
        text = self._replace_special_punctuation(text)
        text = self._clean_numbers(text)
        text = self._replace_keywords(text)
        text = self._remove_space(text) # remove extra spaces that might have been introduced
                
        return text
        
   
    def _remove_space(self, text):
        """
        Remove extra spaces and ending space if any
        """
        for space in TextPreprocessor.spaces:
            if space in text:
                text = text.replace(space, ' ')
        text = text.strip()
        text = re.sub('\s+', ' ', text)
        
        return text
    
    def _replace_special_punctuation(self, text):
        special_punc_mappings = {"—": "-", "–": "-", "_": "-", '”': '"', "″": '"', '“': '"', '•': '.', '−': '-',
                         "’": "'", "‘": "'", "´": "'", "`": "'", '\u200b': ' ', '\xa0': ' ','،':'','„':'',
                         '…': ' ... ', '\ufeff': ''}
        
        for punc in special_punc_mappings:
            if punc in text:
                text = text.replace(punc, special_punc_mappings[punc])
        return text
    
    def _clean_numbers(self, text):
        text = re.sub(r'(\d+)([a-zA-Z])', '\g<1> \g<2>', text)
        text = re.sub(r'(\d+) (th|st|nd|rd) ', '\g<1>\g<2> ', text)
        text = re.sub(r'(\d+),(\d+)', '\g<1>\g<2>', text)
        text = re.sub(r'(\d+)(e)(\d+)','\g<1> \g<3>', text)

        return text