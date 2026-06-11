"""
@author : Hyunwoong
@when : 2019-10-29
@homepage : https://github.com/gusdnd852
"""
import re

try:
    from nltk.tokenize import word_tokenize
except ModuleNotFoundError:
    word_tokenize = None

class Tokenizer:
    def __init__(self):
        self.spacy_de = 'german'
        self.spacy_en = 'english'

    def tokenize_de(self, text):
        """
        Tokenizes German txt from a string into a list of strings
        """
        if word_tokenize is None:
            raise RuntimeError("NLTK is required for Multi30K tokenization but is not installed.")
        return word_tokenize(text, language=self.spacy_de)

    def tokenize_en(self, text):
        """
        Tokenizes English text from a string into a list of strings
        """
        if word_tokenize is None:
            raise RuntimeError("NLTK is required for Multi30K tokenization but is not installed.")
        return word_tokenize(text, language=self.spacy_en)

    def tokenize_char(self, text):
        """Tokenize structured sequence data such as SMILES, InChI, DNA, or proteins."""
        return list(text.strip())

    def tokenize_code(self, text):
        """Tokenize one-line normalized source-code samples without requiring a parser."""
        return re.findall(
            r"<NL>|<TAB>|<SPACE>|[A-Za-z_]\w*|\d+\.\d+|\d+|==|!=|<=|>=|->|:=|\*\*|//|[^\s]",
            text,
        )
