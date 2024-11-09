import re
import regex
import nltk
from spacy.cli import download as spacy_download
from multiprocessing import Value, Lock

# Shared counter and lock for multiprocessing
counter = Value('i', 0)
lock = Lock()

# Caches for stopwords, stemmers, and lemmatizers
stopwords_cache = {}
stemmer_cache = {}
lemmatizer_cache = {}

# Spacy language models for specified languages
spacy_languages = {
    'it': 'it_core_news_sm',
    'es': 'es_core_news_sm',
    'de': 'de_core_news_sm',
    'fr': 'fr_core_news_sm'
}

# Download Spacy models if not present
for lang, model in spacy_languages.items():
    try:
        spacy_download(model)
    except Exception as e:
        print(f"Error downloading model {model}: {e}")

# Download essential NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Language mappings and Latin-based languages
LANGS = {
    'ar': 'arabic',
    'en': 'english',
    'fr': 'french',
    'de': 'german',
    'es': 'spanish',
    'it': 'italian',
    'ko': 'korean'
}

LATIN_BASED_LANGS = {'en', 'it', 'es', 'de', 'fr'}

# Korean tokenizer (to be initialized in workers)
ko_tokenizer = None

# Arabic tokenizer
farasa_segmenter = None

# Regular expressions for word patterns
WORD_PATTERNS = {
    "general": regex.compile(r'[\p{L}\p{N}]+'),
    "arabic": regex.compile(r'[a-zA-Z0-9\u0600-\u06FF]+'),
    "korean": regex.compile(r'[a-zA-Z0-9\p{IsHangul}]+')
}

word_pattern = regex.compile(r'[\p{L}\p{N}]+')
ar_word_pattern = regex.compile(r'[a-zA-Z0-9\u0600-\u06FF]+')
ko_word_pattern = regex.compile(r'[a-zA-Z0-9\p{IsHangul}]+')

# Contractions patterns for different languages
CONTRACTION_PATTERNS = {
    'en': [
        (r"won't", "will not"),
        (r"can't", "cannot"),
        (r"shan't", "shall not"),
        (r"n't", " not"),
        (r"'re", " are"),
        (r"'s", " is"),
        (r"'d", " would"),
        (r"'ll", " will"),
        (r"'ve", " have"),
        (r"'m", " am"),
        (r"let's", "let us"),
        (r"y'all", "you all"),
        (r"ma'am", "madam"),
        (r"gonna", "going to"),
        (r"wanna", "want to"),
        (r"gotta", "got to"),
        (r"ain't", "is not"),
        (r"o'clock", "of the clock")
    ],
    'fr': [
        (r"\bl'", "le "),
        (r"\bd'", "de "),
        (r"\bm'", "me "),
        (r"\bt'", "te "),
        (r"\bs'", "se "),
        (r"\bn'", "ne "),
        (r"\bc'est", "ce est"),
        (r"\bau\b", "à le"),
        (r"\baux\b", "à les"),
        (r"\bdu\b", "de le"),
        (r"\bdes\b", "de les"),
        (r"\bc'était", "ce était"),
        (r"\blorsqu'", "lors que"),
        (r"\bquoiqu'", "quoi que")
    ],
    'es': [
        (r"\bal\b", "a el"),
        (r"\bdel\b", "de el"),
        (r"\bdónde's\b", "dónde es"),
        (r"\bcómo's\b", "cómo es"),
        (r"\bqué's\b", "qué es"),
        (r"\bp'a\b", "para"),
        (r"\bpa'\b", "para"),
        (r"\bpa'l\b", "para el"),
        (r"\bpa'la\b", "para la")
    ],
    'de': [
        (r"\bzur\b", "zu der"),
        (r"\bzum\b", "zu dem"),
        (r"\bim\b", "in dem"),
        (r"\bam\b", "an dem"),
        (r"\bins\b", "in das"),
        (r"\bvom\b", "von dem"),
        (r"\bei'm\b", "bei dem"),
        (r"\büber's\b", "über das"),
        (r"\bauf's\b", "auf das"),
        (r"\bd's\b", "das")
    ],
    'it': [
        (r"\ball'", "a la"),
        (r"\bdell'", "di la"),
        (r"\bun'", "una"),
        (r"\bd'", "di "),
        (r"\bsull'", "su la"),
        (r"\bnell'", "in la"),
        (r"\bell'", "il "),
        (r"\bc'è\b", "che è"),
        (r"\bc'era\b", "che era"),
        (r"\bl'ho\b", "lo ho"),
        (r"\bl'abbiamo\b", "lo abbiamo")
    ]
}
