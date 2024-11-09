import json
import spacy
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import src.prep.prep_globals as prep_globals

def load_corpus_file(file_path):
    """
    Load and return a corpus from a JSON file.

    Args:
        file_path (str): The path to the JSON file containing the corpus.

    Returns:
        dict: The loaded corpus as a dictionary.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_stemmer(lang):
    """
    Retrieve or initialize a stemmer for the given language.

    Args:
        lang (str): The language code (e.g., 'en' for English, 'fr' for French).

    Returns:
        SnowballStemmer or None: The initialized stemmer for the language, or None if unsupported.
    """
    with prep_globals.lock:
        if lang not in prep_globals.stemmer_cache:
            prep_globals.stemmer_cache[lang] = _initialize_stemmer(lang)
    return prep_globals.stemmer_cache[lang]

def _initialize_stemmer(lang):
    """
    Helper function to initialize a stemmer for the given language.

    Args:
        lang (str): The language code for which to initialize the stemmer.

    Returns:
        SnowballStemmer or None: A stemmer for the language, or None if unsupported.
    """
    stemmers = {
        'en': SnowballStemmer("english"),
        'it': SnowballStemmer("italian"),
        'es': SnowballStemmer("spanish"),
        'de': SnowballStemmer("german"),
        'fr': SnowballStemmer("french"),
        'ar': None  # Use ISRIStemmer() if Arabic support is added
    }
    return stemmers.get(lang)  # Returns None for unsupported languages

def _initialize_lemmatizer(lang):
    """
    Helper function to initialize a lemmatizer for the given language.

    Args:
        lang (str): The language code for which to initialize the lemmatizer.

    Returns:
        Lemmatizer or None: A lemmatizer for the language, or None if unsupported.
    """
    lemmatizers = {
        'en': WordNetLemmatizer(),
        'it': spacy.load("it_core_news_sm"),
        'es': spacy.load("es_core_news_sm"),
        'de': spacy.load("de_core_news_sm"),
        'fr': spacy.load("fr_core_news_sm"),
        'ar': None,  # Add Arabic lemmatizer if needed
        'ko': None   # Add Korean lemmatizer if needed
    }
    return lemmatizers.get(lang, None)

def get_lemmatizer(lang):
    """
    Retrieve or initialize a lemmatizer for the given language.

    Args:
        lang (str): The language code (e.g., 'en' for English, 'fr' for French).

    Returns:
        Lemmatizer or None: The initialized lemmatizer for the language, or None if unsupported.
    """
    if lang not in prep_globals.lemmatizer_cache:
        with prep_globals.lock:
            if lang not in prep_globals.lemmatizer_cache:
                prep_globals.lemmatizer_cache[lang] = _initialize_lemmatizer(lang)
    return prep_globals.lemmatizer_cache[lang]
  # Returns None for unsupported languages

# Commented code retained for potential future use

# from konlpy.tag import Kkma
# from nltk.stem.isri import ISRIStemmer

# def get_lemmatizer(lang):
#    """Retrieve or initialize a lemmatizer for the given language."""
#    with prep_globals.lock:
#        if lang not in prep_globals.lemmatizer_cache:
#            prep_globals.lemmatizer_cache[lang] = _initialize_lemmatizer(lang)
#    return prep_globals.lemmatizer_cache[lang]

# def _initialize_lemmatizer(lang):
#    try:
#        lemmatizers = {
#            'en': WordNetLemmatizer(),
#            'it': spacy.load("it_core_news_sm"),
#            'es': spacy.load("es_core_news_sm"),
#            'de': spacy.load("de_core_news_sm"),
#            'fr': spacy.load("fr_core_news_sm"),
#            'ar': None,  # ISRIStemmer() can be used if needed
#            'ko': None   # Kkma() can be used if needed
#        }
#        return lemmatizers.get(lang, None)  # Return None for unsupported languages
#    except ValueError:
#        return None
