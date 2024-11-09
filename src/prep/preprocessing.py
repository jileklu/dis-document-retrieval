import os
import sys
import nltk
import regex
import spacy
import string
import unicodedata
import src.prep.prep_globals as prep_globals
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import word_tokenize
from src.prep.prep_utils import get_stemmer, get_lemmatizer
from src.prep.decorators import call_counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'prep'))

def preprocess_texts(texts, lang):
    """
    Preprocess a list of texts sequentially.

    Args:
        texts (list of str): The list of text strings to preprocess.
        lang (str): The language code of the texts (e.g., 'en', 'fr').

    Returns:
        list of lists of str: A list of tokenized and processed text.
    """
    return [preprocess(text, lang) for text in texts]

def preprocess(text, lang):
    """
    Perform a series of preprocessing steps on the input text.

    Args:
        text (str): The text to preprocess.
        lang (str): The language code of the text.

    Returns:
        list of str: The processed text tokens.
    """
    # 1. Expand contractions
    expanded_text = expand_contractions(text, lang)

    # 2. Normalize Unicode
    normalized_text = unicode_normalized_text_nfkd(expanded_text)

    # 3. Lowercase text
    lowercase_text = lowercase(normalized_text, lang)

    # 4. Remove accents and diacritics
    no_accents_text = remove_accents_and_diacritics(lowercase_text)

    # 5. Remove punctuation
    no_punctuation_text = remove_punctuation(no_accents_text, lang)

    # 6. Remove special characters
    clean_text = remove_special_chars(no_punctuation_text, lang)

    # 7. Tokenize text
    tokens = tokenize(clean_text, lang)

    # 8. Remove stopwords
    clean_tokens = remove_stopwords(tokens, lang)

    # 9. Stemming or lemmatization
    stemmed_tokens = stemming_and_lemmatization(clean_tokens, lang)

    return stemmed_tokens

# ------ CORE ------
# Lower casing
def lowercase(text, lang):
    """
    Lowercase the text unless it's in German.

    Args:
        text (str): The input text.
        lang (str): The language code.

    Returns:
        str: The lowercased text (if applicable).
    """
    return text.lower() if lang != 'german' else text


# Tokenization
def tokenize(text, lang):
    """
    Tokenize text based on the language.

    Args:
        text (str): The text to tokenize.
        lang (str): The language code.

    Returns:
        list of str: Tokenized words.
    """
    if lang in prep_globals.LATIN_BASED_LANGS:
        return tokenize_lat_text(text)
    elif lang == 'ko':
        return tokenize_ko_text(text)
    elif lang == 'ar':
        return tokenize_ar_text(text)
    raise ValueError(f'Language {lang} not supported.')

@call_counter
def tokenize_lat_text(text):
    """
    Tokenize Latin-based text.

    Args:
        text (str): The text to tokenize.

    Returns:
        list of str: Tokenized words.
    """
    try:
        return word_tokenize(text)
    except Exception as e:
        print(f"Error tokenizing Latin text: {e}")
        return []

@call_counter
def tokenize_ko_text(text):
    """
    Tokenize Korean text.

    Args:
        text (str): The text to tokenize.

    Returns:
        list of str: Tokenized words.
    """
    # initialize_ko_tokenizer()
    return prep_globals.WORD_PATTERNS['korean'].findall(text)

def initialize_ko_tokenizer():
    """
    Initialize the Korean tokenizer if not already initialized.
    """
    if prep_globals.ko_tokenizer is None:
        with prep_globals.lock:
            if prep_globals.ko_tokenizer is None:
                try:
                    from konlpy.tag import Okt
                    prep_globals.ko_tokenizer = Okt()
                except ImportError:
                    prep_globals.ko_tokenizer = None
                    print("Could not import konlpy.")

@call_counter
def tokenize_ar_text(text):
    """Tokenize Arabic text."""
    # initialize_farasa_segmenter()
    return prep_globals.WORD_PATTERNS['arabic'].findall(text)

def initialize_farasa_segmenter():
    """Initialize the Farasa segmenter for Arabic if not already initialized."""
    if prep_globals.farasa_segmenter is None:
        with prep_globals.lock:
            if prep_globals.farasa_segmenter is None:
                try:
                    # from farasa.segmenter import FarasaSegmenter
                    # globals.farasa_segmenter = FarasaSegmenter(interactive=False)
                    prep_globals.farasa_segmenter = None
                except ImportError:
                    prep_globals.farasa_segmenter = None
                    print("Could not import FarasaSegmenter.")

# ------ HIGH ------

def remove_stopwords(tokens, lang):
    """
    Remove stopwords based on the language.

    Args:
        tokens (list of str): Tokenized words.
        lang (str): Language code.

    Returns:
        list of str: Tokens with stopwords removed.
    """
    if lang in prep_globals.LATIN_BASED_LANGS:
        return remove_stopwords_lat(tokens, lang)
    elif lang == 'ar':
        return remove_stopwords_ar(tokens)
    elif lang == 'ko':
        return remove_stopwords_ko(tokens)
    raise ValueError(f"Unsupported language '{lang}' for stopword removal.")


def remove_stopwords_lat(tokens, lang):
    """
    Remove stopwords for Latin-based languages.

    Args:
        tokens (list of str): Tokenized words.
        lang (str): Language code.

    Returns:
        list of str: Tokens with stopwords removed.
    """
    with prep_globals.lock:
        if lang not in prep_globals.stopwords_cache:
            prep_globals.stopwords_cache[lang] = get_stopwords_for_lang(lang)
    stop_words = prep_globals.stopwords_cache.get(lang, set())
    return [word for word in tokens if word not in stop_words]


def remove_stopwords_ar(tokens):
    """
    Remove stopwords for Arabic.

    Args:
        tokens (list of str): Tokenized words.

    Returns:
        list of str: Tokens with stopwords removed.
    """
    with prep_globals.lock:
        if 'ar' not in prep_globals.stopwords_cache:
            prep_globals.stopwords_cache['ar'] = get_stopwords_for_lang('ar')
    stop_words = prep_globals.stopwords_cache.get('ar', set())
    return [word for word in tokens if word not in stop_words]


def remove_stopwords_ko(tokens):
    """
    Remove stopwords for Korean.

    Args:
        tokens (list of str): Tokenized words.

    Returns:
        list of str: Tokens with stopwords removed.
    """
    with prep_globals.lock:
        if 'ko' not in prep_globals.stopwords_cache:
            prep_globals.stopwords_cache['ko'] = load_korean_stopwords()
    stop_words = prep_globals.stopwords_cache.get('ko', set())
    return [word for word in tokens if word not in stop_words]


def load_korean_stopwords():
    """
    Load Korean stopwords from a file.

    Returns:
        set: A set of Korean stopwords.
    """
    try:
        with open('utils/ko_stopwords.txt', 'r', encoding='utf-8') as file:
            return {line.strip() for line in file if line.strip()}
    except Exception as e:
        print(f"Error loading Korean stopwords: {e}")
        return set()


def get_stopwords_for_lang(lang):
    """
    Retrieve stopwords for the given language.

    Args:
        lang (str): Language code.

    Returns:
        set: Stopwords for the specified language.
    """
    try:
        return set(stopwords.words(prep_globals.LANGS[lang]))
    except LookupError:
        nltk.download('stopwords')
        return set(stopwords.words(prep_globals.LANGS[lang]))
    except Exception as e:
        print(f"Error retrieving stopwords for {lang}: {e}")
        return set()


def remove_special_chars(text, lang):
    """
    Remove special characters based on the language.

    Args:
        text (str): Input text.
        lang (str): Language code.

    Returns:
        str: Text without special characters.
    """
    if lang in prep_globals.LATIN_BASED_LANGS:
        return remove_special_chars_common(text)
    elif lang == 'ko':
        return remove_special_chars_common(text)
    elif lang == 'ar':
        return remove_special_chars_common(text)
    raise ValueError(f'Language {lang} not supported.')


def remove_special_chars_common(text):
    """
    Remove special characters using regex.

    Args:
        text (str): Input text.

    Returns:
        str: Text with special characters removed.
    """
    words = prep_globals.WORD_PATTERNS['general'].findall(text)
    return ' '.join(words)

def remove_punctuation(text, lang):
    """
    Remove punctuation based on the language.

    Args:
        text (str): Input text.
        lang (str): Language code.

    Returns:
        str: Text without punctuation.
    """
    if lang == 'es':
        return remove_punctuation_es(text)
    elif lang in prep_globals.LATIN_BASED_LANGS:
        return remove_punctuation_common(text)
    elif lang == 'ko':
        return remove_punctuation_ko(text)
    elif lang == 'ar':
        return remove_punctuation_ar(text)
    raise ValueError(f"Unsupported language '{lang}' for punctuation removal.")

def remove_punctuation_common(text):
    """
    Remove standard punctuation characters.

    Args:
        text (str): Input text.

    Returns:
        str: Text with punctuation removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def remove_punctuation_es(text):
    """
    Remove punctuation specific to Spanish.

    Args:
        text (str): Input text.

    Returns:
        str: Text with Spanish punctuation removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation + "¿¡"))


def remove_punctuation_ar(text):
    """
    Remove punctuation specific to Arabic.

    Args:
        text (str): Input text.

    Returns:
        str: Text with Arabic punctuation removed.
    """
    arabic_punctuation = "،؟«»"
    return text.translate(str.maketrans('', '', string.punctuation + arabic_punctuation))


def remove_punctuation_ko(text):
    """
    Remove punctuation specific to Korean.

    Args:
        text (str): Input text.

    Returns:
        str: Text with Korean punctuation removed.
    """
    korean_punctuation = "《》〈〉『』「」"
    return text.translate(str.maketrans('', '', string.punctuation + korean_punctuation))


def remove_accents_and_diacritics(text):
    """
    Remove accents and diacritics from the text.

    Args:
        text (str): Input text.

    Returns:
        str: Text with accents and diacritics removed.
    """
    return ''.join(char for char in text if not unicodedata.combining(char))

def stemming_and_lemmatization(tokens, lang):
    """
    Apply stemming or lemmatization based on the language.

    Args:
        tokens (list of str): Tokenized words.
        lang (str): Language code.

    Returns:
        list of str: Stemmed or lemmatized tokens.
    """
    lemmatizer = get_lemmatizer(lang)

    if lemmatizer is None:
        return tokens

    #new_tokens = [stemmer.stem(token) for token in tokens]
    new_tokens = []

    # Process tokens based on the type of lemmatizer returned
    if lang == 'en':
        for token in tokens:
            new_tokens.append(lemmatizer.lemmatize(token))
    elif lang in ['it', 'es', 'de', 'fr']:
        doc = lemmatizer(" ".join(tokens))
        new_tokens = [token.lemma_ for token in doc]
    elif lang in ['ar', 'ko']:
        new_tokens = tokens

    #for token in tokens:
    #    if lang in ['en']:  # NLTK-based lemmatizers (WordNetLemmatizer, ISRIStemmer)
    #        new_tokens.append(lemmatizer.lemmatize(token))
    #    elif lang in ['it', 'es', 'de', 'fr']:  # SpaCy-based lemmatizers
    #        doc = lemmatizer(token)  # Create a SpaCy doc object
    #        new_tokens.append([t.lemma_ for t in doc][0])  # Get the lemma for the token
    #    elif lang in  ['ar','ko']:  # KoNLPy (Kkma) for Korean
    #        # Perform morphological analysis (assumed, you may need to adapt based on Kkma's results)
    #        #morphs = lemmatizer.morphs(token)
    #        new_tokens.append(token)  # Choose the first morph as lemma

    return new_tokens

# Additional high-prio functions
#def query_expansion_and_reformulation(text):
#    # todo
#    # expanded_query + bigrams + trigrams
#    pass

#def remove_special_chars_lat(text):
#    words = prep_globals.WORD_PATTERNS['general'].findall(text)
#    return ' '.join(words)


#def remove_special_chars_ko(text):
#    words = prep_globals.WORD_PATTERNS['korean'].findall(text)
#    return ' '.join(words)


#def remove_special_chars_ar(text):
#    words = prep_globals.WORD_PATTERNS['arabic'].findall(text)
#    return ' '.join(words)

#def dimensionality_reduction(text):
#    # todo
#    pass

# ----- MEDIUM -----

def expand_contractions(text, lang):
    """
    Expand contractions based on the language.

    Args:
        text (str): Input text.
        lang (str): Language code.

    Returns:
        str: Text with expanded contractions.
    """
    if lang not in prep_globals.CONTRACTION_PATTERNS:
        return text
    patterns = prep_globals.CONTRACTION_PATTERNS[lang]
    for pattern, replacement in patterns:
        text = regex.sub(pattern, replacement, text)
    return text

# Additional mid-prio functions
#def handle_numbers_and_non_text(text):
#    # todo
#    pass


#def doc_len_normalization(text):
#    # todo
#    pass


# ------ LOW -------

def unicode_normalized_text_nfc_nkfc(text):
    """
    Normalize text using NFC + NKFC normalization.

    Args:
        text (str): Input text.

    Returns:
        str: NFC + NKFC normalized text.
    """
    text_nfc = unicodedata.normalize('NFC', text)
    return unicodedata.normalize('NFKC', text_nfc)


def unicode_normalized_text_nfc(text):
    """
    Normalize text using NFC normalization.

    Args:
        text (str): Input text.

    Returns:
        str: NFC normalized text.
    """
    return unicodedata.normalize('NFC', text)


def unicode_normalized_text_nfkd(text):
    """
    Normalize text using NFKD normalization.

    Args:
        text (str): Input text.

    Returns:
        str: NFKD normalized text.
    """
    return unicodedata.normalize('NFKD', text)

# Additional low-prio functions
#def text_segmentation(text):
#    # todo
#    pass


#def clustering_for_doc_sim(text):
#    # todo
#    pass
