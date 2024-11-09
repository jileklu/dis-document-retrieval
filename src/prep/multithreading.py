import src.prep.prep_globals as prep_globals
import nltk
from nltk.corpus import wordnet
from konlpy.tag import Okt
# from farasa.segmenter import FarasaSegmenter

def init_worker(_counter, _lock):
    """
    Initializes global variables and resources in each worker process.

    Args:
        _counter (multiprocessing.Value): Shared counter for text processing.
        _lock (multiprocessing.Lock): Shared lock for synchronizing access to shared resources.
    """
    prep_globals.counter = _counter
    prep_globals.lock = _lock

    # Initialize caches for stopwords, stemmers, and lemmatizers
    prep_globals.stopwords_cache = {}
    prep_globals.stemmer_cache = {}
    prep_globals.lemmatizer_cache = {}

    # Initialize the Korean tokenizer
    prep_globals.ko_tokenizer = _initialize_ko_tokenizer()

    # Set NLTK data path and load WordNet
    nltk.data.path.append('/root/nltk_data')
    wordnet.ensure_loaded()

    # Initialize the Farasa segmenter for Arabic
    # prep_globals.farasa_segmenter = _initialize_farasa_segmenter()

def _initialize_ko_tokenizer():
    """
    Initialize the Korean Okt tokenizer.

    Returns:
        Okt or None: An instance of the Okt tokenizer if available; None if the tokenizer could not be imported.
    """
    try:
        return Okt()
    except ImportError:
        print("Could not import konlpy. Korean tokenizer not available.")
        return None

# def _initialize_farasa_segmenter():
#    """
#    Initialize the Farasa segmenter for Arabic.
#
#    Returns:
#        FarasaSegmenter or None: An instance of the Farasa segmenter if available; None if the segmenter could not be imported.
#    """
#    try:
#        return FarasaSegmenter(interactive=False)
#    except ImportError:
#        print("Could not import FarasaSegmenter. Arabic tokenizer not available.")
#        return None
