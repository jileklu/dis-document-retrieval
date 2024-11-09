import json
import pickle

import math
import logging
import os
from collections import defaultdict
import time
from scipy.sparse import csr_matrix
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from src.prep.preprocessing import preprocess_texts

# Initialize logger
logger = logging.getLogger(__name__)

# Global variables
GLOBAL_IDF_DICT = {}
GLOBAL_TERM_TO_INDEX = {}

def init_worker(idf_dict, term_to_index):
    """
    Initialize worker processes with shared global variables.

    Args:
        idf_dict (dict): IDF dictionary mapping terms to their inverse document frequencies.
        term_to_index (dict): Dictionary mapping terms to unique indices.
    """
    global GLOBAL_IDF_DICT
    global GLOBAL_TERM_TO_INDEX
    GLOBAL_IDF_DICT = idf_dict
    GLOBAL_TERM_TO_INDEX = term_to_index

def compute_smoothed_tf(tokens):
    """
    Compute term frequency with log scaling for each token in the list of tokens.

    Args:
        tokens (list): List of tokens in the document.

    Returns:
        dict: Dictionary of term frequencies with log scaling applied.
    """
    tf_dict = {}
    for term in tokens:
        tf_dict[term] = tf_dict.get(term, 0) + 1
    for term in tf_dict:
        tf_dict[term] = 1 + math.log(tf_dict[term])  # Smoothing
    return tf_dict


def compute_smoothed_idf(corpus_terms_set, num_docs):
    """
    Compute smoothed inverse document frequency for terms across the entire corpus.

    Args:
        corpus_terms_set (dict): Dictionary of terms with document frequency counts.
        num_docs (int): Total number of documents in the corpus.

    Returns:
        dict: Dictionary of smoothed IDF values for each term.
    """
    idf_dict = {}
    for term, doc_freq in corpus_terms_set.items():
        idf_dict[term] = math.log((1 + num_docs) / (1 + doc_freq)) + 1
    return idf_dict


def compute_tf_idf_vector(tokens, idf_dict, term_to_index):
    """
    Compute normalized TF-IDF vector and store it as a sparse vector.

    Args:
        tokens (list): List of tokens in the document.
        idf_dict (dict): Inverse document frequency values for terms.
        term_to_index (dict): Mapping of terms to index positions.

    Returns:
        csr_matrix: Sparse TF-IDF vector for the document.
    """
    tf = compute_smoothed_tf(tokens)
    indices = []
    data = []
    for term, tf_value in tf.items():
        if term in term_to_index:
            idx = term_to_index[term]
            idf_value = idf_dict.get(term, 0)
            tf_idf_value = tf_value * idf_value
            indices.append(idx)
            data.append(tf_idf_value)

    # Create sparse vector
    vector_size = len(term_to_index)
    tf_idf_vector = csr_matrix((data, indices, [0, len(indices)]), shape=(1, vector_size))

    # Normalize the vector
    norm = np.linalg.norm(tf_idf_vector.data)
    if norm > 0:
        tf_idf_vector.data = tf_idf_vector.data / norm
    return tf_idf_vector


def calculate_idf_from_corpus(corpus_path, vocab_output_path='tf-idf/vocab.json', idf_output_path='tf-idf/idf_dict.pkl'):
    """
    First pass over the corpus to calculate document frequencies for each term.

    Args:
        corpus_path (str): Path to the corpus file.
        vocab_output_path (str): Path to save the vocabulary JSON file.
        idf_output_path (str): Path to save the inverse document frequency pickle file.
    Returns:
        tuple: A tuple with the IDF dictionary and term-to-index mapping.
    """
    logging.info("Starting IDF calculation from corpus...")
    start_time = time.time()
    corpus_terms_set = {}
    num_docs = 0
    term_to_index = {}
    index = 0
    with open(corpus_path, 'r') as f:
        for idx, line in enumerate(f):
            doc = json.loads(line)
            tokens = set(doc['tokens'])
            num_docs += 1
            for term in tokens:
                if term not in corpus_terms_set:
                    corpus_terms_set[term] = 1
                    term_to_index[term] = index
                    index += 1
                else:
                    corpus_terms_set[term] += 1
            if idx % 50000 == 0 and idx > 0:
                logging.info(f"Processed {idx} documents...")
    idf_dict = compute_smoothed_idf(corpus_terms_set, num_docs)

    # Ensure the directories exist for the output files
    os.makedirs(os.path.dirname(vocab_output_path), exist_ok=True)
    os.makedirs(os.path.dirname(idf_output_path), exist_ok=True)

    # Save the vocabulary as JSON
    with open(vocab_output_path, 'w') as vocab_file:
        json.dump(term_to_index, vocab_file)

    # Save the IDF dictionary as a pickle file
    with open(idf_output_path, 'wb') as idf_file:
        pickle.dump(idf_dict, idf_file)

    end_time = time.time()
    logging.info("IDF calculation complete.")
    logging.info(f"IDF calculation took {end_time - start_time:.2f} seconds.")
    return idf_dict, term_to_index

def process_document_tf_idf(line, idf_dict, term_to_index):
    """
    Process a single document to compute its TF-IDF vector.

    Args:
        line (str): JSON line of a document with `tokens` and `doc_id`.
        idf_dict (dict): IDF dictionary for terms.
        term_to_index (dict): Term-to-index mapping.

    Returns:
        str: JSON string with the document ID and its sparse TF-IDF vector.
    """
    doc = json.loads(line)
    doc_id = doc['doc_id']
    tokens = doc['tokens']
    tf_idf_vector = compute_tf_idf_vector(tokens, idf_dict, term_to_index)
    # Save the vector as indices and data
    indices = tf_idf_vector.indices.tolist()
    data = tf_idf_vector.data.tolist()
    return json.dumps({doc_id: {"indices": indices, "data": data}})

def wrapper_process_document_tf_idf(args):
    """
    Wrapper function to unpack arguments for multiprocessing.

    Args:
        args (tuple): Arguments for `process_document_tf_idf` function.

    Returns:
        str: Result from `process_document_tf_idf`.
    """
    return process_document_tf_idf(*args)

def compute_tf_idf_for_documents(corpus_path, idf_dict, term_to_index, output_path='doc_tf_idf_vectors.json'):
    """
    Compute TF-IDF vectors for all documents.

    Args:
        corpus_path (str): Path to the corpus file.
        idf_dict (dict): IDF dictionary for terms.
        term_to_index (dict): Term-to-index mapping.
        output_path (str): Path to save document TF-IDF vectors.
    """
    logging.info("Starting TF-IDF computation for documents...")
    start_time = time.time()
    with open(output_path, 'w') as output_file, open(corpus_path, 'r') as f:
        total_docs = sum(1 for _ in f)  # Count total documents
        f.seek(0)  # Reset file pointer to beginning
        logging.info(f"Total documents to process: {total_docs}")
        with ProcessPoolExecutor() as executor:
            # Use a generator to avoid loading all lines into memory
            args = ((line, idf_dict, term_to_index) for line in f)
            results = executor.map(wrapper_process_document_tf_idf, args, chunksize=1000)
            for idx, result in enumerate(results, 1):
                output_file.write(result + '\n')
                if idx % 50000 == 0:
                    logging.info(f"Processed {idx}/{total_docs} documents.")
    end_time = time.time()
    logging.info(f"TF-IDF computation for documents complete. Took {end_time - start_time:.2f} seconds.")
    logging.info(f"TF-IDF vectors saved to {output_path}.")

def process_query_tf_idf(line):
    """
    Process a single query to compute its TF-IDF vector using global IDF and term mappings.

    Args:
        line (str): JSON line representing a query with `tokens` and `query_id`.

    Returns:
        str: JSON string with query ID and its sparse TF-IDF vector.
    """
    query = json.loads(line)
    query_id = query['query_id']
    tokens = query['tokens']
    tf_idf_vector = compute_tf_idf_vector(tokens, GLOBAL_IDF_DICT, GLOBAL_TERM_TO_INDEX)
    # Save the vector as indices and data
    indices = tf_idf_vector.indices.tolist()
    data = tf_idf_vector.data.tolist()
    return json.dumps({query_id: {"indices": indices, "data": data}})

def wrapper_process_query_tf_idf(line):
    """
    Wrapper function to process a query line.

    Args:
        line (str): JSON line for a query to process.

    Returns:
        str: Result from `process_query_tf_idf`.
    """
    return process_query_tf_idf(line)

def compute_tf_idf_for_queries(queries_path, idf_dict, term_to_index, output_path='query_tf_idf_vectors.json'):
    """
    Compute TF-IDF vectors for all queries using multiprocessing with shared IDF and term mappings.

    Args:
        queries_path (str): Path to the queries file.
        idf_dict (dict): IDF dictionary for terms.
        term_to_index (dict): Term-to-index mapping.
        output_path (str): Path to save the computed query TF-IDF vectors.
    """
    logging.info("Starting TF-IDF computation for queries...")
    start_time = time.time()
    with open(output_path, 'w') as output_file, open(queries_path, 'r') as f:
        # Generator to read lines one by one
        lines = f
        total_queries = sum(1 for _ in f)
        f.seek(0)  # Reset file pointer after counting
        logging.info(f"Total queries to process: {total_queries}")
        with ProcessPoolExecutor(initializer=init_worker, initargs=(idf_dict, term_to_index), max_workers=os.cpu_count()) as executor:
            # Process queries with a reasonable chunksize
            results = executor.map(wrapper_process_query_tf_idf, lines, chunksize=100)
            for idx, result in enumerate(results, 1):
                output_file.write(result + '\n')
                if idx % 100 == 0 or idx == total_queries:
                    logging.info(f"Processed {idx}/{total_queries} queries.")
    end_time = time.time()
    logging.info(f"TF-IDF computation for queries complete. Took {end_time - start_time:.2f} seconds.")
    logging.info(f"TF-IDF vectors saved to {output_path}.")
