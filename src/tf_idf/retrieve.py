import json
import logging
import time
from collections import defaultdict
from scipy.sparse import csr_matrix, vstack
import numpy as np

def build_term_to_queries_mapping_top_terms(queries, K):
    """
    Build a mapping from term index to the list of query IDs that contain the term,
    considering only the top K terms per query.

    Args:
        queries (dict): A dictionary where keys are query IDs and values are sparse query vectors.
        K (int): The number of top terms to consider for each query.

    Returns:
        tuple:
            - term_to_queries (defaultdict): Mapping of term indices to lists of query IDs containing those terms.
            - query_top_terms (dict): Mapping of query IDs to sets of their top K term indices.
    """
    term_to_queries = defaultdict(list)
    query_top_terms = {}
    for query_id, query_vector in queries.items():
        indices = query_vector.indices
        data = query_vector.data
        if len(data) > K:
            top_indices = np.argsort(data)[-K:]
            top_terms = indices[top_indices]
        else:
            top_terms = indices
        query_top_terms[query_id] = set(top_terms)
        for term_idx in top_terms:
            term_to_queries[term_idx].append(query_id)
    return term_to_queries, query_top_terms

def retrieve_top_documents_with_limited_candidates(query_tf_idf_path, doc_tf_idf_path, index_path, term_to_index_path, top_n=10, K=5, M=200):
    """
    Retrieve the top N documents for each query using an inverted index with limited candidates.

    Args:
        query_tf_idf_path (str): Path to the JSON file containing query TF-IDF vectors.
        doc_tf_idf_path (str): Path to the JSON file containing document TF-IDF vectors.
        index_path (str): Path to the inverted index JSON file.
        term_to_index_path (str): Path to the JSON file with term-to-index mappings.
        top_n (int): The number of top documents to retrieve per query.
        K (int): The number of top terms to consider per query for matching candidates.
        M (int): The maximum number of documents to consider per term.

    Returns:
        list: A list of dictionaries where each dictionary contains the query ID and its top retrieved document IDs.
    """
    logging.info("Starting retrieval of top documents for each query...")
    start_time = time.time()

    # Load term_to_index mapping
    with open(term_to_index_path, 'r') as f:
        term_to_index = json.load(f)
    logging.info("Term to index mapping loaded.")

    # Load query vectors and get top K terms
    queries = {}
    query_ids_list = []
    with open(query_tf_idf_path, 'r') as f:
        for line in f:
            query_entry = json.loads(line)
            query_id = list(query_entry.keys())[0]
            data = query_entry[query_id]
            indices = np.array(data['indices'])
            values = np.array(data['data'])
            vector_size = len(term_to_index)
            query_vector = csr_matrix((values, indices, [0, len(indices)]), shape=(1, vector_size))
            queries[query_id] = query_vector
            query_ids_list.append(query_id)
    logging.info("Query vectors loaded.")

    # Build term_to_queries mapping considering top K terms per query
    term_to_queries, query_top_terms = build_term_to_queries_mapping_top_terms(queries, K)
    logging.info("Term to queries mapping for top terms created.")

    # Initialize candidate document lists
    query_candidates = defaultdict(set)
    all_candidate_doc_ids = set()

    # Iterate through the inverted index and assign documents to queries
    with open(index_path, 'r') as index_file:
        for idx, line in enumerate(index_file, 1):
            term_entry = json.loads(line)
            term_idx_str = list(term_entry.keys())[0]
            doc_ids = term_entry[term_idx_str]
            term_idx = int(term_idx_str)

            # Retrieve queries that contain this term in their top K terms
            relevant_queries = term_to_queries.get(term_idx, [])
            if relevant_queries:
                # Limit the number of documents per term to M
                limited_doc_ids = doc_ids[:M]
                for query_id in relevant_queries:
                    query_candidates[query_id].update(limited_doc_ids)
                    all_candidate_doc_ids.update(limited_doc_ids)

            # Optional: Log progress every 1,000,000 terms
            if idx % 1000000 == 0:
                logging.info(f"Processed {idx} terms from the inverted index.")

    logging.info("Candidate documents collected for queries.")

    # Build doc_vectors mapping for candidate documents
    doc_vectors = {}
    with open(doc_tf_idf_path, 'r') as doc_file:
        for line in doc_file:
            doc_entry = json.loads(line)
            doc_id = list(doc_entry.keys())[0]
            if doc_id in all_candidate_doc_ids:
                data = doc_entry[doc_id]
                indices = data['indices']
                values = data['data']
                vector_size = len(term_to_index)
                doc_vector = csr_matrix((values, indices, [0, len(indices)]), shape=(1, vector_size))
                doc_vectors[doc_id] = doc_vector
    logging.info("Document vectors for candidates loaded.")

    # Now computing similarities for each query...
    submission_data = []
    total_queries = len(query_ids_list)

    for idx, query_id in enumerate(query_ids_list):
        candidate_doc_ids = query_candidates.get(query_id, set())
        query_vector = queries[query_id]
        result = process_single_query(query_id, query_vector, candidate_doc_ids, doc_vectors, top_n)
        submission_data.append({
            'id': idx,
            'docids': result['docids']
        })
        if (idx + 1) % 100 == 0 or (idx + 1) == total_queries:
            logging.info(f"Processed {idx + 1}/{total_queries} queries.")

    logging.info("Top documents retrieved for all queries.")
    end_time = time.time()
    logging.info(f"Retrieval completed in {end_time - start_time:.2f} seconds.")
    return submission_data

def process_single_query(query_id, query_vector, candidate_doc_ids, doc_vectors, top_n):
    """
    Process a single query to retrieve its top N most relevant documents based on cosine similarity.

    Args:
        query_id (str): Unique identifier for the query.
        query_vector (csr_matrix): TF-IDF vector representation of the query.
        candidate_doc_ids (set): Set of document IDs to consider for retrieval.
        doc_vectors (dict): Dictionary mapping document IDs to their TF-IDF vectors.
        top_n (int): Number of top documents to retrieve.

    Returns:
        dict: Dictionary with the `query_id` and list of its top retrieved document IDs.
    """
    candidate_doc_ids_list = list(candidate_doc_ids)
    docs = []
    doc_ids_list = []
    for doc_id in candidate_doc_ids_list:
        doc_vector = doc_vectors.get(doc_id)
        if doc_vector is not None:
            docs.append(doc_vector)
            doc_ids_list.append(doc_id)
    if not docs:
        retrieved_doc_ids = []
    else:
        doc_matrix = vstack(docs)
        similarities = doc_matrix.dot(query_vector.T).toarray().flatten()
        # Use argpartition to get indices of top N similarities
        top_n = min(top_n, len(similarities))
        top_n_indices = np.argpartition(-similarities, top_n - 1)[:top_n]
        top_docs = [(similarities[i], doc_ids_list[i]) for i in top_n_indices]
        # Sort the top documents by similarity
        top_docs.sort(reverse=True)
        retrieved_doc_ids = [doc_id for sim, doc_id in top_docs]
    formatted_doc_ids = str(retrieved_doc_ids)
    return {'docids': formatted_doc_ids}