import json
import os
import logging
import time
from collections import defaultdict


def build_inverted_index_on_disk(doc_tf_idf_path, output_index_path='inverted_index.json'):
    """
    Build an inverted index from the document TF-IDF vectors stored in JSON format and save it to disk.

    Args:
        doc_tf_idf_path (str): Path to the JSON file containing document TF-IDF vectors.
        output_index_path (str): Path to save the resulting inverted index JSON file.

    The function processes documents in chunks, creating partial inverted indexes on disk to reduce memory usage.
    These partial indexes are then merged into a single inverted index file saved at `output_index_path`.
    """
    logging.info("Starting inverted index construction...")
    start_time = time.time()
    chunk_size = 50000  # Number of documents to process per chunk
    temp_index_files = []
    term_to_doc = defaultdict(set)
    idx = 0
    chunk_idx = 0
    with open(doc_tf_idf_path, 'r') as f:
        for line in f:
            doc_entry = json.loads(line)
            doc_id = list(doc_entry.keys())[0]
            indices = doc_entry[doc_id]["indices"]
            # For each term index, add document ID to term's set
            for term_idx in indices:
                term_to_doc[term_idx].add(doc_id)
            idx += 1
            if idx % chunk_size == 0:
                # Write the partial inverted index to a temporary file
                temp_file_path = f"temp_index_{chunk_idx}.json"
                with open(temp_file_path, 'w') as temp_file:
                    for term_idx, doc_ids in term_to_doc.items():
                        temp_file.write(json.dumps({term_idx: list(doc_ids)}) + '\n')
                temp_index_files.append(temp_file_path)
                term_to_doc.clear()
                chunk_idx += 1
                logging.info(f"Processed {idx} documents. Partial inverted index saved to {temp_file_path}.")
    # Write any remaining terms
    if term_to_doc:
        temp_file_path = f"temp_index_{chunk_idx}.json"
        with open(temp_file_path, 'w') as temp_file:
            for term_idx, doc_ids in term_to_doc.items():
                temp_file.write(json.dumps({term_idx: list(doc_ids)}) + '\n')
        temp_index_files.append(temp_file_path)
        term_to_doc.clear()
        logging.info(f"Processed {idx} documents. Final partial inverted index saved to {temp_file_path}.")
    # Now merge the partial inverted indexes
    logging.info("Merging partial inverted indexes...")
    term_to_doc_merged = defaultdict(set)
    for temp_file_path in temp_index_files:
        with open(temp_file_path, 'r') as temp_file:
            for line in temp_file:
                term_entry = json.loads(line)
                term_idx = list(term_entry.keys())[0]
                doc_ids = term_entry[term_idx]
                term_to_doc_merged[term_idx].update(doc_ids)
        # Remove temporary file to save space
        os.remove(temp_file_path)
        term_to_doc.clear()
        logging.info(f"Merged and removed temporary file {temp_file_path}.")

    # Write the final inverted index
    with open(output_index_path, 'w') as output_file:
        for term_idx, doc_ids in term_to_doc_merged.items():
            output_file.write(json.dumps({term_idx: list(doc_ids)}) + '\n')

    end_time = time.time()
    logging.info("Inverted index building complete and saved to disk.")
    logging.info(f"Inverted index saved to {output_index_path}.")
    logging.info(f"Inverted index construction took {end_time - start_time:.2f} seconds.")