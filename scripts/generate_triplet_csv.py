import csv
from datasets import load_dataset
from tqdm import tqdm
from speedy import *


_load_dataset = imemoize(load_dataset)
def generate_triplet_tsv(output_path, is_toy=False, ignore_query_ids=[]):
    if osp.exists(output_path):
        print(f"File {output_path} already exists. Skipping generation.")
        return
    # Load datasets
    queries = _load_dataset('irds/mmarco_v2_vi_train', 'queries')
    qrels = _load_dataset('irds/mmarco_v2_vi_train', 'qrels')
    triples = _load_dataset('irds/mmarco_v2_vi_train', 'docpairs')
    documents = _load_dataset('irds/mmarco_v2_vi')
    documents_df = documents.to_pandas().set_index('doc_id')
    
    # Create a dictionary for quick lookup of query text by query_id
    query_dict = {record['query_id']: record['text'] for record in queries}

    # Create a dictionary for quick lookup of relevant doc_id by query_id
    relevant_doc_dict = {record['query_id']: record['doc_id'] for record in qrels if record['relevance'] == 1}

    # Open TSV file for writing
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['query', 'relevant_doc', 'non_relevant_doc'])  # Write header

        # If is_toy is True, limit the number of iterations to 100 using tqdm and islice
        triples_iterable = tqdm(triples, total=100) if is_toy else tqdm(triples)

        # Iterate through docpairs to get non-relevant documents
        for triple in triples_iterable:
            query_id = triple['query_id']
            non_relevant_doc_id = triple['doc_id_b']
            if query_id in ignore_query_ids:
                continue
            query_text = query_dict.get(query_id, "")
            relevant_doc_id = relevant_doc_dict.get(query_id, "")
            
            # Assuming doc_id_a is always the non-relevant document

            # Check if we have both relevant and non-relevant docs for the query
            rel_document_text = documents_df.loc[relevant_doc_id, 'text'] if relevant_doc_id else ""
            non_rel_document_text = documents_df.loc[non_relevant_doc_id, 'text'] if non_relevant_doc_id else ""

            if query_text and relevant_doc_id and non_relevant_doc_id:
                writer.writerow([query_text, rel_document_text, non_rel_document_text])

            # If is_toy is True, break the loop after 100 iterations
            if is_toy and triples_iterable.n >= 100:
                break
            
def generate_data(out_data_dir='data/msmarco_vi_toy/',val_frac=0.01, is_toy=True):
    generate_triplet_tsv(f"{out_data_dir}/triplets/raw.tsv", is_toy=is_toy)
    documents = _load_dataset('irds/mmarco_v2_vi')
    qrels = load_dataset('irds/mmarco_v2_vi_train', 'qrels')
    qrels_val = qrels.to_pandas().sample(frac=val_frac, random_state=42)
    qrels_val_0 = qrels_val.groupby('query_id').apply(lambda x: dict(zip(x['doc_id'], x['relevance']))).to_dict()
    dump_json_or_pickle(qrels_val_0, f'{out_data_dir}/qrel/qrel.json')
    documents_val = documents.to_pandas().set_index('doc_id').loc[qrels_val['doc_id'].unique()]
    doc_out_path = f'{out_data_dir}/val_collection/raw.tsv'
    os.makedirs(os.path.dirname(doc_out_path), exist_ok=True)
    documents_val.to_csv(doc_out_path, sep='\t', header=False)
    query_out_path = f'{out_data_dir}/val_queries/raw.tsv'
    os.makedirs(os.path.dirname(query_out_path), exist_ok=True)
    queries_val = load_dataset('irds/mmarco_v2_vi_train', 'queries').to_pandas().set_index('query_id').loc[qrels_val['query_id'].unique()]
    queries_val.to_csv(query_out_path, sep='\t', header=False)


generate_data(out_data_dir='data/msmarco_vi/', is_toy=False, val_frac=0.1)


