import os
import json
import pickle
from tqdm import tqdm

BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'

input_dir = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'documents')
output_fpath = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'processed', 'dictionary.pickle')

dictionary = []
label_ids = set()

for doc_fname in tqdm(os.listdir(input_dir), desc='Loading docuemnts'):
    assert doc_fname.endswith('.json')
    entity_type = doc_fname.split('.')[0]
    if entity_type in ['train', 'test', 'val']:
        continue
    with open(os.path.join(input_dir, doc_fname), 'r') as f:
        for idx, line in enumerate(f):
            record = {}
            entity = json.loads(line.strip())
            record["cui"] = entity["document_id"]
            record["title"] = entity["title"]
            record["description"] = entity["text"]
            record["type"] = entity_type
            dictionary.append(record)
            label_ids.add(record["cui"])

assert len(dictionary) == len(label_ids)

print(f"Finished reading {len(dictionary)} entities")
print("Saving entity dictionary...")

with open(output_fpath, 'wb') as write_handle:
    pickle.dump(dictionary, write_handle,
                protocol=pickle.HIGHEST_PROTOCOL)