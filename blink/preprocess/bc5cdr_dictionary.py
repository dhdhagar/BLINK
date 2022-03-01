import os
import json
import pickle

BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'

input_fpath = os.path.join(BLINK_ROOT, 'data', 'bc5cdr', 'documents', 'entity_documents.json')
output_fpath = os.path.join(BLINK_ROOT, 'data', 'bc5cdr', 'processed', 'dictionary.pickle')

dictionary = []

with open(input_fpath, 'r') as f:
    for idx, line in enumerate(f):
        record = {}
        entity = json.loads(line.strip())
        record["cui"] = entity["document_id"]
        record["title"] = entity["title"]
        record["description"] = entity["text"]
        record["type"] = entity["type"]
        dictionary.append(record)

print(f"Finished reading {idx+1} entities")
print("Saving entity dictionary...")

with open(output_fpath, 'wb') as write_handle:
    pickle.dump(dictionary, write_handle,
                protocol=pickle.HIGHEST_PROTOCOL)