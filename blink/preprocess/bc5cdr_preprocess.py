# This file converts zeshel style data into the format BLINK expects
#
# Keys for each mention:
#   - mention
#   - context_left
#   - context_right
#   - label_id
#   - world (zeshel only)
#   - label
#   - label_title

import os
import json
from tqdm import tqdm
from IPython import embed

DATASET = 'bc5cdr'
BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'
DATA_DIR = os.path.join(BLINK_ROOT, 'data')
OUTPUT_DIR = os.path.join(DATA_DIR, DATASET)
SPLIT = 'test'

# get all of the documents
entity2idx = {}
documents = {}
doc_dir = os.path.join(DATA_DIR, DATASET, 'documents')
for doc_fname in tqdm(os.listdir(doc_dir), desc='Processing entities and context'):
    assert doc_fname.endswith('.json')
    with open(os.path.join(doc_dir, doc_fname), 'r') as f:
        for idx, line in enumerate(f):
            one_doc = json.loads(line.strip())
            doc_id = one_doc['document_id']
            documents[doc_id] = one_doc
            entity2idx[doc_id] = idx


# get all of the test mentions
blink_mentions = []
null_count = 0
with open(os.path.join(DATA_DIR, DATASET, 'mentions', SPLIT + '.json'), 'r') as f:
    for line in tqdm(f, desc="Processing mentions"):
        one_mention = json.loads(line.strip())
        context_doc = documents[one_mention['context_document_id']]
        start_index = one_mention['start_index']
        end_index = one_mention['end_index']
        context_tokens = context_doc['text'].split()
        extracted_mention = ' '.join(context_tokens[start_index:end_index+1])
        assert extracted_mention == one_mention['text']
        context_left = ' '.join(context_tokens[:start_index])
        context_right = ' '.join(context_tokens[end_index+1:])
        transformed_mention = {}
        transformed_mention['mention'] = extracted_mention
        transformed_mention['mention_id'] = one_mention['mention_id']
        transformed_mention['context_left'] = context_left
        transformed_mention['context_right'] = context_right
        transformed_mention['context_doc_id'] = one_mention['context_document_id']
        transformed_mention['type'] = one_mention['category']
        transformed_mention['labels'] = []
        for label_document_id in one_mention['label_document_id']:
            try:
                label_doc = documents[label_document_id]
            except:
                null_count += 1
                print(f"label_document_id={label_document_id} not found in entity documents")
                continue
            label = {
                'label': label_doc['text'],
                'label_title': label_doc['title'],
                'label_id': entity2idx[label_document_id],
                'label_umls_cuid': label_document_id
            }
            transformed_mention['labels'].append(label)
        blink_mentions.append(transformed_mention)


# write all of the transformed test mentions
print('\nWriting processed mentions to file...')
output_file = os.path.join(OUTPUT_DIR, 'processed', SPLIT + '.jsonl')
with open(output_file, 'w') as f:
    f.write('\n'.join([json.dumps(m) for m in blink_mentions]))
print(f"{len(blink_mentions)} mentions written to {os.path.abspath(output_file)}")
print(f"{null_count} mentions found with unknown label_document_id")
print('Done!')
