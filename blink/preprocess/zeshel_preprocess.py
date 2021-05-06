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


DATA_DIR = os.path.join(BLINK_ROOT, 'data', 'zeshel')
OUTPUT_DIR = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'processed')

# get all of the documents
entity2idx = {}
documents = {}
doc_dir = os.path.join(DATA_DIR, 'documents')
for doc_fname in tqdm(os.listdir(doc_dir), desc='Loading docuemnts'):
    assert doc_fname.endswith('.json')
    with open(os.path.join(doc_dir, doc_fname), 'r') as f:
        for idx, line in enumerate(f):
            one_doc = json.loads(line.strip())
            doc_id = one_doc['document_id']
            assert doc_id not in documents
            documents[doc_id] = one_doc


# get all of the train mentions
print('Processing mentions...')

splits = ['train', 'val', 'test']

for split in splits:
    blink_mentions = []
    with open(os.path.join(DATA_DIR, 'mentions', split + '.json'), 'r') as f:
        for line in tqdm(f):
            one_mention = json.loads(line.strip())
            label_doc = documents[one_mention['label_document_id']]
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
            transformed_mention['type'] = one_mention['corpus']
            transformed_mention['label_id'] = one_mention['label_document_id']
            transformed_mention['label'] = label_doc['text']
            transformed_mention['label_title'] = label_doc['title']
            blink_mentions.append(transformed_mention)
    print('Done.')
    # write all of the transformed train mentions
    print('Writing processed mentions to file...')
    with open(os.path.join(OUTPUT_DIR, split + '.jsonl'), 'w') as f:
        f.write('\n'.join([json.dumps(m) for m in blink_mentions]))
    print('Done.')
