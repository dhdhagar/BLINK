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


DATA_DIR = '/mnt/nfs/scratch1/rangell/lerac/coref_entity_linking/data/'
DATASET = 'zeshel'

OUTPUT_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/{}/'.format(DATASET)


# get all of the documents
entity2idx = {}
documents = {}
doc_dir = os.path.join(DATA_DIR, DATASET, 'documents')
for doc_fname in tqdm(os.listdir(doc_dir), desc='Loading docuemnts'):
    assert doc_fname.endswith('.json')

    if DATASET == 'zeshel':
        world = doc_fname.split('.')[0]

    with open(os.path.join(doc_dir, doc_fname), 'r') as f:
        for idx, line in enumerate(f):
            one_doc = json.loads(line.strip())
            doc_id = one_doc['document_id']
            one_doc['world'] = world
            documents[doc_id] = one_doc
            entity2idx[doc_id] = idx


# get all of the train mentions
print('Processing mentions...')
blink_mentions = []
split = 'train'
with open(os.path.join(DATA_DIR, DATASET, 'mentions', split + '.json'), 'r') as f:
    for line in f:
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
        #assert ' '.join([context_left, extracted_mention, context_right]) == context_doc['text']

        transformed_mention = {}
        transformed_mention['mention'] = extracted_mention
        transformed_mention['context_left'] = context_left
        transformed_mention['context_right'] = context_right
        transformed_mention['label_id'] = entity2idx[one_mention['label_document_id']]
        transformed_mention['label'] = label_doc['text']
        transformed_mention['label_title'] = label_doc['title']
        if DATASET == 'zeshel':
            transformed_mention['world'] = one_mention['corpus']

        blink_mentions.append(transformed_mention)

print('Done.')


# write all of the transformed train mentions
print('Writing processed mentions to file...')
split = 'train'
with open(os.path.join(OUTPUT_DIR, split + '.jsonl'), 'w') as f:
    f.write('\n'.join([json.dumps(m) for m in blink_mentions]))
print('Done.')
