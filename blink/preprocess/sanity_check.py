import pickle
import os
import tqdm

BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'

custom_dict_path = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'processed', 'dictionary.pickle')
custom_split_dir = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'processed')

original_split_dir = os.path.join(BLINK_ROOT, 'data', 'zeshel', 'blink_format')

with open(custom_dict_path, 'rb') as f1:
    entity_dictionary = pickle.load(f1)

dict_by_type = {}
for e in entity_dictionary:
    if e['type'] not in dict_by_type:
        dict_by_type[e['type']] = set()
    dict_by_type[e['type']].add(e['title'])

for doc_fname in tqdm(os.listdir(custom_split_dir), desc='Loading custom data'):
    assert doc_fname.endswith('.jsonl')
    split_name = doc_fname.split('.')[0]
    with open(os.path.join(custom_split_dir, doc_fname), 'r') as f2:
        with open(os.path.join(original_split_dir, doc_fname), 'r') as f3:
            for idx, line in enumerate(f2):
                custom_mention = json.loads(line.strip())
                original_mention = json.loads(f3.readline().strip())
                assert custom_mention['mention'] == original_mention['mention']
                assert custom_mention['context_left'].lower() == original_mention['context_left'].lower()
                assert custom_mention['context_right'].lower() == original_mention['context_right'].lower()
                assert custom_mention['label'] == original_mention['label']
                assert custom_mention['label_title'] == original_mention['label_title']
                assert custom_mention['type'] == original_mention['world']
                assert custom_mention['label_title'] in dict_by_type[custom_mention['type']]
print('PASS!')
