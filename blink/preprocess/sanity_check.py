import pickle
import os
from tqdm import tqdm
import json
from IPython import embed

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
    if not doc_fname.endswith('.jsonl'):
        continue
    split_name = doc_fname.split('.')[0]
    with open(os.path.join(custom_split_dir, doc_fname), 'r') as f2:
        with open(os.path.join(original_split_dir, doc_fname), 'r') as f3:
            for idx, line in enumerate(tqdm(f2)):
                custom_mention = json.loads(line.strip())
                original_mention = json.loads(f3.readline().strip())
                try:
                    assert custom_mention['mention'].lower().strip() == original_mention['mention'].lower().strip()
                    # assert custom_mention['context_left'].lower().strip() == original_mention['context_left'].lower().strip()
                    # assert custom_mention['context_right'].lower().strip() == original_mention['context_right'].lower().strip()
                    assert custom_mention['label'].lower().strip() == original_mention['label'].lower().strip()
                    assert custom_mention['label_title'].lower().strip() == original_mention['label_title'].lower().strip()
                    assert custom_mention['type'].lower().strip() == original_mention['world'].lower().strip()
                    assert custom_mention['label_title'] in dict_by_type[custom_mention['type']]
                except:
                    print("Original:")
                    print(original_mention)
                    print("Custom:")
                    print(custom_mention)

                    if custom_mention['mention'].lower().strip() != original_mention['mention'].lower().strip():
                        print('Mismatch key: mention')
                    elif custom_mention['context_left'].lower().strip() != original_mention['context_left'].lower().strip():
                        print('Mismatch key: context_left')
                    elif custom_mention['context_right'].lower().strip() != original_mention['context_right'].lower().strip():
                        print('Mismatch key: context_right')
                    elif custom_mention['label'].lower().strip() != original_mention['label'].lower().strip():
                        print('Mismatch key: label')
                    elif custom_mention['label_title'].lower().strip() != original_mention['label_title'].lower().strip():
                        print('Mismatch key: label_title')
                    elif custom_mention['type'].lower().strip() != original_mention['world'].lower().strip():
                        print('Mismatch key: type | world')
                    else:
                        print('label_title not found in dictionary')
                    embed()
                    exit()

print('PASS!')
