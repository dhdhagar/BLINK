import os
import csv
import json
from collections import defaultdict
from tqdm import tqdm

from pytorch_transformers.tokenization_bert import BertTokenizer

from IPython import embed


PUBTATOR_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/corpus_pubtator.txt'
PRED_PUBTATOR_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/pred_corpus_pubtator.txt'
PRED_MATCHES_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/matches_pred_corpus_pubtator.tsv'
TEST_PMIDS_FILE = '/mnt/nfs/scratch1/rangell/BLINK/tmp/corpus_pubtator_pmids_test.txt'
DATA_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/'
DATASET = 'medmentions'

OUTPUT_DIR = '/mnt/nfs/scratch1/rangell/BLINK/data/{}/taggerOne'.format(DATASET)


if __name__ == '__main__':

    # get tokenizer
    tokenizer = BertTokenizer(
        '../lerac/coref_entity_linking/models/biobert_v1.1_pubmed/vocab.txt',
        do_lower_case=False
    )

    # get all test pmids
    with open(TEST_PMIDS_FILE, 'r') as f:
        test_pmids = set(map(lambda x : x.strip(), f.readlines()))

    # get all of the documents
    raw_docs = defaultdict(str)
    gold_mention_labels = {}
    with open(PUBTATOR_FILE, 'r') as f:
        for line in f:
            line_split = line.split('|')
            if len(line_split) == 3:
                if line_split[0] not in test_pmids:
                    continue
                _text_to_add = ' ' if line_split[1] == 'a' else ''
                _text_to_add += line_split[2].strip()
                raw_docs[line_split[0]] += _text_to_add
            line_split = line.strip().split('\t')
            if len(line_split) == 6:
                if line_split[0] not in test_pmids:
                    continue
                gold_key = (line_split[0],line_split[1], line_split[2])
                gold_mention_labels[gold_key] = line_split[-1].replace('UMLS:', '')

    # get taggerOne predictions
    taggerOne_pred_mention_types = {}
    taggerOne_pred_mention_labels = {}
    with open(PRED_PUBTATOR_FILE, 'r') as f:
        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) == 6:
                if line_split[0] not in test_pmids:
                    continue
                pred_key = (line_split[0],line_split[1], line_split[2])
                taggerOne_pred_mention_types[pred_key] = line_split[4]
                taggerOne_pred_mention_labels[pred_key] = line_split[5].replace('UMLS:', '')

    # tokenize all of the documents
    tokenized_docs = {}
    for pmid, raw_text in raw_docs.items():
        wp_tokens = tokenizer.tokenize(raw_text)
        tokenized_text = ' '.join(wp_tokens).replace(' ##', '')
        tokenized_docs[pmid] = tokenized_text

    # get all of the mentions and their tfidf candidates in raw form
    print('Reading pred mentions and tfidf candidates...')
    pred_mention_cands = defaultdict(list)
    with open(PRED_MATCHES_FILE, 'r') as f:
        reader = csv.reader(f, delimiter="\t", quotechar='"')
        keys = next(reader)
        for row in tqdm(reader):
            if row[0] not in test_pmids:
                continue
            pred_mention_key = (row[0], row[1], row[2])
            pred_mention_cand_val = {k : v for k, v in zip(keys, row)}
            pred_mention_cands[pred_mention_key].append(pred_mention_cand_val)
    print('Done.')
    
    # organize mentions by pmid
    pred_mentions = defaultdict(list)
    for key, value in pred_mention_cands.items():
        pred_mentions[key[0]].append(value[0])

    # sort the predicted mentions in each doc
    for key in list(pred_mentions.keys()):
        pred_mentions[key] = sorted(pred_mentions[key], key=lambda x : int(x['char_start']))
        _mentions = []
        for m in pred_mentions[key]:
            condition = any(
                [((int(m['char_start']) >= int(x['char_start'])
                     and int(m['char_start']) <= int(x['char_end']))
                    or (int(m['char_end']) >= int(x['char_start'])
                     and int(m['char_end']) <= int(x['char_end'])))
                 and (m['char_start'] != x['char_start'] or m['char_end'] != x['char_end'])
                 and (int(m['char_end']) - int(m['char_start']) <= int(x['char_end']) - int(x['char_start']))
                    for x in pred_mentions[key]]
            )
            if not condition:
                _mentions.append(m)
        pred_mentions[key] = _mentions


    # do a token match and get the start offset
    def get_offset(index_list, query_list, start_offset):
        for i in range(start_offset, len(index_list)):
            match = True
            for j, query_elt in enumerate(query_list):
                if query_elt != index_list[i+j]:
                    match = False
                    break
            if match:
                return i
        return -1


    # process mentions
    mention_objs = []
    tfidf_candidate_objs = []
    for pmid, mentions in tqdm(pred_mentions.items(), desc='Process mentions'):
        start_offset = 0
        for i, m in enumerate(mentions):

            # tokenize meniton and expanded mention
            tokenized_mention = tokenizer.tokenize(m['mention'])
            tokenized_mention = ' '.join(tokenized_mention).replace(' ##', '')
            tokenized_mention = tokenized_mention.split()
            tokenized_mention_exp = tokenizer.tokenize(m['mention_exp'])
            tokenized_mention_exp = ' '.join(tokenized_mention_exp).replace(' ##', '')
            tokenized_mention_exp = tokenized_mention_exp.split()

            # do find and replace
            tokenized_doc = tokenized_docs[pmid].split()
            start_index = get_offset(
                tokenized_doc, tokenized_mention, start_offset
            )
            if start_index == -1: # somehow the mention was not found, ignore
                continue
            tokenized_doc = tokenized_doc[:start_index] \
                            + tokenized_mention_exp \
                            + tokenized_doc[start_index+len(tokenized_mention):] 
            end_index = start_index + len(tokenized_mention_exp) - 1
            start_offset = end_index + 1
            assert ' '.join(tokenized_doc[start_index:end_index+1]) == ' '.join(tokenized_mention_exp)

            # create mention object and add to list of mentions
            start_char = m['char_start']
            end_char = m['char_end']
            mention_obj = {
                'mention_id' : '.'.join([pmid, str(i)]),
                'context_document_id' : pmid,
                'start_index' : start_index,
                'end_index' : end_index,
                'text' : ' '.join(tokenized_mention_exp),
                'category': m['mention_tid'],
                'label_document_id': gold_mention_labels.get(
                        (pmid, start_char, end_char), None
                    ),
                'taggerOne_pred_document_id': taggerOne_pred_mention_labels.get(
                        (pmid, start_char, end_char), None
                    ),
                'taggerOne_pred_type': taggerOne_pred_mention_types.get(
                        (pmid, start_char, end_char), None
                    )
            }
            mention_objs.append(mention_obj)

            # get candidates
            tfidf_cand_cuids = []
            for cand in pred_mention_cands[(pmid, start_char, end_char)]:
                tfidf_cand_cuids.append(cand['match_cuid'])

            # create tfidf candidates object and add to list
            tfidf_cands_obj = {
                'mention_id' : '.'.join([pmid, str(i)]),
                'tfidf_candidates' : tfidf_cand_cuids
            }
            tfidf_candidate_objs.append(tfidf_cands_obj)

            tokenized_docs[pmid] = ' '.join(tokenized_doc)

    # write to output files
    with open('taggerOne_test_mentions.jsonl', 'w') as f:
        for m in mention_objs:
            f.write(json.dumps(m) + '\n')

    with open('taggerOne_test_tfidf_candidates.jsonl', 'w') as f:
        for c in tfidf_candidate_objs:
            f.write(json.dumps(c) + '\n')

    with open('taggerOne_test_documents.jsonl', 'w') as f:
        for pmid, doc_text in tokenized_docs.items():
            doc_dict = {
                'document_id' : pmid,
                'title' : pmid,
                'text' : doc_text
            }
            f.write(json.dumps(doc_dict) + '\n')
