import pickle
import os
import json
from IPython import embed

BLINK_ROOT = f'{os.path.abspath(os.path.dirname(__file__))}/../..'

output_file_path = os.path.join(BLINK_ROOT, 'models/trained/medmentions', 'seen_unseen.json')
train_data_path = os.path.join(BLINK_ROOT, 'models/trained/medmentions', 'train_processed_data.pickle')
result_paths = {
    'in-batch (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_blink/eval/wo_type/eval_results_1621123985-directed-1.json'),
    'in-batch (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_blink/eval/wo_type/eval_results_1621123985-undirected-1.json'),
    'knn (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/eval/pos_neg_loss/no_type/wo_type/probe10/eval_results_1621123098-directed-0.json'),
    'knn (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions/eval/pos_neg_loss/no_type/wo_type/probe10/eval_results_1621123098-undirected-0.json'),
    'mst (directed)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type/eval_results_1621123562-directed-2.json'),
    'mst (undirected)': os.path.join(BLINK_ROOT, 'models/trained/medmentions_mst/eval/pos_neg_loss/no_type/wo_type/eval_results_1621123562-undirected-1.json')
}

if __name__ == '__main__':
    seen_unseen_results = {}

    with open(train_data_path, 'rb') as read_handle:
        train_data = pickle.load(read_handle)
    
    seen_cui_ids = set()
    for mention in train_data:
        seen_cui_ids.add(mention['label_cuis'][0])

    n_seen_in_test = 0.
    total_seen_computed = False
    for mode in result_paths:
        print(f"Mode: {mode}")
        n_correct_seen = 0.
        with open(result_paths[mode]) as f:
            results = json.load(f)
        n_queries = results['n_mentions']
        overall_acc = float(results['accuracy'].split(' ')[0]) # Percentage

        for m in results['success']:
            if m['mention_gold_cui'] in seen_cui_ids:
                if not total_seen_computed:
                    n_seen_in_test += 1
                n_correct_seen += 1

        if not total_seen_computed:
            for m in results['failure']:
                if m['mention_gold_cui'] in seen_cui_ids:
                    n_seen_in_test += 1
            total_seen_computed = True

        n_correct_unseen = len(results['success']) - n_correct_seen

        embed()

        seen_acc = (n_correct_seen / n_seen_in_test) * 100
        unseen_acc = (n_correct_unseen / n_seen_in_test) * 100

        seen_unseen_results[mode] = {
            'overall': overall_acc,
            'seen': seen_acc,
            'unseen': unseen_acc
        }

    with open(output_file_path, 'w') as f:
        json.dump(seen_unseen_results, f, indent=2)
        print(f"\nAnalysis saved at: {output_file_path}")
