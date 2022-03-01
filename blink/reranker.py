
from blink.candidate_ranking.bert_reranking import BertReranker


def get_model(params):
    return BertReranker(params)
