import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset  # J Used for wrapping tensors into a dataset
from blink.biencoder.zeshel_utils import world_to_id  # D Not used
from blink.common.params import (
    ENT_START_TAG,
    ENT_END_TAG,
    ENT_TITLE_TAG,
)  # D Can be created here, no need to import
from torch.utils.data import DataLoader, SequentialSampler
import math
import faiss  # C FAISS (Facebook AI Similarity Search) primarily used for efficient similarity search and clustering of dense vectors.

from IPython import embed


def select_field(data, key1, key2=None):
    """
    Description
    -----------
    #J Extracts specific data from a LIST of dictionaries (key1) or a list of nested dictionaries (key2)

    Parameters
    ----------
    data :
        List of dictionaries
    key1 :
        First key to look up in each dictionary
    key2 :
        Optional second key
    """
    if key2 is None:
        return [example[key1] for example in data]
    else:
        return [example[key1][key2] for example in data]


def get_context_representation(
    sample,
    tokenizer,
    max_seq_length,
    mention_key="mention",
    context_key="context",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
):
    """
    Description
    -----------
    Prepare the text data for input into a transformer-based model like BERT.
    Given a sample of text, it uses a tokenizer to convert the text into a format that is compatible with these models,
    which typically require input to be in a specific tokenized format with special tokens and padding.

    Parameters
    ----------
    sample : #D1 list of dics
        Each dict represents a text sample with mention and context. (And we have plenty of them)
    tokenizer : Object
        Def : Converts text into token (smaller chunk of text such as subwords)
    max_seq_length : int
        Maximum length of the token sequence for the model input.
    mention_key : str
        Key in the sample dictionary that holds the text of the mention
    context_key : str
        Key used to access the surrounding context of the mention in the sample dictionary.
    ent_start_token :
        Token before the mention to indicate its start.
    ent_end_token :
        Token that is added after the mention to indicate its end

    -------
    Returns {tokens + ID} of "[CLS] c_left [START] mention [END] c_right [SEP]"
    """
    mention_tokens = []
    if (
        sample[mention_key] and len(sample[mention_key]) > 0
    ):  # redundant : one condition would have been enough
        mention_tokens = tokenizer.tokenize(sample[mention_key])
        mention_tokens = [ent_start_token] + mention_tokens + [ent_end_token]

    # Tokenization of context
    context_left = sample[context_key + "_left"]  # = sample["context_left"]
    context_left = tokenizer.tokenize(context_left)  # tokenize the sentences
    context_right = sample[context_key + "_right"]  # = sample["context_right"]
    context_right = tokenizer.tokenize(context_right)

    # D2 Quota Calculation for left and right context
    left_quota = (max_seq_length - len(mention_tokens)) // 2 - 1
    right_quota = max_seq_length - len(mention_tokens) - left_quota - 2
    left_add = len(context_left)
    right_add = len(context_right)
    if left_add <= left_quota:
        if right_add > right_quota:
            right_quota += left_quota - left_add
    else:
        if right_add <= right_quota:
            left_quota += right_quota - right_add

    context_tokens = (
        context_left[-left_quota:] + mention_tokens + context_right[:right_quota]
    )
    context_tokens = ["[CLS]"] + context_tokens + ["[SEP]"]

    # Conversion to IDs and Padding
    # The context tokens are converted to IDs, and then padding is added to ensure the sequence length is equal to max_seq_length.
    input_ids = tokenizer.convert_tokens_to_ids(context_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    # Output : Returns the processed tokens and their corresponding IDs in a dictionary.
    return {
        "tokens": context_tokens,
        "ids": input_ids,
    }


def get_candidate_representation(
    candidate_desc,
    tokenizer,
    max_seq_length,
    candidate_title=None,
    title_tag=ENT_TITLE_TAG,
):
    """
    Description
    -----------
    Take a description of a candidate entity and convert it into a format suitable for input into a transformer-based model (= tokens + ID).

    Parameters
    ----------
    candidate_desc: str
        Description of the candidate entity
    tokenizer : Object
        Def : Converts text into token (smaller chunk of text such as subwords)
    max_seq_length : int
        Maximum length of the token sequence for the model input.
    candidate_title : str, optional
        Title of the candidate entity.
    title_tag : str
        Special token added to the token sequence to separate the candidate_title from the candidate_desc
    -------
    Returns {tokens + ID} of "[CLS] e_title [TITLE] e_desc [SEP]"
    """
    # Retrieves the special [CLS] and [SEP] tokens used by the BERT tokenizer (can also be from other models like RoBERTa, DistilBERT, etc...).
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    # Candidate description is tokenized
    cand_tokens = tokenizer.tokenize(candidate_desc)

    # If a title is provided, tokenizes it, and then appropriately integrates it with the description tokens, adding a title_tag between them
    if candidate_title is not None:
        title_tokens = tokenizer.tokenize(candidate_title)
        if len(title_tokens) <= len(cand_tokens):
            cand_tokens = (
                title_tokens
                + [title_tag]
                + cand_tokens[
                    (
                        0
                        if title_tokens != cand_tokens[: len(title_tokens)]
                        else len(title_tokens)
                    ) :
                ]
            )  # Filter title from description
        else:
            cand_tokens = title_tokens + [title_tag] + cand_tokens

    cand_tokens = cand_tokens[: max_seq_length - 2]
    cand_tokens = [cls_token] + cand_tokens + [sep_token]

    # Token to ID Conversion and Padding
    input_ids = tokenizer.convert_tokens_to_ids(cand_tokens)
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    assert len(input_ids) == max_seq_length

    # Output : Returns the processed tokens and their corresponding IDs
    return {
        "tokens": cand_tokens,
        "ids": input_ids,
    }


def process_mention_data(
    samples,
    entity_dictionary,
    tokenizer,
    max_context_length,
    max_cand_length,
    silent,
    knn,
    dictionary_processed=False,
    mention_key="mention",
    context_key="context",
    label_key="label",
    multi_label_key=None,
    title_key="label_title",
    label_id_key="label_id",
    ent_start_token=ENT_START_TAG,
    ent_end_token=ENT_END_TAG,
    title_token=ENT_TITLE_TAG,
    debug=False,
    logger=None,
):
    """
    Description #DD2
    -----------
    Process the raw text mention data to a tokenized version and packages it into a form that can be directly used for training or inference in NLP models

    Parameters
    ----------
    samples : list of dicts
        Each dict represents a text sample with mention and context.
    entity_dictionary : dict
        A dictionary of entities with their descriptions, titles, etc.
    tokenizer : object
        A tokenizer object for converting text to tokens (like from BERT).
    max_context_length : int
        Maximum length of tokenized context.
    max_cand_length : int
        Maximum length for tokenized candidate entities.
    silent : bool
        If True, suppresses progress output.
    knn : int
        Number of nearest neighbors to consider in processing.
    dictionary_processed : bool, optional:
        Indicates if entity dictionary is already processed.
    mention_key, context_key, label_key, label_id_key : str
        Keys to access respective elements in samples.
    multi_label_key str, optional:
        Key for multiple labels, if applicable.
    title_key : str
        Key for accessing entity titles in the dictionary.
    ent_start_token, ent_end_token, title_token : str
        Special tokens used in tokenization.
    debug : bool, optional
        If True, processes a subset for debugging.
    logger : object, optional
        Logger object for logging information.

    -------
    Returns the processed samples (=mentions), updated entity dictionary (=tokenized entities), and tensor dataset (wrapping tensor into a dataset)
    """

    # Entity Dictionary Processing :
    # Tokenize entities from the entity dictionary if not already processed
    processed_samples = []
    dict_cui_to_idx = {}
    for idx, ent in enumerate(tqdm(entity_dictionary, desc="Tokenizing dictionary")):
        dict_cui_to_idx[ent["cui"]] = idx
        if not dictionary_processed:
            label_representation = get_candidate_representation(
                ent["description"], tokenizer, max_cand_length, ent["title"]
            )
            entity_dictionary[idx]["tokens"] = label_representation["tokens"]
            entity_dictionary[idx]["ids"] = label_representation["ids"]

    # Debugging test
    if debug:
        samples = samples[:200]

    iter_ = samples

    # Sample (= Mention data) Processing :
    # Processes each sample, tokenizing its context and resolving its labels to indices in the entity dictionary
    for idx, sample in enumerate(iter_):
        context_tokens = get_context_representation(
            sample,
            tokenizer,
            max_context_length,
            mention_key,
            context_key,
            ent_start_token,
            ent_end_token,
        )

        labels, record_labels, record_cuis = [sample], [], []
        # If the multi_label_key is provided (meaning samples can have multiple labels),
        # labels is set to the list of labels from the sample corresponding to this key.
        if multi_label_key is not None:
            labels = sample[multi_label_key]

        not_found_in_dict = False
        for l in labels:
            label = l[label_key]
            label_idx = l[label_id_key]
            if label_idx not in dict_cui_to_idx:
                not_found_in_dict = True
                break
            record_labels.append(dict_cui_to_idx[label_idx])
            record_cuis.append(label_idx)

        if not_found_in_dict:
            continue

        record = {
            "mention_id": sample.get("mention_id", idx),
            "mention_name": sample["mention"],  # The actual text of the mention
            "context": context_tokens,  # Contains the tokenized version of (mention + surrounding context + special tokens)
            "n_labels": len(
                record_labels
            ),  # Number of labels associated with the mention
            # An array of indices pointing to the entities in the entity dictionary that are considered correct labels for the mention.
            # This array is padded with -1 to reach a length of knn, which facilitates operations that consider a fixed number of potential labels for each mention
            "label_idxs": record_labels + [-1] * (knn - len(record_labels)),
            "label_cuis": record_cuis,  # CUIs associated with the correct labels for the mention
            "type": sample["type"],  # Represents the type of entity or mention
        }

        processed_samples.append(record)

    # Optionally logs processed samples for inspection
    if logger:
        logger.info("====Processed samples: ====")
        for sample in processed_samples[:5]:
            logger.info("Context tokens : " + " ".join(sample["context"]["tokens"]))
            logger.info(
                "Context ids : " + " ".join([str(v) for v in sample["context"]["ids"]])
            )
            for l in sample["label_idxs"]:
                if l == -1:
                    break
                logger.info(
                    f"Label {l} tokens : " + " ".join(entity_dictionary[l]["tokens"])
                )
                logger.info(
                    f"Label {l} ids : "
                    + " ".join([str(v) for v in entity_dictionary[l]["ids"]])
                )

    # Tensor Dataset Creation :
    # Converts processed samples into PyTorch tensors for model input.
    context_vecs = torch.tensor(
        select_field(processed_samples, "context", "ids"),
        dtype=torch.long,
    )
    label_idxs = torch.tensor(
        select_field(processed_samples, "label_idxs"),
        dtype=torch.long,
    )
    n_labels = torch.tensor(
        select_field(processed_samples, "n_labels"),
        dtype=torch.int,
    )
    mention_idx = torch.arange(len(n_labels), dtype=torch.long)

    tensor_data = TensorDataset(context_vecs, label_idxs, n_labels, mention_idx)

    return processed_samples, entity_dictionary, tensor_data


def compute_gold_clusters(mention_data):
    """
    Description
    -----------
    Group the mentions (in mention_data) into clusters based on their labels
    (Retrieves the position of each mention that shares the same label)

    Parameters
    ----------
    mention_data : list of dict
        List where each element is a dictionary representing a mention
    -------
    Returns a dict(list) where each key is a label index and each value is a list of mention indices that share that same label.
    Ex : clusters = {"label_1" : [2, 7, 12],
                    "label_2" : [3, 6, 10]}
    """
    clusters = {}
    for men_idx, mention in enumerate(mention_data):
        for i in range(mention["n_labels"]):
            label_idx = mention["label_idxs"][i]
            if label_idx not in clusters:
                clusters[label_idx] = []
            clusters[label_idx].append(men_idx)
    return clusters


def build_index(embeds, force_exact_search, probe_mult_factor=1):  # DD7
    """
    Description
    -----------
    Constructs a search index using FAISS to efficiently perform similarity searches on a set of embeddings

    Parameters
    ----------
    embeds : numpy.ndarray
        Embeddings for which the index is to be built
    force_exact_search : bool
        Boolean flag indicating whether to use exact search instead of approximate search
    probe_mult_factor : int or float ##DD8
        A multiplier factor used to determine the number of cells to probe in an approximate search
    -------
    Returns index (which is a FAISS index object). This index can be used for efficient similarity search among the embeddings.
    """
    # Convert Embeddings to NumPy Array (format required for further processing with faiss library)
    if type(embeds) is not np.ndarray:
        if torch.is_tensor(embeds):
            embeds = embeds.numpy()  # Converts pytorch tensor to a numpy array
        else:  # Could be a list, or any iterable
            embeds = np.array(embeds)

    # Initialize Index Variables
    nembeds = embeds.shape[0]  # number of embeddings
    d = embeds.shape[1]  # dimensionality of each embedding

    if (
        nembeds <= 10000 or force_exact_search
    ):  # if the number of embeddings is small, don't approximate
        index = faiss.IndexFlatIP(d)  # D,C
        index.add(embeds)
    else:
        # number of quantized clusters / cells that we want for the search
        nlist = int(math.floor(math.sqrt(nembeds)))
        # number of the quantized clusters / cells to probe (=to search)
        nprobe = int(math.floor(math.sqrt(nlist) * probe_mult_factor))
        quantizer = faiss.IndexFlatIP(d)
        index = faiss.IndexIVFFlat(
            quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT
        )  # D,C
        index.train(embeds)  # D
        index.add(embeds)
        index.nprobe = nprobe
    return index


# Only need to do "distances, indices = index.search(np.array([query]), k)" to get the k closest neighbors and their distances from the query


def embed_and_index(
    model,
    token_id_vecs,
    encoder_type,
    batch_size=768,
    # n_gpu=1,
    only_embed=False,
    corpus=None,
    force_exact_search=False,
    probe_mult_factor=1,
):
    """
    Description
    -----------
    Designed for embedding the token ID vectors AND INDEXING them for efficient search (tldr : embedding + indexing)

    Parameters
    ----------
    model :
        Object representing a ML model
    token_id_vecs : tensor
        Tensors representing tokenized text data
    encoder_type : str
        Determines the type of encoder to be used from the model. It must be either "context" or "candidate".
    batch_size : int
        The number of token vectors to process in each batch.
    n_gpu : int
        The number of GPUs to be used for processing.
    only_embed : bool
        If set to True, the function returns only the embeddings, skipping the indexing part.
    corpus : list of dict
        Collection of entities, which is used to build type-specific search indexes if provided.
    force_exact_search : bool
        Determine whether to use exact search methods or approximate methods while building the search index.
    probe_mult_factor : int
        A multiplier factor used in index building for probing in case of approximate search
    -------
    Returns :
        The embeddings of the token ID vectors with dim = (len(token_id_vecs), embedding_dimension)
        If encoder_type is "context", embeds contains the embeddings of : [CLS] c_left [START] mention [END] c_right [SEP]
        If encoder_type is "candidate", embeds contains the embeddings of [CLS] e_title [TITLE] e_desc [SEP]

        And optionally : Builds and returns a search index based on the embeddings.
    """

    with torch.no_grad():  # DD9 Set all the gradients in the computation graph to be non-updatable
        # Selects the appropriate encoder
        if encoder_type == "context":  # mention
            print("encode context")
            encoder = model.encode_context
        elif encoder_type == "candidate":  # entity
            print("encode candidate")
            encoder = model.encode_candidate
        else:
            raise ValueError("Invalid encoder_type: expected context or candidate")

        # Compute embeddings
        embeds = None
        sampler = SequentialSampler(
            token_id_vecs
        )  # Type of sampler (other one is RandomSampler)
        dataloader = DataLoader(token_id_vecs, sampler=sampler, batch_size=batch_size)
        iter_ = tqdm(dataloader, desc="Embedding in batches")
        for step, batch in enumerate(iter_):
            # print("batch size:", batch.size())
            batch_embeds = encoder(
                batch.cuda()
            )  # batch is being transferred to a GPU (if available) for encoder processing.
            embeds = (  # D Concatenate all embeddings into a single array
                batch_embeds
                if embeds is None
                else np.concatenate((embeds, batch_embeds), axis=0)  # DD10
            )
        # print("embeds size:", len(embeds))
        # print("embedding operation is over")

        if only_embed:
            return embeds

        if corpus is None:  # DD11
            # When "use_types" is False
            index = build_index(
                embeds, force_exact_search, probe_mult_factor=probe_mult_factor
            )
            return embeds, index

        # Build type-specific search indexes
        search_indexes = (
            {}
        )  # Dictionary that will store search indexes (!= indices)for each unique entity type found in the corpus
        corpus_idxs = (
            {}
        )  # Dictionary to store indices of the corpus elements, grouped by their entity type.
        for i, e in enumerate(corpus):
            ent_type = e["type"]
            if ent_type not in corpus_idxs:
                corpus_idxs[ent_type] = []
            corpus_idxs[ent_type].append(i)
        for ent_type in corpus_idxs:
            search_indexes[ent_type] = build_index(
                embeds[corpus_idxs[ent_type]],
                force_exact_search,
                probe_mult_factor=probe_mult_factor,
            )
            corpus_idxs[ent_type] = np.array(corpus_idxs[ent_type])
        return embeds, search_indexes, corpus_idxs


def get_index_from_embeds(
    embeds, corpus_idxs=None, force_exact_search=False, probe_mult_factor=1
):
    """
    Description
    -----------
    Designed to build search indexes from embeddings.
    Similar to "build_index" function but it can handle the creation of multiple indexes if corpus_idxs is provided (set of embeddings).

    Parameters
    ----------
    embeds : numpy.ndarray
        Embeddings for which the index is to be built
    corpus_idxs : dict
        Dictionary containing entity types (each entity type have the indices from the corpus corresponding to that entity)
    force_exact_search : bool
        Determine whether to use exact search methods or approximate methods while building the search index.
    probe_mult_factor : int
        A multiplier factor used in index building for probing in case of approximate search
    -------
    Returns a single index if corpus_idxs is None, or a dictionary of indexes (one for each entity type) if corpus_idxs is provided.
    """
    if corpus_idxs is None:
        index = build_index(
            embeds, force_exact_search, probe_mult_factor=probe_mult_factor
        )
        return index
    search_indexes = {}
    for ent_type in corpus_idxs:
        search_indexes[ent_type] = build_index(
            embeds[corpus_idxs[ent_type]],
            force_exact_search,
            probe_mult_factor=probe_mult_factor,
        )
    return search_indexes


def get_idxs_by_type(corpus):
    """
    Description
    -----------
    Regroup the entities INDICES based on entity types
    It simply organizes the data.

    Parameters
    ----------
    corpus : list of dict
        Collection of entities.
    -------
    Returns the corpus_idxs dictionary, which maps each entity type to an array of INDICES representing the positions of entities of that type in the original corpus
    Ex : entity_dict = {"entity_1" : [2, 7, 12], #indices
                    "entity_2" : [3, 6, 10]}
    """
    corpus_idxs = (
        {}
    )  # Dictionary used to store indices of the corpus elements, grouped by their entity type.
    for i, e in enumerate(corpus):
        ent_type = e["type"]
        if ent_type not in corpus_idxs:
            corpus_idxs[ent_type] = []
        corpus_idxs[ent_type].append(i)
    for ent_type in corpus_idxs:
        corpus_idxs[ent_type] = np.array(corpus_idxs[ent_type])
    return corpus_idxs
