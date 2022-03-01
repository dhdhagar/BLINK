#!/bin/bash




precomputed_index_path=$1
parent_output_folder=$2

# Generate mappings between wikipedia and wikidata ids, and titles and wikidata ids
python blink/candidate_retrieval/generate_wiki2wikidata_mappings.py --input_file $precomputed_index_path --output_folder $parent_output_folder

# Generate mappings between wikipedia and wikidata ids, and titles and wikidata ids
python blink/candidate_retrieval/link_wikipedia_and_wikidata.py --output_folder $parent_output_folder
