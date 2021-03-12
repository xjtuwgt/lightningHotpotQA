#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""A script to read in and store documents in a sqlite database."""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util
import bz2
import pickle
import spacy

from multiprocessing import Pool as ProcessPool
from tqdm import tqdm
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

nlp = spacy.load("en_core_web_lg", disable=['parser'])


def get_contents(filename):
    documents = []
    docred_data = json.load(open(filename))
    id = 0
    title_to_id = {}
    for data in tqdm(docred_data):
        text = []
        title = ""
        doc_id = -1
        for d in data['context']:
            title = d[0][:-2]
            if title in title_to_id:
                doc_id = title_to_id[title]
            else:
                doc_id = id
            text.extend(d[1])

        id += 1

        if title not in title_to_id:
            _text = pickle.dumps(text)
            _text_with_links = pickle.dumps([])
            _text_ner = []
            for sent in text:
                ent_list = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(sent).ents]
                _text_ner.append(ent_list)
            _text_ner_str = pickle.dumps(_text_ner)

            documents.append((utils.normalize(str(doc_id)), "", title, _text, _text_with_links, _text_ner_str, len(text)))
            title_to_id[title] = doc_id

    return documents


def store_contents(data_path, save_path):
    """Preprocess and store a corpus of documents in sqlite.

    Args:
        data_path: Root path to directory (or directory of directories) of files
          containing json encoded documents (must have `id` and `text` fields).
        save_path: Path to output sqlite db.
    """
    if os.path.isfile(save_path):
        raise RuntimeError('%s already exists! Not overwriting.' % save_path)

    logger.info('Reading into database...')
    conn = sqlite3.connect(save_path)
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, url, title, text, text_with_links, text_ner, sent_num);")
    pairs = get_contents(data_path)
    c.executemany("INSERT INTO documents VALUES (?,?,?,?,?,?,?)", pairs)
    logger.info('Read %d docs.' % len(pairs))
    logger.info('Committing...')
    conn.commit()
    conn.close()


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', required=True, type=str, help='/path/to/data')
    parser.add_argument('--save_path', required=True, type=str, help='/path/to/saved/db.db')
    args = parser.parse_args()

    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

    store_contents(
        args.data_path, args.save_path
    )

# python scripts/0_build_db_docred.py ./data/dataset/data_raw/converted_docred_total.json  ./data/knowledge/enwiki_ner_docred.db

# python scripts/1_extract_db.py ./data/dataset/data_raw/converted_docred_total.json ./data/knowledge/enwiki_ner_docred.db ./data/dataset/data_processed/docred/doc_link_ner.json

# python3 scripts/1_extract_db.py utils/converted_docred_total.json data/knowledge/enwiki_ner_docred.db data/dataset/data_processed/docred/doc_link_ner.json

# python scripts/2_extract_ner.py ./data/dataset/data_raw/converted_docred_total.json ./data/dataset/data_processed/docred/doc_link_ner.json ./data/dataset/data_processed/docred/ner.json

# python3 scripts/2_extract_ner.py utils/converted_docred_total.json data/dataset/data_processed/docred/doc_link_ner.json data/dataset/data_processed/docred/ner.json

#        python3 scripts/5_dump_features.py --para_path ./data/dataset/data_processed/docred/multihop_para.json --full_data ./data/dataset/data_raw/converted_docred_total.json --model_name_or_path roberta-large --ner_path ./data/dataset/data_processed/docred/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir ./data/dataset/data_feat/docred --doc_link_ner ./data/dataset/data_processed/docred/doc_link_ner.json

       # python3 scripts/5_dump_features.py --para_path utils/multihop_para.json --full_data utils/converted_docred_total.json --model_name_or_path roberta-large --ner_path data/dataset/data_processed/docred/ner.json --model_type roberta --tokenizer_name roberta-large --output_dir data/dataset/data_feat/docred --doc_link_ner data/dataset/data_processed/docred/doc_link_ner.json