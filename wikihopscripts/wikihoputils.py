import json
from os.path import join
from tqdm import tqdm
import spacy
from envs import DATASET_FOLDER
path = '/Users/xjtuwgt/Downloads/qangaroo_v1.1/wikihop'

train_data = 'train.json'





nlp = spacy.load("en_core_web_lg", disable=['parser'])

def get_contents_with_ner(wiki_data_name):
    with open(wiki_data_name, 'r', encoding='utf-8') as reader:
        wiki_data = json.load(reader)
    print('Loading {} data from {}'.format(len(wiki_data), wiki_data_name))
    documents = []
    for row_idx, row in tqdm(enumerate(wiki_data)):
        for key, value in row.items():
            print(key, value)
        break
    return documents

def data_stats(wiki_data_name):
    with open(wiki_data_name, 'r', encoding='utf-8') as reader:
        wiki_data = json.load(reader)
    print('Loading {} data from {}'.format(len(wiki_data), wiki_data_name))
    relation_dict = {}
    def relation_entity_split(query: str):
        first_space_idx = query.index(' ')
        relation = query[:first_space_idx]
        entity = query[first_space_idx:].strip()
        return relation, entity
    ## query, answer, candidate, supports
    for row_idx, row in tqdm(enumerate(wiki_data)):
        relation, entity = relation_entity_split(query=row['query'])
        if relation not in relation_dict:
            relation_dict[relation] = 1
        else:
            relation_dict[relation] = relation_dict[relation] + 1
        # print(relation)
        # print(entity)
        # print(row_idx)
        # print(row)
        # for key, value in row.items():
        #     print(key, value)
        # break
    for key, value in relation_dict.items():
        print('{}\t{}'.format(key, value))
    print('Number of relations = {}'.format(len(relation_dict)))
