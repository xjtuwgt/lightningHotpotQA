import json
from os.path import join
from tqdm import tqdm
import spacy
# from envs import DATASET_FOLDER
# path = '/Users/xjtuwgt/Downloads/qangaroo_v1.1/wikihop'
#
# train_data = 'train.json'

nlp = spacy.load("en_core_web_lg", disable=['parser'])
def get_contents_with_ner(wiki_data_name):
    with open(wiki_data_name, 'r', encoding='utf-8') as reader:
        wiki_data = json.load(reader)
    print('Loading {} data from {}'.format(len(wiki_data), wiki_data_name))
    documents = []
    for row_idx, row in tqdm(enumerate(wiki_data)):
        _text_ner = []
        for sent in row['supports']:
            ent_list = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in nlp(sent).ents]
            _text_ner.append(ent_list)
        row['supports_ner'] = _text_ner
        documents.append(row)
    return documents



def data_stats(wiki_data_name):
    with open(wiki_data_name, 'r', encoding='utf-8') as reader:
        wiki_data = json.load(reader)
    print('Loading {} data from {}'.format(len(wiki_data), wiki_data_name))
    relation_dict = {}
    num_sents_dict = {}
    num_ents_dict = {}
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
        supports = row['supports']
        sent_num = len(supports)
        if sent_num not in num_sents_dict:
            num_sents_dict[sent_num] = 1
        else:
            num_sents_dict[sent_num] = num_sents_dict[sent_num] + 1

        support_ners = row['supports_ner']
        assert len(supports) == len(support_ners)
        print(support_ners)

        # if ent_num not in num_ents_dict:
        #     num_ents_dict[ent_num] = 1
        # else:
        #     num_ents_dict[ent_num] = num_ents_dict[ent_num] + 1
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
    for key, value in num_sents_dict.items():
        print('{}\t{}'.format(key, value))
    print('Number of sentences = {}'.format(len(relation_dict)))

    for key, value in num_ents_dict.items():
        print('{}\t{}'.format(key, value))
    print('Number of entities = {}'.format(len(num_ents_dict)))
