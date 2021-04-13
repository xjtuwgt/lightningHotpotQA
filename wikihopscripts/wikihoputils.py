import json
from os.path import join
from tqdm import tqdm
path = '/Users/xjtuwgt/Downloads/qangaroo_v1.1/wikihop'

train_data = 'train.json'

with open(join(path, train_data), 'r', encoding='utf-8') as reader:
    wiki_hop_train_data = json.load(reader)

print(len(wiki_hop_train_data))

def data_stats(data):
    relation_dict = {}
    def relation_entity_split(query: str):
        first_space_idx = query.index(' ')
        relation = query[:first_space_idx]
        entity = query[first_space_idx:].strip()
        return relation, entity
    ## query, answer, candidate, supports
    for row_idx, row in tqdm(enumerate(data)):

        print(row['query'])
        relation, entity = relation_entity_split(query=row['query'])
        print(relation)
        print(entity)
        # print(row_idx)
        # print(row)
        # for key, value in row.items():
        #     print(key, value)
        # break

if __name__ == '__main__':
    data_stats(data=wiki_hop_train_data)