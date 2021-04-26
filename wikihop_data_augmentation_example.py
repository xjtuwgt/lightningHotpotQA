import argparse
from wikihopscripts.wikihoputils import get_contents_with_ner, data_stats, read_wikihop_examples
from os.path import join
import json
from envs import DATASET_FOLDER

# step 1: wiki_hop ner extraction
# wiki_data_file_name = join(data_folder, args.data_name)
# data_with_ner = get_contents_with_ner(wiki_data_name=wiki_data_file_name)
# # data_stats(wiki_data_name=wiki_data_file_name)
# out_put_ner_file_name = join(args.output_path, 'ner_' + args.data_name)
# with open(out_put_ner_file_name, 'w') as f:
#     json.dump(data_with_ner, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    # parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_raw/wikihop_data'))
    # parser.add_argument("--data_name", type=str, default='ner_train.json')
    # parser.add_argument("--output_path", type=str, default=join(DATASET_FOLDER, 'data_processed/wikihop'))
    #
    # args = parser.parse_args()
    # data_folder = args.data_path
    # for key, value in vars(args).items():
    #     print('Parameter {}: {}'.format(key, value))
    #
    #
    #
    # wiki_ner_data_file_name = join(args.output_path, args.data_name)
    # # data_stats(wiki_data_name=wiki_ner_data_file_name)
    # token_data_examples = read_wikihop_examples(full_file=wiki_ner_data_file_name)
    # wiki_token_ner_data_file_name = join(args.output_path, 'token_' + args.data_name)
    # with open(wiki_token_ner_data_file_name, 'w') as f:
    #     json.dump(token_data_examples, f)

    import json
    from tqdm import tqdm

    dev_data_file_name = join(DATASET_FOLDER, 'data_raw', 'hotpot_dev_distractor_v1.json')
    test_data_file_name = join(DATASET_FOLDER, 'data_raw', 'hotpot_test_distractor_v1.json')

    with open(dev_data_file_name, 'r', encoding='utf-8') as reader:
        dev_full_data = json.load(reader)

    print(type(dev_full_data))
    test_full_data = []
    test_col_names = ['_id', 'question', 'context']

    for row in tqdm(dev_full_data):
        test_row = dict([(key, value) for key, value in row.items() if key in test_col_names])
        test_full_data.append(test_row)

    json.dump(test_full_data, open(test_data_file_name, 'w'))

    with open(test_data_file_name, 'r', encoding='utf-8') as reader:
        test_full_data = json.load(reader)

    for row in tqdm(test_full_data):
        print(row)
