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
    parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_raw/wikihop_data'))
    parser.add_argument("--data_name", type=str, default='ner_train.json')
    parser.add_argument("--output_path", type=str, default=join(DATASET_FOLDER, 'data_processed/wikihop'))

    args = parser.parse_args()
    data_folder = args.data_path
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))



    wiki_ner_data_file_name = join(args.output_path, args.data_name)
    data_stats(wiki_data_name=wiki_ner_data_file_name)
    # read_wikihop_examples(full_file=wiki_ner_data_file_name)