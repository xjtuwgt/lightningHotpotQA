import argparse
from wikihopscripts.wikihoputils import get_contents_with_ner, data_stats
from os.path import join
import json
from envs import DATASET_FOLDER

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_raw/wikihop_data'))
    parser.add_argument("--data_name", type=str, default='train.json', required=True)
    parser.add_argument("--output_path", type=str, default=join(DATASET_FOLDER, 'data_processed/wikihop'))

    args = parser.parse_args()
    data_folder = args.data_path
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))

    # step 1: wiki_hop ner extraction
    # wiki_data_file_name = join(data_folder, args.data_name)
    # data_with_ner = get_contents_with_ner(wiki_data_name=wiki_data_file_name)
    # # data_stats(wiki_data_name=wiki_data_file_name)
    # out_put_ner_file_name = join(args.output_path, 'ner_' + args.data_name)
    # with open(out_put_ner_file_name, 'w') as f:
    #     json.dump(data_with_ner, f)