import argparse
from wikihopscripts.wikihoputils import get_contents_with_ner
from os.path import join
from envs import DATASET_FOLDER

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--data_path", type=str, default=join(DATASET_FOLDER, 'data_raw/wikihop_data'))
    parser.add_argument("--data_name", type=str, default='train.json', required=True)

    args = parser.parse_args()
    data_folder = args.data_path
    for key, value in vars(args).items():
        print('Parameter {}: {}'.format(key, value))
    wiki_data_file_name = join(data_folder, args.data_name)
    get_contents_with_ner(wiki_data_name=wiki_data_file_name)