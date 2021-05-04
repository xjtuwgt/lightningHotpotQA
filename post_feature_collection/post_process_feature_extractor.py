from post_feature_collection.post_process_argument_parser import train_parser
from os.path import join
from leaderboardscripts.lb_postprocess_utils import load_json_score_data
from time import time


if __name__ == '__main__':
    args = train_parser()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

    train_score_file_name = join(args.output_dir, args.exp_name, args.train_score_name)
    dev_score_file_name = join(args.output_dir, args.exp_name, args.dev_score_name)

    train_score_data = load_json_score_data(train_score_file_name)
    print('Loading {} records from {}'.format(len(train_score_data), train_score_file_name))
    dev_score_data = load_json_score_data(dev_score_file_name)
    print('Loading {} records from {}'.format(len(dev_score_data), dev_score_file_name))
