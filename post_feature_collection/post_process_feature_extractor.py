from post_feature_collection.post_process_argument_parser import train_parser
from os.path import join
from leaderboardscripts.lb_postprocess_utils import load_json_score_data

def feat_label_extraction(raw_data_name, score_data_name):
    raw_data = load_json_score_data(raw_data_name)
    print('Loading {} records from {}'.format(len(raw_data), raw_data_name))
    score_data = load_json_score_data(score_data_name)
    print('Loading {} records from {}'.format(len(score_data), score_data_name))

def train_feature_label_extraction(args):
    raw_train_file_name = join(args.input_dir, args.raw_train_data)
    train_score_file_name = join(args.output_dir, args.exp_name, args.train_score_name)
    train_feat_file_name = join(args.output_dir, args.exp_name, args.train_feat_json_name)
    feat_label_extraction(raw_data_name=raw_train_file_name, score_data_name=train_score_file_name)
    return

def dev_feature_label_extraction(args):
    raw_dev_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.output_dir, args.exp_name, args.dev_score_name)
    dev_feat_file_name = join(args.output_dir, args.exp_name, args.dev_feat_json_name)
    feat_label_extraction(raw_data_name=raw_dev_file_name, score_data_name=dev_score_file_name)
    return

if __name__ == '__main__':
    args = train_parser()
    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))
    print('*' * 75)
    train_feature_label_extraction(args=args)
    dev_feature_label_extraction(args=args)