import numpy as np
from adaptive_threshold.atutils import distribution_feat, distribution_feat_extraction, parse_args, feat_label_extraction
from os.path import join
import json

if __name__ == '__main__':

    # x = np.random.random(10)
    # print(x)
    # y = distribution_feat_extraction(scores=x, keep_num=True)
    # print(len(y))
    # z = distribution_feat(scores=x)
    # print(len(z))

    args = parse_args()
    dev_raw_data_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_score_name)
    feat_label_extraction(raw_data_name=dev_raw_data_file_name, score_data_name=dev_score_file_name, train_type=args.train_type, train=False)
    # with open(dev_score_file_name, 'r', encoding='utf-8') as reader:
    #     dev_score_data = json.load(reader)
    # print(len(dev_score_data))
    # print(dev_score_file_name)