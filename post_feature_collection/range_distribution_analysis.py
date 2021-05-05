import json
from os.path import join
from tqdm import tqdm
from post_feature_collection.post_process_feature_extractor import feat_label_extraction
from leaderboardscripts.lb_postprocess_utils import load_json_score_data



raw_data_path = '/Users/xjtuwgt/PycharmProjects/LongSeqMultihopReason/data/hotpotqa'
score_data_path = '/Users/xjtuwgt/Desktop/HotPotQA'

raw_train_name = 'hotpot_train_v1.1.json'
raw_dev_name = 'hotpot_dev_distractor_v1.json'

score_train_name = 'train_post_6_4_score.json'
score_dev_name = 'dev_distractor_post_6_4_score.json'

train_feat_name = 'train_feat_data.json'
dev_feat_name = 'dev_feat_data.json'

def train_feat_extractor():
    raw_train_file_name = join(raw_data_path, raw_train_name)
    train_score_file_name = join(score_data_path, score_train_name)
    train_feat_file_name = join(score_data_path, train_feat_name)
    train_feat_dict = feat_label_extraction(raw_data_name=raw_train_file_name, score_data_name=train_score_file_name)
    json.dump(train_feat_dict, open(train_feat_file_name, 'w'))
    print('Saving {} records into {}'.format(len(train_feat_dict), train_feat_file_name))

def dev_feat_extractor():
    raw_dev_file_name = join(raw_data_path, raw_dev_name)
    dev_score_file_name = join(score_data_path, score_dev_name)
    dev_feat_file_name = join(score_data_path, dev_feat_name)
    dev_feat_dict = feat_label_extraction(raw_data_name=raw_dev_file_name, score_data_name=dev_score_file_name)
    json.dump(dev_feat_dict, open(dev_feat_file_name, 'w'))
    print('Saving {} records into {}'.format(len(dev_feat_dict), dev_feat_file_name))

def train_range_analysis():
    train_feat_file_name = join(score_data_path, train_feat_name)
    feat_data = load_json_score_data(train_feat_file_name)
    print('Loading {} records from {}'.format(len(feat_data), train_feat_file_name))

def dev_range_analysis():
    raw_dev_file_name = join(raw_data_path, raw_dev_name)
    raw_data = load_json_score_data(raw_dev_file_name)
    dev_feat_file_name = join(score_data_path, dev_feat_name)
    feat_data = load_json_score_data(dev_feat_file_name)
    print('Loading {} records from {}'.format(len(feat_data), dev_feat_file_name))
    x_feat_list = []
    y_label_list = []
    for case in tqdm(raw_data):
        key = case['_id']
        if key in feat_data:
            feat_case = feat_data[key]
            x_feat = feat_case['x_feat']
            y_label = feat_case['y_label']
            print(y_label)
        # y_label = case['y_label']
        # print(y_label)

if __name__ == '__main__':
    # dev_feat_extractor()
    # train_feat_extractor()
    # train_range_analysis()
    dev_range_analysis()
    print()