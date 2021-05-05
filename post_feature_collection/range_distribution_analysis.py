import json
from os.path import join
from tqdm import tqdm
from post_feature_collection.post_process_feature_extractor import feat_label_extraction
from leaderboardscripts.lb_postprocess_utils import load_json_score_data
import numpy as np
import matplotlib.pyplot as plt
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    raw_train_file_name = join(raw_data_path, raw_train_name)
    raw_data = load_json_score_data(raw_train_file_name)
    train_feat_file_name = join(score_data_path, train_feat_name)
    feat_data = load_json_score_data(train_feat_file_name)
    x_feat_np, y_label_np = feat_to_np_array(raw_data=raw_data, feat_data=feat_data)
    return x_feat_np, y_label_np

def feat_to_np_array(raw_data, feat_data):
    x_feat_list = []
    y_label_list = []
    for case in tqdm(raw_data):
        key = case['_id']
        if key in feat_data:
            feat_case = feat_data[key]
            x_feat = feat_case['x_feat']
            y_label = feat_case['y_label']
            x_feat_list.append(x_feat)
            y_label_list.append([y_label[0], y_label[1][0], y_label[1][1]])
    x_feat_np = np.array(x_feat_list)
    y_label_np = np.array(y_label_list)
    return x_feat_np, y_label_np

def dev_range_analysis():
    raw_dev_file_name = join(raw_data_path, raw_dev_name)
    raw_data = load_json_score_data(raw_dev_file_name)
    dev_feat_file_name = join(score_data_path, dev_feat_name)
    feat_data = load_json_score_data(dev_feat_file_name)
    x_feat_np, y_label_np = feat_to_np_array(raw_data=raw_data, feat_data=feat_data)
    return x_feat_np, y_label_np

def y_label_plot(y_label_np):
    y_min = y_label_np[:,1]
    y_max = y_label_np[:,2]
    f1_score = y_label_np[:,0]
    x = np.arange(1, y_label_np.shape[0] + 1)
    sorted_idxes = np.argsort(y_min)
    y_min = y_min[sorted_idxes]
    y_max = y_max[sorted_idxes]
    f1_score = f1_score[sorted_idxes]
    y_diff = y_max - y_min
    print(sum(y_diff < 0))
    print(y_min[y_diff < 0])
    print(y_max[y_diff < 0])
    print(f1_score[y_diff < 0])
    plt.plot(x, y_min, 'x')
    plt.plot(x, y_max, '.')
    plt.show()

if __name__ == '__main__':
    dev_feat_extractor()
    train_feat_extractor()
    # # train_range_analysis()
    # # x_feat_np, y_label_np = dev_range_analysis()
    # x_feat_np, y_label_np = train_range_analysis()
    #
    # print(x_feat_np.shape, y_label_np.shape)
    # y_label_plot(y_label_np=y_label_np)
    print()