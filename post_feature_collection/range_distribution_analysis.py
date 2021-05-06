import json
from os.path import join
from tqdm import tqdm
from post_feature_collection.post_process_feature_extractor import feat_label_extraction
from leaderboardscripts.lb_postprocess_utils import load_json_score_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from time import time
import pandas as pd
import seaborn as sns
from numpy import ndarray
from sklearn.preprocessing import normalize as skl_norm
from post_feature_collection.post_process_feature_extractor import threshold_map_to_label, np_sigmoid
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
raw_data_path = '/Users/xjtuwgt/PycharmProjects/LongSeqMultihopReason/data/hotpotqa'
score_data_path = '/Users/xjtuwgt/Desktop/HotPotQA'

raw_train_name = 'hotpot_train_v1.1.json'
raw_dev_name = 'hotpot_dev_distractor_v1.json'

score_train_name = 'train_post_6_4_score.json'
score_dev_name = 'dev_distractor_post_6_4_score.json'

train_feat_name = 'train_feat_data.json'
dev_feat_name = 'dev_feat_data.json'
# threshold_category = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
interval_num = 100
interval_range = 1.0/interval_num
threshold_category = [(i * interval_range, (i+1) * interval_range) for i in range(interval_num)]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
    split_count = 0
    for idx in range(y_label_np.shape[0]):
        y_min_i = y_min[idx]
        y_max_i = y_max[idx]
        if y_min_i < 0 and y_max_i > 0:
            split_count = split_count + 1

    # f1_score = f1_score[sorted_idxes]
    # y_diff = y_max - y_min
    # print(sum(y_diff < 0))
    # print(y_min[y_diff < 0])
    # print(y_max[y_diff < 0])
    # print(f1_score[y_diff < 0])
    print(split_count)
    plt.plot(x, y_min, 'x')
    plt.plot(x, y_max, '.')
    plt.show()

def pca_analysis(x_feat, y_label):
    pca = PCA(n_components=3)
    # x_feat = skl_norm(x_feat, axis=1)
    pca_results = pca.fit_transform(x_feat)
    pca_data = {'pca-2d-one': pca_results[:, 0], 'pca-2d-two': pca_results[:, 1]}
    df_subset = pd.DataFrame.from_dict(pca_data)
    flag_idx_list, _, flag_label_freq, _ = threshold_map_to_label(y_label=y_label,
                                                                  threshold_category=threshold_category)
    df_subset['y'] = np.array(flag_idx_list)
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x="pca-2d-one", y="pca-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(flag_label_freq)),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    # plt.plot(pca_results[:,0], pca_results[:,1], '.')
    plt.show()

def tsne_analysis(x_feat, y_label, perplexity=100):
    time_start = time()
    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=500)
    # x_feat = skl_norm(x_feat, axis=1)
    tsne_results = tsne.fit_transform(x_feat)
    print('t-SNE done! Time elapsed: {} seconds'.format(time() - time_start))
    tsne_data = {'tsne-2d-one': tsne_results[:, 0], 'tsne-2d-two': tsne_results[:, 1]}
    df_subset = pd.DataFrame.from_dict(tsne_data)
    flag_idx_list, _, flag_label_freq, _ = threshold_map_to_label(y_label=y_label, threshold_category=threshold_category)
    df_subset['y'] = np.array(flag_idx_list)
    # y = np.zeros(y_label.shape[0])
    # y_min_sigmoid, y_max_sigmoid = np_sigmoid(y_label[:,1]), np_sigmoid(y_label[:,2])
    # f1_score = y_label[:,0]
    # for i in range(y_label.shape[0]):
    #     f1_i = f1_score[i]
    #     y_min_i = y_min_sigmoid[i]
    #     y_max_i = y_max_sigmoid[i]
    #     if f1_i == 1:
    #        if y_min_i < 0.25 and y_max_i > 0.25:
    #            y[i] = 0
    #        else:
    #            y[i] = 1
    #     else:
    #         y[i] = 2
    # df_subset['y'] = y
    plt.figure(figsize=(16, 10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue="y",
        palette=sns.color_palette("hls", len(flag_label_freq)),
        data=df_subset,
        legend="full",
        alpha=0.1
    )
    # plt.plot(tsne_results[:,0], tsne_results[:,1], 'x')
    plt.show()

def range_distribution(y_label):
    y_min, y_max = np_sigmoid(y_label[:, 1]), np_sigmoid(y_label[:, 2])
    y_diff = y_max - y_min
    # plt.hist(x=y_diff, bins='auto', color='#0504aa',
    #          alpha=0.7, rwidth=0.85)

    plt.hist(x=y_diff, bins='auto', density=True, histtype='step', cumulative=True,
            label='Reversed emp.')
    plt.show()

if __name__ == '__main__':
    # dev_feat_extractor()
    # train_feat_extractor()
    # # train_range_analysis()

    x_feat_np, y_label_np = dev_range_analysis()

    # threshold_map_to_label(y_label=y_label_np, threshold_category=threshold_category)
    # # print(flag_list)
    # print(flag_freq)
    # print(len(flag_freq))
    # x_feat_np, y_label_np = train_range_analysis()

    # idx_arr = np.arange(x_feat_np.shape[0])
    # np.random.shuffle(idx_arr)
    # sel_idx = idx_arr[:40000]
    # x_feat_np = x_feat_np[sel_idx,:]
    # y_label_np = y_label_np[sel_idx,:]

    tsne_analysis(x_feat=x_feat_np, y_label=y_label_np)
    # pca_analysis(x_feat=x_feat_np, y_label=y_label_np)

    # range_distribution(y_label=y_label_np)

    # print(x_feat_np.shape, y_label_np.shape)
    # y_label_plot(y_label_np=y_label_np)
    print()