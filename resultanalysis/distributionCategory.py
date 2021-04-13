import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy import ndarray
from scipy import stats
from pandas import DataFrame
from adaptive_threshold.atutils import load_npz_data, load_npz_data_for_classification
from adaptive_threshold.atutils import threshold_map_to_label, adaptive_threshold_to_classification

path = '/Users/xjtuwgt/Desktop'
dev_json_file_name = 'dev_error_res.json'
train_json_file_name = 'hgn_low_saeerror_train_res.json'

npz_data = os.path.join(path, 'HotPotQA/train_np_data.npz')
dev_npz_data = os.path.join(path, 'HotPotQA/dev_np_data.npz')

train_npz_class_data = os.path.join(path, 'HotPotQA/train_class_np_data.npz')
dev_npz_class_data = os.path.join(path, 'HotPotQA/dev_class_np_data.npz')

# train_x, train_y_p, train_y_n, train_y_np = load_npz_data(npz_data)
# dev_x, dev_y_p, dev_y_n, dev_y_np = load_npz_data(dev_npz_data)

train_x, train_y_p, train_y_n, train_y_np, train_y_labels = load_npz_data_for_classification(npz_file_name=train_npz_class_data)
dev_x, dev_y_p, dev_y_n, dev_y_np, dev_y_labels = load_npz_data_for_classification(npz_file_name=dev_npz_class_data)

conf_category = [(0.0, 0.5), (0.5, 0.85), (0.85, 1.0)]
# threshold_category = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
threshold_category = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
# threshold_category = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]


# dev_error_df = pd.read_json(os.path.join(path, dev_json_file_name))
# train_error_df = pd.read_json(os.path.join(path, train_json_file_name))

# plt.plot(train_y_p-train_y_n, '.')
# plt.hist(x=train_y_p, bins='auto', color='b',
#          alpha=0.7, rwidth=0.85)
# print(((train_y_p-train_y_n) > 0.9).sum())
# plt.show()

def y_p_distribution(y_p: ndarray):
    cate_freq = [0] * len(conf_category)
    # for idx, bound in enumerate(conf_category):
    #     low_bound, up_bound = bound
    #     freq_i = (y_p[(y_p >= low_bound)] < up_bound).sum()
    #     cate_freq[idx] = freq_i/y_p.shape[0]
    print(cate_freq)

# def over_lap_ratio(ht_pair1, ref_ht_pair2):
#     h, t = ht_pair1
#     r_h, r_t = ref_ht_pair2
#     if t < r_h or h > r_t: ## no overlap
#         return 0.0, 1
#     if r_h >= h and r_t <= t: ## subset: ref is a subset of given pair
#         return 1.0, 2
#     if r_h <= h and r_t >= t:
#         return (t - h) / (r_t - r_h), 3 ## superset: ref is a superset of given pair
#     if h >= r_h and h < r_t:
#         return (r_t - h) / (r_t - r_h), 4
#     if r_h >= h and r_h < t:
#         return (t - r_h) / (r_t - r_h), 4

# def threshold_distribution(y_p: ndarray, y_n: ndarray):
#     over_lap_res = []
#     for i in range(y_p.shape[0]):
#         p_i = y_p[i]
#         n_i = y_n[i]
#         p_flag = True
#         if p_i > n_i:
#             ht_pair_i = (n_i, p_i)
#         else:
#             ht_pair_i = (p_i, n_i)
#             p_flag = False
#         over_lap_list = []
#         for b_idx, bound in enumerate(threshold_category):
#             over_lap_value, over_lap_type = over_lap_ratio(ht_pair_i, bound)
#             over_lap_list.append((over_lap_value, over_lap_type, p_flag))
#         over_lap_res.append((over_lap_list, p_flag))
#
#     flag_list = []
#     freq = {}
#     for i in range(y_p.shape[0]):
#         three_types = ''.join([str(int(x[1] > 1)) for x in over_lap_res[i][0]])
#         if over_lap_res[i][1]:
#             flag_label = 'T_' + str(three_types)
#         else:
#             flag_label = 'F_' + str(three_types)
#         if flag_label not in freq:
#             freq[flag_label] = 1
#         else:
#             freq[flag_label] = freq[flag_label] + 1
#         flag_list.append(flag_label)
#     keys = sorted(list(freq.keys()))
#     for key in keys:
#         print('{}\t{}'.format(key, freq[key] * 1.0 / y_n.shape[0]))
#     print('Number of itemsets = {}'.format(len(freq)))
#
#     return


def deep_analysis(y_p, y_n, conf_prob=0.85):
    y_diff = y_p - y_n

    conf_pred_ratio = (y_diff[y_p > 0.9] > conf_prob).sum()/y_diff.shape[0]
    print(conf_pred_ratio)
    conf_pred_ratio = (y_p > 0.9).sum() / y_diff.shape[0]
    print(conf_pred_ratio)



# deep_analysis(y_p=train_y_p, y_n=train_y_n)
#
# deep_analysis(y_p=dev_y_p, y_n=dev_y_n)

# y_p_distribution(y_p=train_y_p)
#
# y_p_distribution(y_p=dev_y_p)
# threshold_distribution(y_p=train_y_p, y_n=train_y_n)
#
# threshold_distribution(y_p=dev_y_p, y_n=dev_y_n)

# _, train_flag_label_freq = threshold_map_to_label(y_p=train_y_p, y_n=train_y_n, threshold_category=threshold_category)
#
# keys = sorted(list(train_flag_label_freq.keys()))
# for k_idx, key in enumerate(keys):
#     print('{}\t{}'.format(key, train_flag_label_freq[key] * 1.0 / train_y_p.shape[0]))
# print('Number of itemsets = {}'.format(len(train_flag_label_freq)))
#
# _, dev_flag_label_freq = threshold_map_to_label(y_p=dev_y_p, y_n=dev_y_n, threshold_category=threshold_category)
# keys = sorted(list(dev_flag_label_freq.keys()))
# for k_idx, key in enumerate(keys):
#     print('{}\t{}'.format(key, dev_flag_label_freq[key] * 1.0 / dev_y_p.shape[0]))
# print('Number of itemsets = {}'.format(len(dev_flag_label_freq)))
#
# flag_label_keys = sorted(list({**train_flag_label_freq, **dev_flag_label_freq}.keys()))
# for k_idx, key in enumerate(flag_label_keys):
#     print('{}\t{}\t{}\t{}'.format(k_idx, key, train_flag_label_freq[key] * 1.0 / train_y_p.shape[0], dev_flag_label_freq[key] * 1.0 / dev_y_p.shape[0]))


adaptive_threshold_to_classification(train_npz_file_name=npz_data, dev_npz_file_name=dev_npz_data,
                                     threshold_category=threshold_category, train_npz_class_file_name=train_npz_class_data,
                                     dev_npz_class_file_name=dev_npz_class_data)

# for i in range(30):
#     print('{}\t{}\t{}'.format(i, (train_y_labels == i).sum()/train_y_labels.shape[0], (dev_y_labels == i).sum()/dev_y_labels.shape[0]))

