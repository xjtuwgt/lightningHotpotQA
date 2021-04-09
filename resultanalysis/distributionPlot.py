import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from pandas import DataFrame
from adaptive_threshold.atutils import load_npz_data

path = '/Users/xjtuwgt/Desktop'
dev_json_file_name = 'dev_error_res.json'
train_json_file_name = 'hgn_low_saeerror_train_res.json'

npz_data = os.path.join(path, 'HotPotQA/train_np_data.npz')
dev_npz_data = os.path.join(path, 'HotPotQA/dev_np_data.npz')
x, y, y_n, y_np = load_npz_data(npz_data)
d_x, d_y, d_y_n, d_y_np = load_npz_data(dev_npz_data)


dev_error_df = pd.read_json(os.path.join(path, dev_json_file_name))
train_error_df = pd.read_json(os.path.join(path, train_json_file_name))

print(dev_error_df.shape, train_error_df.shape)


def hist_plot(dev_data, train_data):
    dev_min_positive_scores = dev_data['min_p'].to_numpy()
    dev_max_negative_scores = dev_data['max_n'].to_numpy()

    dev_diff_score = dev_min_positive_scores - dev_max_negative_scores
    train_min_positive_scores = train_data['min_p'].to_numpy()
    train_max_negative_scores = train_data['max_n'].to_numpy()

    train_diff_score = train_min_positive_scores - train_max_negative_scores
    plt.hist(x=train_diff_score, bins='auto', color='b',
                                alpha=0.7, rwidth=0.85)
    plt.hist(x=dev_diff_score, bins='auto', color='r',
                                alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Diff Value')
    # plt.ylabel('Frequency')
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()

def dist_plot(dev_data, train_data):
    dev_data['diff_score'] = dev_data['min_p'] - dev_data['max_n']
    sns.distplot(dev_data['diff_score'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='dev')

    train_data['diff_score'] = train_data['min_p'] - train_data['max_n']
    sns.distplot(train_data['diff_score'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='Train')

    plt.legend(prop={'size': 16}, title='Score diff')
    plt.title('Density Plot')
    plt.ylabel('Density')
    plt.show()

def dist_plot_min_p(dev_data, train_data):
    # dev_data['diff_score'] = dev_data['min_p'] - dev_data['max_n']
    sns.distplot(dev_data['min_p'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='dev')

    # train_data['diff_score'] = train_data['min_p'] - train_data['max_n']
    sns.distplot(train_data['min_p'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='Train')

    plt.legend(prop={'size': 16}, title='Min positive')
    plt.title('Density Plot')
    plt.ylabel('Density')
    plt.show()

def dist_plot_max_n(dev_data, train_data):
    # dev_data['diff_score'] = dev_data['min_p'] - dev_data['max_n']
    sns.distplot(dev_data['max_n'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='dev')

    # train_data['diff_score'] = train_data['min_p'] - train_data['max_n']
    sns.distplot(train_data['max_n'], hist=False, kde=True,
                 kde_kws={'linewidth': 3}, label='Train')

    plt.legend(prop={'size': 16}, title='Max negative')
    plt.title('Density Plot')
    plt.ylabel('Density')
    plt.show()

def dev_plot(dev_data: DataFrame):
    min_positive = dev_data['min_p'].to_numpy()
    max_negative = dev_data['max_n'].to_numpy()
    pred_threshold = dev_data['threshold'].to_numpy()

    sorted_idxes = np.argsort(min_positive)
    min_positive = min_positive[sorted_idxes]
    max_negative = max_negative[sorted_idxes]
    pred_threshold = pred_threshold[sorted_idxes]

    x = np.arange(0, min_positive.shape[0])
    plt.plot(x, min_positive, color='r')
    # plt.plot(x, max_negative, label="max_n")
    plt.plot(x, pred_threshold, '.')

    plt.show()


def train_plot(y, y_n):
    x = np.arange(0, y.shape[0])

    sorted_idxes = np.argsort(y)
    y = y[sorted_idxes]
    y_n = y_n[sorted_idxes]
    #
    # plt.plot(x, y)
    # plt.plot(x, y_n, '.')
    # plt.plot(x, y_np)
    # plt.plot(y, y_np, '.')

    diff_y = y - y_n
    sorted_y = np.sort(diff_y)
    plt.plot(x, sorted_y)
    plt.show()


def my_hist_plot(y):
    plt.hist(x=y, bins=10, color='b',
             alpha=0.7, rwidth=0.85)

    # plt.hist(x=d_y, bins='auto', color='r',
    #          alpha=0.7, rwidth=0.85)
    # plt.grid(axis='y', alpha=0.75)
    # plt.xlabel('Diff Value')
    # plt.ylabel('Frequency')
    # maxfreq = n.max()
    # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()



# hist_plot(data=dev_error_df)
# hist_plot(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_min_p(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_min_p(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_max_n(dev_data=dev_error_df, train_data=train_error_df)

# dev_plot(dev_data=dev_error_df)

train_plot(y, y_n)

# train_plot(y=d_y, y_np=d_y_np)
# my_hist_plot(y=y)

# dist_plot(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])
# dist_plot_min_p(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])
# dist_plot_max_n(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])

# dist_plot(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])
# dist_plot_min_p(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])
# dist_plot_max_n(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])