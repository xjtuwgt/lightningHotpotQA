import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

path = '/Users/xjtuwgt/Desktop'
dev_json_file_name = 'error_res.json'
train_json_file_name = 'hgn_low_saeerror_train_res.json'

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


# hist_plot(data=dev_error_df)
# hist_plot(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_min_p(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_min_p(dev_data=dev_error_df, train_data=train_error_df)
# dist_plot_max_n(dev_data=dev_error_df, train_data=train_error_df)

# dist_plot(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])
# dist_plot_min_p(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])
dist_plot_max_n(dev_data=dev_error_df[dev_error_df['q_type']=='comparison'], train_data=train_error_df[train_error_df['q_type']=='comparison'])

# dist_plot(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])
# dist_plot_min_p(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])
# dist_plot_max_n(dev_data=dev_error_df[dev_error_df['q_type']=='bridge'], train_data=train_error_df[train_error_df['q_type']=='bridge'])