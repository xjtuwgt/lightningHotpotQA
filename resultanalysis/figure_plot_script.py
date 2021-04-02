import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/xjtuwgt/Desktop'
json_file_name = 'error_res.json'

error_df = pd.read_json(os.path.join(path, json_file_name))

print(error_df.shape)
for col in error_df.columns:
    print(col)



# x = np.arange(1, error_df.shape[0]+1)
# min_positive_scores = error_df['min_p'].to_numpy()
# max_negative_scores = error_df['max_n'].to_numpy()
#
# # plt.plot(x, min_positive_scores, 'o', label = "line 1")
# # plt.plot(x, max_negative_scores, 'x', label = "line 2")
# diff_score = min_positive_scores - max_negative_scores
# sorted_score = np.sort(diff_score)
#
# plt.plot(sorted_score, '.')
#
# plt.show()

def show_min_max_score(data):
    min_positive_scores = data['min_p'].to_numpy()
    max_negative_scores = data['max_n'].to_numpy()



    # plt.plot(x, min_positive_scores, 'o', label = "line 1")
    # plt.plot(x, max_negative_scores, 'x', label = "line 2")
    diff_score = min_positive_scores - max_negative_scores
    sorted_score = np.sort(diff_score)
    sorted_idxes = np.argsort(diff_score)
    min_positive_scores = min_positive_scores[sorted_idxes]
    max_negative_scores = max_negative_scores[sorted_idxes]

    x = np.arange(1, data.shape[0] + 1)
    zeros_x = np.zeros(data.shape[0])
    plt.plot(sorted_score, '.')
    plt.plot(x, zeros_x, 'r')
    # plt.plot(x, min_positive_scores, '.', label = "min postive")
    # plt.plot(x, max_negative_scores, 'x', label = "max negative")

    # plt.plot(max_negative_scores, min_positive_scores, 'o')

    plt.show()

def show_min_max_score_2(data):
    min_positive_scores = data['min_p'].to_numpy()
    max_negative_scores = data['max_n'].to_numpy()

    neg_idxes = np.argsort(max_negative_scores)

    diff_score = min_positive_scores - max_negative_scores
    sorted_score = np.sort(diff_score)

    plt.plot(sorted_score, '.')

    plt.show()


# show_min_max_score(data=error_df)

# comparison = error_df[error_df['q_type'] == 'comparison']
# show_min_max_score(data=comparison)

# bridge = error_df[error_df['q_type'] == 'bridge']
# show_min_max_score(data=bridge)

em_df = error_df[error_df['sp_sent_type'] == 'em']
em_df = em_df[em_df['flag']]
print(em_df.shape)
show_min_max_score(data=em_df)

# for row_idx, row in em_df.iterrows():
#     if row['min_p'] < row['max_n']:
#         print(row_idx)
#         print(row)



