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

x = np.arange(1, error_df.shape[0]+1)
min_positive_scores = error_df['min_p'].to_numpy()
max_negative_scores = error_df['max_n'].to_numpy()

# plt.plot(x, min_positive_scores, 'o', label = "line 1")
# plt.plot(x, max_negative_scores, 'x', label = "line 2")
diff_score = min_positive_scores - max_negative_scores
sorted_score = np.sort(diff_score)

plt.plot(sorted_score, '.')

plt.show()


