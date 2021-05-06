from post_feature_collection.range_distribution_analysis import *
import scipy.stats as sci_stat

def feat_correlation_analysis(x_feat, y_label, sigmoid=False, x_norm=False):
    if sigmoid:
        y_min = np_sigmoid(y_label[:,1])
        y_max = np_sigmoid(y_label[:,2])
        y_mean = (y_min + y_max)/2
        threshold = 0.5
        y_thr = np.ones(y_label.shape[0])
        y_thr[y_min >= threshold] = 0
        y_thr[y_max <= threshold] = 0
    else:
        y_min = y_label[:,1]
        y_max = y_label[:,2]
        y_mean = (y_min + y_max)/2
        threshold = 0
        y_thr = np.ones(y_label.shape[0])
        y_thr[y_min >= threshold] = 0
        y_thr[y_max <= threshold] = 0

    # plt.hist(x=y_mean, bins='auto', color='#0504aa',
    #          alpha=0.7, rwidth=0.85)
    correlation_np = np.zeros(x_feat.shape[1])
    x_feat_mean = np.mean(x_feat, axis=1)
    x_feat_std = np.std(x_feat, axis=1)

    for idx in range(x_feat.shape[1]):
        if x_norm:
            feat_i = (x_feat[:,idx] - x_feat_mean[idx]) /(x_feat_std[idx] + 1e-6)
        else:
            feat_i = x_feat[:,idx]
        pear_i = sci_stat.pearsonr(feat_i, y_thr)[0]
        correlation_np[idx] = pear_i
    sorted_idxes = np.argsort(correlation_np)
    for x in sorted_idxes:
        print(x, correlation_np[x])

    # for idx in range(x_feat.shape[1]):
    #     print('{}\t{}'.format(idx, correlation_np[idx]))
    plt.plot(x_feat[:,sorted_idxes[-1]], y_thr, '.')
    x_max = x_feat[:,sorted_idxes[-1]]
    print(sum(x_max > 15))

    plt.show()

# x_feat_np, y_label_np = train_range_analysis()
x_feat_np, y_label_np = dev_range_analysis()

print(x_feat_np.shape)
print(y_label_np.shape)

feat_correlation_analysis(x_feat=x_feat_np, y_label=y_label_np, sigmoid=True)