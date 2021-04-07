import numpy as np
from adaptive_threshold.atutils import distribution_feat, distribution_feat_extraction, parse_args
from os.path import join

if __name__ == '__main__':

    x = np.random.random(10)
    print(x)
    y = distribution_feat_extraction(scores=x, keep_num=True)
    print(len(y))
    z = distribution_feat(scores=x)
    print(len(z))

    args = parse_args()
    dev_score_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_score_data)
    print(dev_score_file_name)