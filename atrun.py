import numpy as np
from adaptive_threshold.atutils import distribution_feat, distribution_feat_extraction, \
    parse_args, feat_label_extraction, save_numpy_array, load_npz_data
from adaptive_threshold.ATModel import at_boostree_model_train
from os.path import join
from sklearn.metrics import mean_squared_error
import json

def dev_data_collection(args):
    dev_raw_data_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_score_name)
    x_feats, y_value = feat_label_extraction(raw_data_name=dev_raw_data_file_name, score_data_name=dev_score_file_name, train_type=args.train_type, train=False)
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    save_numpy_array(x_feats=x_feats, y=y_value, npz_file_name=dev_npz_file_name)
    print('Saving dev data into {}'.format(dev_npz_file_name))

def train_data_collection(args):
    train_raw_data_file_name = join(args.input_dir, args.raw_train_data)
    train_score_file_name = join(args.pred_dir, args.model_name_or_path, args.train_type + '_' + args.train_score_name)
    train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
    x_feats, y_value = feat_label_extraction(raw_data_name=train_raw_data_file_name, score_data_name=train_score_file_name,
                                             train_type=args.train_type, train=True)
    save_numpy_array(x_feats=x_feats, y=y_value, npz_file_name=train_npz_file_name)
    print('Saving train data into {}'.format(train_npz_file_name))

def train_and_evaluation_at(args, params):
    train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    train_x, train_y = load_npz_data(npz_file_name=train_npz_file_name)
    print('Loading x: {} and y: {} from {}'.format(train_x.shape, train_y.shape, train_npz_file_name))
    dev_x, dev_y = load_npz_data(npz_file_name=dev_npz_file_name)
    print('Loading x: {} and y: {} from {}'.format(dev_x.shape, dev_y.shape, dev_npz_file_name))
    reg = at_boostree_model_train(X=train_x, y=train_y, params=params)
    mse = mean_squared_error(dev_y, reg.predict(dev_x))
    print(mse)

if __name__ == '__main__':

    args = parse_args()
    # dev_data_collection(args=args)
    # train_data_collection(args=args)

    params = {'n_estimators': 1000,
              'max_depth': 4,
              'min_samples_split': 5,
              'learning_rate': 0.002,
              'verbose': True,
              'random_state': 1,
              'loss': 'ls'}
    train_and_evaluation_at(args=args, params=params)