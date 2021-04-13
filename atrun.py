import numpy as np
from adaptive_threshold.atutils import distribution_feat, distribution_feat_extraction, \
    parse_args, feat_label_extraction, save_numpy_array, load_npz_data_for_classification, load_npz_data, \
    adaptive_threshold_to_classification
from adaptive_threshold.ATModel import xgboost_model_train, save_sklearn_pickle_model, load_sklearn_pickle_model
from sklearn.metrics import confusion_matrix, accuracy_score
from os.path import join
import json

def dev_data_collection(args):
    dev_raw_data_file_name = join(args.input_dir, args.raw_dev_data)
    dev_score_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_score_name)
    x_feats, y_value, y_n_np_value, y_np_value, x_feat_dict = feat_label_extraction(raw_data_name=dev_raw_data_file_name, score_data_name=dev_score_file_name, train_type=args.train_type, train=False)
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    dev_json_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_json_name)
    save_numpy_array(x_feats=x_feats, y=y_value, y_n=y_n_np_value, y_np=y_np_value, npz_file_name=dev_npz_file_name)
    print('Saving dev data into {}'.format(dev_npz_file_name))
    json.dump(x_feat_dict, open(dev_json_file_name, 'w'))
    print('Saving dev data into {}'.format(dev_json_file_name))

def train_data_collection(args, train_filter):
    train_raw_data_file_name = join(args.input_dir, args.raw_train_data)
    train_score_file_name = join(args.pred_dir, args.model_name_or_path, args.train_type + '_' + args.train_score_name)
    if train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
    x_feats, y_value, y_n_np_value, y_np_value, _ = feat_label_extraction(raw_data_name=train_raw_data_file_name, score_data_name=train_score_file_name,
                                             train_type=args.train_type, train=True, train_filter=train_filter)
    save_numpy_array(x_feats=x_feats, y=y_value, y_n=y_n_np_value, y_np=y_np_value, npz_file_name=train_npz_file_name)
    print('Saving train data into {}'.format(train_npz_file_name))


def train_dev_map_to_classification(args, train_filter, threshold_category):
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    if train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)

    dev_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_class_name)
    if train_filter:
        train_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_class_name)
    else:
        train_class_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_class_name)

    class_label_dict = adaptive_threshold_to_classification(train_npz_file_name=train_npz_file_name, dev_npz_file_name=dev_npz_file_name,
                                         threshold_category=threshold_category, train_npz_class_file_name=train_class_npz_file_name,
                                         dev_npz_class_file_name=dev_class_npz_file_name)
    if train_filter:
        class_label_dict_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.class_label_map_name)
    else:
        class_label_dict_file_name = join(args.pred_dir, args.model_name_or_path, args.class_label_map_name)
    for key, value in class_label_dict.items():
        print(key, value)
    json.dump(class_label_dict, open(class_label_dict_file_name, 'w'))


# def train_and_evaluation_at(args, params, train_filter):
#     for key, value in params.items():
#         print('Parameter {} = {}'.format(key, value))
#     print('*' * 75)
#     if train_filter:
#         train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
#     else:
#         train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
#     dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
#     train_x, _, _, train_y_np = load_npz_data(npz_file_name=train_npz_file_name)
#     print('Loading x: {} and y: {} from {}'.format(train_x.shape, train_y_np.shape, train_npz_file_name))
#     dev_x, _, _, dev_y_np = load_npz_data(npz_file_name=dev_npz_file_name)
#     print('Loading x: {} and y: {} from {}'.format(dev_x.shape, dev_y_np.shape, dev_npz_file_name))
#     reg = at_boostree_model_train(X=train_x, y=train_y_np, params=params)
#     mse = mean_squared_error(dev_y_np, reg.predict(dev_x))
#     print('Evaluation mse = {}'.format(mse))
#
#     if train_filter:
#         pickle_model_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_n_est_' + str(params['n_estimators']) + '_depth_' +str(params['max_depth']) + args.pickle_model_name)
#     else:
#         pickle_model_file_name = join(args.pred_dir, args.model_name_or_path,
#                                       'n_est_' + str(params['n_estimators']) + '_depth_' +str(params['max_depth']) + args.pickle_model_name)
#     save_sklearn_pickle_model(model=reg, pkl_filename=pickle_model_file_name)
#     load_reg = load_sklearn_pickle_model(pkl_filename=pickle_model_file_name)
#     mse = mean_squared_error(dev_y_np, load_reg.predict(dev_x))
#     print('Evaluation mse on loaded model = {}'.format(mse))


def xgboost_train_and_evaluation(args, params, train_filter):
    for key, value in params.items():
        print('Parameter {} = {}'.format(key, value))
    print('*' * 75)
    if train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_class_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_class_name)
    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_class_name)
    train_x, _, _, _, train_y_label = load_npz_data_for_classification(npz_file_name=train_npz_file_name)
    print('Loading x: {} and y: {} from {}'.format(train_x.shape, train_y_label.shape, train_npz_file_name))
    dev_x, _, _, _, dev_y_label = load_npz_data_for_classification(npz_file_name=dev_npz_file_name)
    print('Loading x: {} and y: {} from {}'.format(dev_x.shape, dev_y_label.shape, dev_npz_file_name))

    xgbc = xgboost_model_train(X=train_x, y=train_y_label, params=params)

    ypred = xgbc.predict(dev_x)
    cm = confusion_matrix(ypred, dev_y_label)
    print(cm)
    print(type(cm))
    accuracy = accuracy_score(ypred, dev_y_label)
    print(accuracy)
    for key, value in params.items():
        print('Parameter {} = {}'.format(key, value))
    print('*' * 75)

    if train_filter:
        pickle_model_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_n_est_' + str(params['n_estimators']) + '_depth_' +str(params['max_depth']) + args.pickle_model_name)
    else:
        pickle_model_file_name = join(args.pred_dir, args.model_name_or_path,
                                      'n_est_' + str(params['n_estimators']) + '_depth_' +str(params['max_depth']) + args.pickle_model_name)
    save_sklearn_pickle_model(model=xgbc, pkl_filename=pickle_model_file_name)
    xgbc_model = load_sklearn_pickle_model(pkl_filename=pickle_model_file_name)
    model_ypred = xgbc_model.predict(dev_x)
    accuracy = accuracy_score(model_ypred, dev_y_label)
    print('Load model acc: {}'.format(accuracy))


# def prediction(args):
#     dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
#     dev_x, _, _, dev_y_np = load_npz_data(npz_file_name=dev_npz_file_name)
#     pickle_model_name = join(args.pred_dir, args.model_name_or_path, args.pickle_model_check_point_name)
#     load_reg = load_sklearn_pickle_model(pkl_filename=pickle_model_name)
#     pred_y = load_reg.predict(dev_x)
#     count = 0
#     for i in range(dev_y_np.shape[0]):
#         print('{}\t{:.5f}\t{:.5f}'.format(i + 1, dev_y_np[i], pred_y[i]))
#         if pred_y[i] < 0.45:
#             count = count + 1
#             print('*' * 100)
#     mse = mean_squared_error(dev_y_np, load_reg.predict(dev_x))
#     print(np.mean(pred_y), np.mean(dev_y_np))
#     print(count)
#     print('Evaluation mse on loaded model = {}'.format(mse))
#
# def prediction_analysis(args):
#     dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
#     dev_x, _, _, dev_y_np = load_npz_data(npz_file_name=dev_npz_file_name)
#     pickle_model_name = join(args.pred_dir, args.model_name_or_path, args.pickle_model_check_point_name)
#     load_reg = load_sklearn_pickle_model(pkl_filename=pickle_model_name)
#     pred_y = load_reg.predict(dev_x)
#     count = 0
#     for i in range(dev_y_np.shape[0]):
#         print('{}\t{:.5f}\t{:.5f}'.format(i + 1, dev_y_np[i], pred_y[i]))
#         if pred_y[i] < 0.45:
#             count = count + 1
#             print('*' * 100)
#     mse = mean_squared_error(dev_y_np, load_reg.predict(dev_x))
#     print(np.mean(pred_y), np.mean(dev_y_np))
#     print(count)
#     print('Evaluation mse on loaded model = {}'.format(mse))
#
# def json_prediction(args):
#     dev_json_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_json_name)
#     with open(dev_json_file_name, 'r', encoding='utf-8') as reader:
#         json_data = json.load(reader)
#     pickle_model_name = join(args.pred_dir, args.model_name_or_path, args.pickle_model_check_point_name)
#     load_reg = load_sklearn_pickle_model(pkl_filename=pickle_model_name)
#     count = 0
#     pred_threshold_dict = {}
#     for row_idx, row in enumerate(json_data):
#         x_feat = np.array(json_data[row]).reshape(1, -1)
#         pred_threshold = load_reg.predict(x_feat)
#         pred_threshold_dict[row] = pred_threshold[0]
#         if pred_threshold < 0.45:
#             count = count + 1
#     threshold_pred_json_name = join(args.pred_dir, args.model_name_or_path, args.pred_threshold_json_name)
#     print(count)
#     json.dump(pred_threshold_dict, open(threshold_pred_json_name, 'w'))
#     print('Saving threshold data {} into {}'.format(len(pred_threshold_dict), threshold_pred_json_name))

if __name__ == '__main__':

    args = parse_args()
    ### step 1: data collection
    # threshold_category = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    # threshold_category = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
    # # # threshold_category = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    # dev_data_collection(args=args)
    # train_data_collection(args=args, train_filter=False)
    # train_dev_map_to_classification(args=args, train_filter=False, threshold_category=threshold_category)
    #
    # train_data_collection(args=args, train_filter=True)
    # train_dev_map_to_classification(args=args, train_filter=True, threshold_category=threshold_category)

    ## step 2: model training and evaluation
    param = {
        'max_depth': 6,  # the maximum depth of each tree
        'n_estimators': 500,
        'learning_rate': 0.005,
        'eta': 0.3,  # the training step for each iteration
        'verbosity': 2,  # logging mode - quiet
        'use_label_encoder': False,
        'objective': 'multi:softprob',  # error evaluation for multiclass training
        'eval_metric': 'mlogloss',
        'num_class': 20}  # the number of classes that exist in this datset
    xgboost_train_and_evaluation(args=args, params=param, train_filter=False)

    # dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    # dev_x, dev_y, dev_y_np = load_npz_data(npz_file_name=dev_npz_file_name)
    # for i in range(dev_y.shape[0]):
    #     print(i, dev_y[i], dev_y_np[i])


    # # args.pickle_model_check_point_name = 'n_est_1500_at_pred_model.pkl'
    # args.pickle_model_check_point_name = 'filter_n_est_2000_depth_4at_pred_model.pkl'
    # # prediction(args=args)
    # json_prediction(args=args)