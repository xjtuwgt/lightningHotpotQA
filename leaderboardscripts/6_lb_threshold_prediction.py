import argparse
from envs import OUTPUT_FOLDER, DATASET_FOLDER
from os.path import join
import torch
from leaderboardscripts.lb_postprocess_model import RangeModel
from leaderboardscripts.lb_hotpotqa_evaluation import jd_adaptive_threshold_prediction


def parse_args():
    parser = argparse.ArgumentParser(
        description='Adaptive threshold prediction')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--raw_dev_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--raw_train_data", type=str, default='data_raw/hotpot_train_v1.1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--pred_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--exp_name",
                        type=str,
                        default='albert_orig',
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--test_feat_json_name", type=str, default='test_feat.json')
    parser.add_argument("--pred_threshold_json_name", type=str, default='test_pred_thresholds.json')

    parser.add_argument("--pickle_model_name", type=str, default='threshold_pred_model.pkl')
    parser.add_argument("--test_batch_size", type=int, default=1024, help='evaluation batch size')

    parser.add_argument("--cls_emb_dim", type=int, default=300, help='cls_emb_dim')
    parser.add_argument("--emb_dim", type=int, default=338, help='cls_emb_dim')
    parser.add_argument("--hid_dim", type=int, default=512, help='cls_emb_dim')
    parser.add_argument("--cpu_number", type=int, default=4, help='cpu number')
    parser.add_argument("--feat_drop", type=float, default=0.3, help='feature dropout ratio')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    model = RangeModel(args=args)
    pickle_file_name = join(args.output_dir, args.exp_name, args.pickle_model_name)
    model.load_state_dict(torch.load(pickle_file_name))
    test_feature_file_name = join(args.output_dir, args.exp_name, args.test_feat_json_name)
    pred_thresh_dict = jd_adaptive_threshold_prediction(args=args, model=model, feat_dict_file_name=test_feature_file_name)
    pred_threshold_json_name = join(args.pred_dir, args.exp_name, args.pred_threshold_json_name)