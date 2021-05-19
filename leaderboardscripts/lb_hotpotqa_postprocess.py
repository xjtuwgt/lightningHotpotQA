import logging
import argparse
from os.path import join
import torch
import json
from envs import OUTPUT_FOLDER, DATASET_FOLDER
from utils.gpu_utils import single_free_cuda
from leaderboardscripts.lb_postprocess_model import RangeSeqModel
from leaderboardscripts.lb_postprocess_utils import RangeDataset
from leaderboardscripts.lb_postprocess_utils import get_threshold_category
from leaderboardscripts.lb_hotpotqa_evaluation import jd_postprocess_score_prediction, jd_adaptive_threshold_post_process
from torch.utils.data import DataLoader


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Adaptive threshold prediction')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--raw_test_data", type=str, default='data_raw/hotpot_test_distractor_v1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--exp_name",
                        type=str,
                        default='albert_orig',
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--test_answer_predict_name", type=str, default='test_pred.json')
    parser.add_argument("--test_score_name", type=str, default='test_score.json')
    parser.add_argument("--test_feat_name", type=str, default='test_feature.json')
    parser.add_argument("--pred_threshold_name", type=str, default='pred_thresholds.json')

    parser.add_argument("--pickle_model_name", type=str, default='at_pred_model.pkl')
    parser.add_argument("--pickle_model_check_point_name", type=str, help='checkpoint name')
    parser.add_argument("--rand_seed", type=int, default=1234)
    parser.add_argument("--test_batch_size", type=int, default=1024, help='evaluation batch size')
    parser.add_argument("--span_window_size", type=int, default=170, help='span_window_size')
    parser.add_argument("--decoder_window_size", type=int, default=180, help='span_window_size')
    parser.add_argument("--encoder_type", type=str, default='ff',
                        help='the encoder type to fuse cls, and score: ff, conv, transformer')
    parser.add_argument("--encoder_layer", type=int, default=2,
                        help='number of layer in encoder')
    parser.add_argument("--encoder_hid_dim", type=int, default=512,
                        help='hid_dim of encoder')
    parser.add_argument("--encoder_drop_out", type=float, default=0.3,
                        help='hid_dim of encoder')
    parser.add_argument("--trim_drop_ratio", type=float, default=0.1,
                        help='trim drop ratio')

    parser.add_argument("--cls_emb_dim", type=int, default=300, help='cls_emb_dim')
    parser.add_argument("--emb_dim", type=int, default=344, help='cls_emb_dim')
    parser.add_argument("--hid_dim", type=int, default=512, help='cls_emb_dim')
    parser.add_argument("--cpu_number", type=int, default=6, help='cpu number')
    parser.add_argument("--interval_number", type=int, default=200, help='interval number')
    parser.add_argument("--alpha", type=float, default=0.05, help='prediction alpha')
    parser.add_argument("--feat_drop", type=float, default=0.3, help='feature dropout ratio')
    args = parser.parse_args()
    return args

args = parse_args()
if torch.cuda.is_available():
    device_ids, _ = single_free_cuda()
    device = torch.device('cuda:{}'.format(device_ids[0]))
else:
    device = torch.device('cpu')
args.device = device
logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))
print('-' * 100)

output_test_feature_file = join(args.output_dir, args.exp_name, args.test_feat_name)
output_test_score_file = join(args.output_dir, args.exp_name, args.test_score_name)
prediction_score_file = join(args.output_dir, args.exp_name, args.pred_threshold_name)
threshold_category = get_threshold_category(interval_num=args.interval_number)
# print(output_test_feature_file)
# print(output_test_score_file)
# print(prediction_score_file)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
test_data_set = RangeDataset(json_file_name=output_test_feature_file)
test_data_loader = DataLoader(dataset=test_data_set,
                                 shuffle=False,
                                 collate_fn=RangeDataset.collate_fn,
                                 batch_size=args.test_batch_size)

model = RangeSeqModel(args=args)
model.to(args.device)

for name, param in model.named_parameters():
    print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
print('-' * 75)
prediction_score_dict = jd_postprocess_score_prediction(args=args, model=model, data_loader=test_data_loader,
                                                        threshold_category=threshold_category)
# with open(prediction_score_file, 'w') as fp:
#     json.dump(prediction_score_dict, fp)
# print('Saving {} records into {}'.format(len(prediction_score_dict), prediction_score_file))
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# raw_test_data_file = join(args.input_dir, args.raw_test_data)
# test_answer_file = join(args.output_dir, args.exp_name, args.test_answer_predict_name)
# jd_adaptive_threshold_post_process(args=args, full_file=raw_test_data_file,
#                                    score_dict_file=output_test_score_file,
#                                    prediction_answer_file=test_answer_file,
#                                    threshold_pred_dict_file=prediction_score_file)

