import logging
import os
import argparse
from os.path import join
import json
# from leaderboardscripts.lb_hotpotqa_data_structure import DataHelper
from plmodels.pldata_processing import DataHelper
from envs import OUTPUT_FOLDER, DATASET_FOLDER
import torch
from utils.gpu_utils import single_free_cuda
from leaderboardscripts.lb_ReaderModel import UnifiedHGNModel
from post_feature_collection.feature_collection_utils import jd_unified_post_feature_collection_model


logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Evaluating albert based reader Model')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--daug_type", default='long_low', type=str, help="Train Data augumentation type.")
    parser.add_argument("--devf_type", default='long_low', type=str, help="Dev data type")

    parser.add_argument('--output_dir',
                        type=str,
                        default=OUTPUT_FOLDER,
                        help='Directory to save model and summaries')
    parser.add_argument("--exp_name",
                        type=str,
                        default='albert_orig',
                        help="If set, this will be used as directory name in OUTOUT folder")
    parser.add_argument("--config_file",
                        type=str,
                        default=None,
                        help="configuration file for command parser")
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # parser.add_argument('--input_data', default=None, type=str, required=True)
    parser.add_argument("--encoder_ckpt", default='encoder.pkl', type=str)
    parser.add_argument("--model_ckpt", default='model.pkl', type=str)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # hyper-parameter
    parser.add_argument('--q_update', type=boolean_string, default='False', help='Whether update query')
    parser.add_argument("--trans_drop", type=float, default=0.2)
    parser.add_argument("--trans_heads", type=int, default=3)
    parser.add_argument("--do_rerank", action='store_true', help="Whether re-rank")


    # graph
    parser.add_argument('--num_edge_type', type=int, default=8)  ### number of edge types
    parser.add_argument('--mask_edge_types', type=str, default="0")  ### masked edge types

    parser.add_argument('--gnn', default='gat:1,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--gnn_drop", type=float, default=0.3)
    #########
    parser.add_argument("--gnn_attn_drop", type=float, default=0.3)
    #########
    parser.add_argument('--q_attn', type=boolean_string, default='True', help='whether use query attention in GAT')
    parser.add_argument("--lstm_drop", type=float, default=0.3)
    parser.add_argument("--lstm_layer", type=int, default=1)  ###++++++
    parser.add_argument('--graph_residual', type=boolean_string, default='True',
                        help='whether use residual connection in GAT')  ##+++++++++

    parser.add_argument("--max_para_num", default=5, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_ans_ent_num", default=15, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)

    # bi attn
    parser.add_argument('--ctx_attn', type=str, default='gate_att_up',
                        choices=['no_gate', 'gate_att_or', 'gate_att_up'])
    parser.add_argument("--ctx_attn_hidden_dim", type=int, default=300)
    parser.add_argument("--bi_attn_drop", type=float, default=0.3)
    parser.add_argument("--hidden_dim", type=int, default=300)
    # ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument('--topk_para_num', default=3, type=int, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # parser.add_argument("--eval_ckpt", default=None, type=str, required=True, help="evaluation checkpoint")
    parser.add_argument("--encoder_name_or_path",
                        default='albert-xxlarge-v2',
                        type=str,
                        help="Path to pre-trained model or shortcut name selected")
    parser.add_argument("--model_type", default='albert', type=str, help="alber reader model")
    parser.add_argument('--gpus', default=1, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    parser.add_argument('--eval_batch_size', default=16, type=int)
    parser.add_argument('--test_log_steps', default=10, type=int)
    parser.add_argument('--cpu_num', default=24, type=int)
    parser.add_argument("--dev_gold_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_dev_distractor_v1.json'))
    parser.add_argument("--train_gold_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_train_v1.1.json'))

    parser.add_argument("--para_path",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_processed/test_distractor', 'rerank_topk_4_long_low_long_multihop_para.json'))

    parser.add_argument("--data_type", type=str, required=True)
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    return parser.parse_args(args)

def complete_default_test_parser(args):
    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')
    args.device = device
    args.num_gnn_layers = int(args.gnn.split(':')[1].split(',')[0])
    args.num_gnn_heads = int(args.gnn.split(':')[1].split(',')[1])
    if len(args.mask_edge_types):
        args.mask_edge_types = list(map(int, args.mask_edge_types.split(',')))
    # TODO: only support albert-xxlarge-v2 now
    args.input_dim = 768 if 'base' in args.encoder_name_or_path else (4096 if 'albert' in args.encoder_name_or_path else 1024)
    # output dir name
    args.exp_name = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(args.exp_name, exist_ok=True)

    encoder_path = join(args.exp_name, args.encoder_ckpt)  ## replace encoder.pkl as encoder
    model_path = join(args.exp_name, args.model_ckpt)  ## replace encoder.pkl as encoder
    args.encoder_path = encoder_path
    args.model_path = model_path
    return args

#########################################################################
# Initialize arguments
##########################################################################
args = parse_args()
args = complete_default_test_parser(args=args)
if args.do_rerank:
    assert args.topk_para_num >= 2
    args.devf_type = '{}_reranker{}'.format(args.devf_type, args.topk_para_num)
    args.daug_type = '{}_reranker{}'.format(args.daug_type, args.topk_para_num)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)
# Set datasets
if args.data_type in ['train']:
    data_example_dict = helper.train_example_dict
    data_feature_dict = helper.train_feature_dict
    data_features = helper.train_features
    data_loader = helper.hotpot_train_dataloader
    gold_file = args.train_gold_file
    file_type = args.daug_type

elif args.data_type in ['dev_distractor']:
    data_example_dict = helper.dev_example_dict
    data_feature_dict = helper.dev_feature_dict
    data_features = helper.dev_features
    data_loader = helper.hotpot_val_dataloader
    gold_file = args.dev_gold_file
    file_type = args.devf_type
else:
    raise 'Wrong data type = {}'.format(args.data_type)
#
# # # #########################################################################
# # # # Initialize Model
# # # ##########################################################################
model = UnifiedHGNModel(config=args)
model.to(args.device)

output_pred_file = join(args.exp_name, '{}_{}_{}_pred.json'.format(args.data_type, args.max_para_num, args.topk_para_num))
output_eval_file = join(args.exp_name, '{}_{}_{}_eval.txt'.format(args.data_type, args.max_para_num, args.topk_para_num))
output_score_file = join(args.exp_name, '{}_{}_{}_score.json'.format(args.data_type, args.max_para_num, args.topk_para_num))

best_metrics, best_threshold = jd_unified_post_feature_collection_model(args, model, data_loader, data_example_dict, data_feature_dict,
                                output_pred_file, output_eval_file, gold_file, output_score_file=output_score_file)
for key, val in best_metrics.items():
    print("{} = {}".format(key, val))
print('Best threshold = {}'.format(best_threshold))