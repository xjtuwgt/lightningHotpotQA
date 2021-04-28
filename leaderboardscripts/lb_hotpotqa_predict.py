import logging
import os
import argparse
from os.path import join
import json
from leaderboardscripts.lb_hotpotqa_data_structure import DataHelper
from envs import OUTPUT_FOLDER, DATASET_FOLDER
import torch
from utils.gpu_utils import single_free_cuda
from leaderboardscripts.lb_ReaderModel import UnifiedHGNModel


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
    parser.add_argument('--testf_type', default=None, type=str, required=True)
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
    parser.add_argument('--test_log_steps', default=10, type=int)
    parser.add_argument('--cpu_num', default=24, type=int)
    parser.add_argument("--dev_gold_file",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_raw', 'hotpot_dev_distractor_v1.json'))

    parser.add_argument("--para_path",
                        type=str,
                        default=join(DATASET_FOLDER, 'data_processed/test_distractor', 'rerank_topk_3_long_low_long_multihop_para.json'))
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
    args.testf_type = '{}_{}'.format(args.testf_type, args.topk_para_num)


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
test_example_dict = helper.test_example_dict
test_feature_dict = helper.test_feature_dict
test_features = helper.test_features
test_data_loader = helper.hotpot_test_dataloader
#
# # # #########################################################################
# # # # Initialize Model
# # # ##########################################################################
model = UnifiedHGNModel(config=args)
model.to(args.device)
model.eval()

# gold = json.load(open(args.dev_gold_file, 'r'))
# para_data = json.load(open(args.para_path, 'r'))
# import itertools
# recall_list = []
# for idx, case in enumerate(gold):
#     key = case['_id']
#     supp_title_set = set([x[0] for x in case['supporting_facts']])
#     pred_paras = para_data[key]
#     # print('selected para {}'.format(pred_paras))
#     sel_para_names = set(itertools.chain.from_iterable(pred_paras))
#     # print('Gold para {}'.format(supp_title_set))
#     if supp_title_set.issubset(sel_para_names) and len(supp_title_set) == 2:
#         recall_list.append(1)
#     else:
#         recall_list.append(0)
# print('Recall = {}'.format(sum(recall_list)*1.0/len(para_data)))

# import numpy as np
# import logging
# import sys
# from utils.gpu_utils import single_free_cuda
#
# from os.path import join
# import torch
#
# # from csr_mhqa.argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
# from plmodels.jd_argument_parser import default_dev_parser, complete_default_dev_parser, json_to_argv
# from plmodels.pldata_processing import Example, InputFeatures, DataHelper
# from csr_mhqa.utils import load_encoder_model, eval_model
# from utils.jdevalUtil import jd_eval_model
#
# # from models.HGN import HierarchicalGraphNetwork
# from jdmodels.jdHGN import HierarchicalGraphNetwork
# from model_envs import MODEL_CLASSES
#
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
#
# #########################################################################
# # Initialize arguments
# ##########################################################################
# parser = default_dev_parser()
#
# logger.info("IN CMD MODE")
# args_config_provided = parser.parse_args(sys.argv[1:])
# if args_config_provided.config_file is not None:
#     argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
# else:
#     argv = sys.argv[1:]
# args = parser.parse_args(argv)
# args = complete_default_dev_parser(args)
#
# logger.info('-' * 100)
# logger.info('Input Argument Information')
# logger.info('-' * 100)
# args_dict = vars(args)
# for a in args_dict:
#     logger.info('%-28s  %s' % (a, args_dict[a]))
#
# #########################################################################
# # Read Data
# ##########################################################################
# helper = DataHelper(gz=True, config=args)
#
# # Set datasets
# dev_example_dict = helper.dev_example_dict
# dev_feature_dict = helper.dev_feature_dict
# # dev_dataloader = helper.dev_loader
# dev_dataloader = helper.hotpot_val_dataloader
#
# # #########################################################################
# # # Initialize Model
# # ##########################################################################
# config_class, model_encoder, tokenizer_class = MODEL_CLASSES[args.model_type]
# config = config_class.from_pretrained(args.encoder_name_or_path)
#
# encoder_path = join(args.exp_name, args.encoder_name) ## replace encoder.pkl as encoder
# model_path = join(args.exp_name, args.model_name) ## replace encoder.pkl as encoder
# logger.info("Loading encoder from: {}".format(encoder_path))
# logger.info("Loading model from: {}".format(model_path))
#
# if torch.cuda.is_available():
#     device_ids, _ = single_free_cuda()
#     device = torch.device('cuda:{}'.format(device_ids[0]))
# else:
#     device = torch.device('cpu')
#
# args.device = device
#
# encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
# model = HierarchicalGraphNetwork(config=args)
#
# if encoder_path is not None:
#     state_dict = torch.load(encoder_path)
#     print('loading parameter from {}'.format(encoder_path))
#     for key in list(state_dict.keys()):
#         if 'module.' in key:
#             state_dict[key.replace('module.', '')] = state_dict[key]
#             del state_dict[key]
#     encoder.load_state_dict(state_dict)
# if model_path is not None:
#     state_dict = torch.load(model_path)
#     print('loading parameter from {}'.format(model_path))
#     for key in list(state_dict.keys()):
#         if 'module.' in key:
#             state_dict[key.replace('module.', '')] = state_dict[key]
#             del state_dict[key]
#     model.load_state_dict(state_dict)
#
# encoder.to(args.device)
# model.to(args.device)
#
# encoder.eval()
# model.eval()
#
# #########################################################################
# # Evaluation
# ##########################################################################
# output_pred_file = join(args.exp_name, 'dev_pred.json')
# output_eval_file = join(args.exp_name, 'dev_eval.txt')
# output_score_file = join(args.exp_name, 'dev_score.json')
#
# metrics, threshold = jd_eval_model(args, encoder, model,
#                                 dev_dataloader, dev_example_dict, dev_feature_dict,
#                                 output_pred_file, output_eval_file, args.dev_gold_file, output_score_file=output_score_file)
# # metrics, threshold = eval_model(args, encoder, model,
# #                                 dev_dataloader, dev_example_dict, dev_feature_dict,
# #                                 output_pred_file, output_eval_file, args.dev_gold_file)
# print("Best threshold: {}".format(threshold))
# for key, val in metrics.items():
#     print("{} = {}".format(key, val))
#
# # import json
# # with open(output_score_file, 'r') as fp:
# #     data = json.load(fp)
# #     print(len(data))
# #     for x in data:
# #         print(x)
# #         print(data[x])