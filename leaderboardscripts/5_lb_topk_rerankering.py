import logging
import os
import argparse
from os.path import join
from leaderboardscripts.lb_hotpotqa_data_structure import DataHelper
from envs import OUTPUT_FOLDER
import torch
from utils.gpu_utils import single_free_cuda
from leaderboardscripts.lb_ReaderModel import UnifiedHGNModel

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

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
                        default='albert',
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
    parser.add_argument('--num_edge_type', type=int, default=8) ### number of edge types
    parser.add_argument('--mask_edge_types', type=str, default="0") ### masked edge types
    parser.add_argument('--gnn', default='gat:1,2', type=str, help='gat:n_layer, n_head')
    parser.add_argument("--hidden_dim", type=int, default=300)
    # ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # parser.add_argument('--data_dir', default=None, type=str, required=True)
    parser.add_argument("--max_entity_num", default=60, type=int)
    parser.add_argument("--max_para_num", default=5, type=int)
    parser.add_argument("--max_sent_num", default=40, type=int)
    parser.add_argument("--max_query_length", default=50, type=int)
    parser.add_argument("--max_seq_length", default=512, type=int)
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
# for key, value in test_feature_dict.items():
#     print(value)
#     print(test_example_dict[key])
# for case in test_features:
#     print(case[])
test_data_loader = helper.hotpot_test_dataloader
#
# # # #########################################################################
# # # # Initialize Model
# # # ##########################################################################
# # config_class, model_encoder, tokenizer_class = MODEL_CLASSES[args.model_type]
# # config = config_class.from_pretrained(args.encoder_name_or_path)
#
# # print(test_example_dict)
for batch in test_data_loader:
    # print(batch['ids'])
    print(batch)

model = UnifiedHGNModel(config=args)
model.to(args.device)

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
# if args.exp_name is not None:
#     if 'seed' in args.exp_name:
#         idx = args.exp_name.index('seed')
#         model_name = args.exp_name[idx:] + args.model_type
#     else:
#         model_name = args.model_type
# else:
#     model_name = '' + args.model_type
#
# selected_para_dict, para_rank_dict = para_ranker_model(args=args, encoder=encoder, model=model, dataloader=dev_dataloader, example_dict=dev_example_dict, topk=args.topk_para_num, gold_file=args.dev_gold_file)
#
# output_pred_para_file = join(args.exp_name, 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_multihop_para.json')
# json.dump(selected_para_dict, open(output_pred_para_file, 'w'))
# print('Saving {} examples in {}'.format(len(selected_para_dict), output_pred_para_file))
#
# output_rank_para_file = join(args.exp_name, 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_para_ranking.json')
# json.dump(para_rank_dict, open(output_rank_para_file, 'w'))
# print('Saving {} examples in {}'.format(len(para_rank_dict), output_rank_para_file))
#
# data_processed_pred_para_file = join(DATASET_FOLDER, 'data_processed/dev_distractor', 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_multihop_para.json')
# json.dump(selected_para_dict, open(data_processed_pred_para_file, 'w'))
# print('Saving {} examples in {}'.format(len(selected_para_dict), data_processed_pred_para_file))
#
# data_processed_rank_para_file = join(DATASET_FOLDER, 'data_processed/dev_distractor', 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_para_ranking.json')
# json.dump(para_rank_dict, open(data_processed_rank_para_file, 'w'))
# print('Saving {} examples in {}'.format(len(para_rank_dict), data_processed_rank_para_file))
# # metrics, threshold = jd_eval_model(args, encoder, model,
# #                                 dev_dataloader, dev_example_dict, dev_feature_dict,
# #                                 output_pred_file, output_eval_file, args.dev_gold_file, output_score_file=output_score_file)
# # metrics, threshold = eval_model(args, encoder, model,
# #                                 dev_dataloader, dev_example_dict, dev_feature_dict,
# #                                 output_pred_file, output_eval_file, args.dev_gold_file)
# # print("Best threshold: {}".format(threshold))
# # for key, val in metrics.items():
# #     print("{} = {}".format(key, val))
#
# # import json
# # with open(output_score_file, 'r') as fp:
# #     data = json.load(fp)
# #     print(len(data))
# #     for x in data:
# #         print(x)
# #         print(data[x])