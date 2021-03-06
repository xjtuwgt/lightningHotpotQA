import logging
import sys
import json
from utils.gpu_utils import single_free_cuda
from hgnrankerDataUtils.paragraph_ranker import para_ranker_model
from hgnrankerDataUtils.reranker_argument_parser import default_parser, complete_default_parser

from os.path import join
import torch

from plmodels.jd_argument_parser import json_to_argv
from plmodels.pldata_processing import Example, InputFeatures, DataHelper
from csr_mhqa.utils import load_encoder_model

from jdmodels.jdHGN import HierarchicalGraphNetwork
from model_envs import MODEL_CLASSES
from envs import DATASET_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

#########################################################################
# Initialize arguments
##########################################################################
parser = default_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
args = complete_default_parser(args)

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
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
# dev_dataloader = helper.dev_loader
dev_dataloader = helper.hotpot_val_dataloader

# #########################################################################
# # Initialize Model
# ##########################################################################
config_class, model_encoder, tokenizer_class = MODEL_CLASSES[args.model_type]
config = config_class.from_pretrained(args.encoder_name_or_path)

encoder_path = join(args.exp_name, args.encoder_name) ## replace encoder.pkl as encoder
model_path = join(args.exp_name, args.model_name) ## replace encoder.pkl as encoder
logger.info("Loading encoder from: {}".format(encoder_path))
logger.info("Loading model from: {}".format(model_path))

if torch.cuda.is_available():
    device_ids, _ = single_free_cuda()
    device = torch.device('cuda:{}'.format(device_ids[0]))
else:
    device = torch.device('cpu')

encoder, _ = load_encoder_model(args.encoder_name_or_path, args.model_type)
model = HierarchicalGraphNetwork(config=args)

if encoder_path is not None:
    state_dict = torch.load(encoder_path)
    print('loading parameter from {}'.format(encoder_path))
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict[key]
    encoder.load_state_dict(state_dict)
if model_path is not None:
    state_dict = torch.load(model_path)
    print('loading parameter from {}'.format(model_path))
    for key in list(state_dict.keys()):
        if 'module.' in key:
            state_dict[key.replace('module.', '')] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)

encoder.to(args.device)
model.to(args.device)

encoder.eval()
model.eval()

#########################################################################
# Evaluation
##########################################################################
if args.exp_name is not None:
    if 'seed' in args.exp_name:
        idx = args.exp_name.index('seed')
        model_name = args.exp_name[idx:] + args.model_type
    else:
        model_name = args.model_type
else:
    model_name = '' + args.model_type

selected_para_dict, para_rank_dict = para_ranker_model(args=args, encoder=encoder, model=model, dataloader=dev_dataloader, example_dict=dev_example_dict, topk=args.topk_para_num, gold_file=args.dev_gold_file)

output_pred_para_file = join(args.exp_name, 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_multihop_para.json')
json.dump(selected_para_dict, open(output_pred_para_file, 'w'))
print('Saving {} examples in {}'.format(len(selected_para_dict), output_pred_para_file))

output_rank_para_file = join(args.exp_name, 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_para_ranking.json')
json.dump(para_rank_dict, open(output_rank_para_file, 'w'))
print('Saving {} examples in {}'.format(len(para_rank_dict), output_rank_para_file))

data_processed_pred_para_file = join(DATASET_FOLDER, 'data_processed/dev_distractor', 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_multihop_para.json')
json.dump(selected_para_dict, open(data_processed_pred_para_file, 'w'))
print('Saving {} examples in {}'.format(len(selected_para_dict), data_processed_pred_para_file))

data_processed_rank_para_file = join(DATASET_FOLDER, 'data_processed/dev_distractor', 'rerank_' + model_name+'topk_' + str(args.topk_para_num) + '_' + args.devf_type + '_para_ranking.json')
json.dump(para_rank_dict, open(data_processed_rank_para_file, 'w'))
print('Saving {} examples in {}'.format(len(para_rank_dict), data_processed_rank_para_file))
# metrics, threshold = jd_eval_model(args, encoder, model,
#                                 dev_dataloader, dev_example_dict, dev_feature_dict,
#                                 output_pred_file, output_eval_file, args.dev_gold_file, output_score_file=output_score_file)
# metrics, threshold = eval_model(args, encoder, model,
#                                 dev_dataloader, dev_example_dict, dev_feature_dict,
#                                 output_pred_file, output_eval_file, args.dev_gold_file)
# print("Best threshold: {}".format(threshold))
# for key, val in metrics.items():
#     print("{} = {}".format(key, val))

# import json
# with open(output_score_file, 'r') as fp:
#     data = json.load(fp)
#     print(len(data))
#     for x in data:
#         print(x)
#         print(data[x])