# coding=utf-8
#/usr/bin/env python3
import os
import argparse
import torch
import json
import logging
import random
import numpy as np
from os.path import join
from utils.gpu_utils import gpu_id_setting

from envs import DATASET_FOLDER, OUTPUT_FOLDER, PRETRAINED_MODEL_FOLDER
logger = logging.getLogger(__name__)

def boolean_string(s):
    if s.lower() not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s.lower() == 'true'

def json_to_argv(json_file):
    j = json.load(open(json_file))
    argv = []
    for k, v in j.items():
        new_v = str(v) if v is not None else None
        argv.extend(['--' + k, new_v])
    return argv

def train_parser():
    parser = argparse.ArgumentParser(
        description='Adaptive threshold prediction')
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    ##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    parser.add_argument("--raw_dev_data", type=str, default='data_raw/hotpot_dev_distractor_v1.json')
    parser.add_argument("--raw_train_data", type=str, default='data_raw/hotpot_train_v1.1.json')
    parser.add_argument("--input_dir", type=str, default=DATASET_FOLDER, help='define output directory')
    parser.add_argument("--output_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    parser.add_argument("--pred_dir", type=str, default=OUTPUT_FOLDER, help='define output directory')
    # Other parameters
    parser.add_argument("--train_type", type=str, default='hgn_low_sae', help='data type')
    parser.add_argument("--model_name_or_path",
                        default='train.graph.roberta.bs2.as1.lr2e-05.lrslayer_decay.lrd0.9.gnngat1.4.datahgn_docred_low_saeRecAdam.cosine.seed103',
                        type=str,
                        help="Path to pre-trained model")
    parser.add_argument("--dev_score_name", type=str, default='dev_score.json')
    parser.add_argument("--train_score_name", type=str, default='train_score.json')

    parser.add_argument("--dev_feat_name", type=str, default='dev_np_data.npz')
    parser.add_argument("--dev_feat_class_name", type=str, default='dev_class_np_data.npz')
    parser.add_argument("--dev_feat_json_name", type=str, default='dev_json_data.json')
    parser.add_argument("--train_feat_name", type=str, default='train_np_data.npz')
    parser.add_argument("--train_feat_class_name", type=str, default='train_class_np_data.npz')
    parser.add_argument("--class_label_map_name", type=str, default='class_label_dict.json')
    parser.add_argument("--pred_threshold_json_name", type=str, default='pred_thresholds.json')

    parser.add_argument("--pickle_model_name", type=str, default='at_pred_model.pkl')
    parser.add_argument("--pickle_model_check_point_name", type=str, help='checkpoint name')
    parser.add_argument("--rand_seed", type=int, default=4321)

    parser.add_argument("--train_batch_size", type=int, default=4096, help='training batch size')
    parser.add_argument("--eval_batch_size", type=int, default=1024, help='evaluation batch size')

    parser.add_argument("--cls_emb_dim", type=int, default=300, help='cls_emb_dim')
    parser.add_argument("--emb_dim", type=int, default=375, help='cls_emb_dim')
    parser.add_argument("--hid_dim", type=int, default=512, help='cls_emb_dim')

    parser.add_argument("--train_filter", type=bool, default=False, help='whether performing train filter')
    parser.add_argument("--cpu_number", type=int, default=12, help='cpu number')

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=1e-5, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30, type=int,
                        help="epochs")
    parser.add_argument('--eval_interval_ratio', type=float, default=0.1,
                        help="evaluate every X updates steps.")
    parser.add_argument("--feat_drop", type=float, default=0.3, help='feature dropout ratio')
    args = parser.parse_args()
    return args