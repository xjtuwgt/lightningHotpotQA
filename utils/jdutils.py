from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import torch
import os
import torch.nn.functional as F
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from utils.jdevalUtil import doc_recall_eval
import json
import shutil
from csr_mhqa.utils import convert_to_tokens
import logging
import string
from torch import nn
from csr_mhqa.data_processing import IGNORE_INDEX
import re
from hgntransformers import AdamW
from csr_mhqa.utils import get_optimizer
from utils.optimizerutils import RecAdam

import random
import os

def seed_everything(seed: int) -> int:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    return seed


def log_metrics(mode, metrics):
    '''
    Print the evaluation logs
    '''
    for metric in metrics:
        logging.info('{} {}: {:.4f}'.format(mode, metric, metrics[metric]))

def normalize_question(question: str) -> str:
    question = question
    if question[-1] == '?':
        question = question[:-1]
    return question


def normalize_text(s: str) -> str:
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_doc_prediction(predict_para_support_np_ith, example_dict, batch_ids_ith):
    arg_order_ids = np.argsort(predict_para_support_np_ith)[::-1].tolist()
    cand_para_names = example_dict[batch_ids_ith].para_names
    assert len(cand_para_names) >=2
    cur_sp_para_pred = [cand_para_names[arg_order_ids[0]], cand_para_names[arg_order_ids[1]]]
    return cur_sp_para_pred
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def supp_sent_prediction(predict_support_np_ith, example_dict, batch_ids_ith, thresholds):
    N_thresh = len(thresholds)
    cur_sp_pred = [[] for _ in range(N_thresh)]
    cur_id = batch_ids_ith
    arg_order_ids = np.argsort(predict_support_np_ith)[::-1].tolist()
    filtered_arg_order_ids = [_ for _ in arg_order_ids if _ < len(example_dict[cur_id].sent_names)]
    assert len(filtered_arg_order_ids) >= 2
    for thresh_i in range(N_thresh):
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[0]])
        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[filtered_arg_order_ids[1]])
    second_score = predict_support_np_ith[filtered_arg_order_ids[1]]
    for j in range(2, len(filtered_arg_order_ids)):
        jth_idx = filtered_arg_order_ids[j]
        for thresh_i in range(N_thresh):
            if predict_support_np_ith[jth_idx] > thresholds[thresh_i] * second_score:
                cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[jth_idx])
    return cur_sp_pred

def jd_eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.025)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]
    ##++++++++++++++++++++++++++++++++++
    total_para_sp_dict = {}
    ##++++++++++++++++++++++++++++++++++
    best_sp_dict = {}
    threshold_inter_count = 0
    ##++++++++++++++++++++++++++++++++++

    for batch in tqdm(dataloader):
        with torch.no_grad():
            inputs = {'input_ids':      batch['context_idxs'],
                      'attention_mask': batch['context_mask'],
                      'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet', 'electra'] else None}  # XLM don't use segment_ids
            outputs = encoder(**inputs)

            batch['context_encoding'] = outputs[0]
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start, end, q_type, paras, sent, ent, yp1, yp2 = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        para_mask = batch['para_mask']
        sent_mask = batch['sent_mask']
        # print(para_mask.shape, paras.shape)
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)
        ##++++++++++++++++++++++++++++++++++++++++
        paras = paras[:,:,1] - (1 - para_mask) * 1e30
        predict_para_support_np = torch.sigmoid(paras).data.cpu().numpy()
        # predict_para_support_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        ##++++++++++++++++++++++++++++++++++++++++
        # print('sent shape {}'.format(sent.shape))
        sent = sent[:,:,1] - (1 - sent_mask) * 1e30
        # predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()
        predict_support_np = torch.sigmoid(sent).data.cpu().numpy()
        # print('supp sent np shape {}'.format(predict_support_np.shape))
        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            predict_para_support_np_ith = predict_para_support_np[i]
            predict_support_np_ith = predict_support_np[i]
            # ################################################
            cur_para_sp_pred = supp_doc_prediction(predict_para_support_np_ith=predict_para_support_np_ith, example_dict=example_dict, batch_ids_ith=cur_id)
            total_para_sp_dict[cur_id] = cur_para_sp_pred
            # ################################################
            cur_sp_pred = supp_sent_prediction(predict_support_np_ith=predict_support_np_ith,
                                               example_dict=example_dict, batch_ids_ith=cur_id, thresholds=thresholds)
            # ###################################

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        #####
        best_threshold_idx = -1
        #####
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                #####
                best_threshold_idx = thresh_i
                #####
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold, best_threshold_idx

    best_metrics, best_threshold, best_threshold_idx = choose_best_threshold(answer_dict, prediction_file)
    ##############++++++++++++
    doc_recall_metric = doc_recall_eval(doc_prediction=total_para_sp_dict, gold_file=dev_gold_file)
    ##############++++++++++++
    json.dump(best_metrics, open(eval_file, 'w'))
    # -------------------------------------
    best_prediction = {'answer': answer_dict,
                  'sp': best_sp_dict,
                  'type': answer_type_dict,
                  'type_prob': answer_type_prob_dict}
    print('Number of inter threshold = {}'.format(threshold_inter_count))
    best_tmp_file = os.path.join(os.path.dirname(prediction_file), 'best_tmp.json')
    with open(best_tmp_file, 'w') as f:
        json.dump(best_prediction, f)
    best_th_metrics = hotpot_eval(best_tmp_file, dev_gold_file)
    for key, val in best_th_metrics.items():
        print("{} = {}".format(key, val))
    # -------------------------------------
    return best_metrics, best_threshold, doc_recall_metric


def get_group_decay_lr_optimizer(hgn_encoder, hgn_model, args, learning_rate):
    "Prepare optimizer and schedule (linear warmup and decay)"
    encoder_layer_number_dict = {'roberta-large': 24, 'albert-xxlarge-v2': 1}
    assert args.encoder_name_or_path in encoder_layer_number_dict

    def achieve_module_groups(encoder, number_of_layer, number_of_groups):
        layer_num_each_group = number_of_layer // number_of_groups
        number_of_divided_groups = number_of_groups + 1 if number_of_layer % number_of_groups > 0 else number_of_groups
        groups = []
        groups.append([encoder.embeddings, *encoder.encoder.layer[:layer_num_each_group]])
        for group_id in range(1, number_of_divided_groups):
            groups.append(
                [*encoder.encoder.layer[(group_id * layer_num_each_group):((group_id + 1) * layer_num_each_group)]])
        return groups, number_of_divided_groups

    if args.encoder_name_or_path == 'roberta-large':
        encoder_layer_number = encoder_layer_number_dict[args.encoder_name_or_path]
        encoder_group_number = 2
        module_groups, encoder_group_number = achieve_module_groups(encoder=hgn_encoder,
                                                                    number_of_layer=encoder_layer_number,
                                                                    number_of_groups=encoder_group_number)
        module_groups.append([hgn_model])
        assert len(module_groups) == encoder_group_number + 1
    elif args.encoder_name_or_path == 'albert-xxlarge-v2':
        module_groups = []
        module_groups.append([hgn_encoder])
        module_groups.append([hgn_model])
        assert len(module_groups) == 2
    else:
        raise 'Not supported {}'.format(args.encoder_name_or_path)

    def achieve_parameter_groups(module_group, weight_decay, lr):
        named_parameters = []
        no_decay = ["bias", "LayerNorm.weight"]
        for module in module_group:
            named_parameters += module.named_parameters()
        grouped_parameters = [
            {
                "params": [p for n, p in named_parameters if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": weight_decay, 'lr': lr
            },
            {
                "params": [p for n, p in named_parameters if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0, 'lr': lr
            }
        ]
        return grouped_parameters

    optimizer_grouped_parameters = []
    group_number = len(module_groups)
    for idx, module_group in enumerate(module_groups):
        lr = learning_rate * ((0.2 ** (group_number - idx - 1)))
        logging.info('group {} lr = {}'.format(idx, lr))
        grouped_parameters = achieve_parameter_groups(module_group=module_group,
                                                      weight_decay=args.weight_decay,
                                                      lr=lr)
        optimizer_grouped_parameters += grouped_parameters

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                      eps=args.adam_epsilon)
    return optimizer


def get_layer_wised_lr_optimizer(hgn_encoder, hgn_model, args, learning_rate):
    "Prepare optimizer and schedule (linear warmup and decay)"
    encoder_layer_number_dict = {'roberta-large': 24, 'albert-xxlarge-v2': 1}
    assert args.encoder_name_or_path in encoder_layer_number_dict

    def achieve_module_groups(encoder, number_of_layer, number_of_groups):
        layer_num_each_group = number_of_layer // number_of_groups
        number_of_divided_groups = number_of_groups + 1 if number_of_layer % number_of_groups > 0 else number_of_groups
        groups = []
        groups.append([encoder.embeddings, *encoder.encoder.layer[:layer_num_each_group]])
        for group_id in range(1, number_of_divided_groups):
            groups.append(
                [*encoder.encoder.layer[(group_id * layer_num_each_group):((group_id + 1) * layer_num_each_group)]])
        return groups, number_of_divided_groups

    if args.encoder_name_or_path == 'roberta-large':
        encoder_layer_number = encoder_layer_number_dict[args.encoder_name_or_path]
        encoder_group_number = encoder_layer_number
        module_groups, encoder_group_number = achieve_module_groups(encoder=hgn_encoder,
                                                                    number_of_layer=encoder_layer_number,
                                                                    number_of_groups=encoder_group_number)
        module_groups.append([hgn_model])
        assert len(module_groups) == encoder_group_number + 1
    elif args.encoder_name_or_path == 'albert-xxlarge-v2':
        module_groups = []
        module_groups.append([hgn_encoder])
        module_groups.append([hgn_model])
        assert len(module_groups) == 2
    else:
        raise 'Not supported {}'.format(args.encoder_name_or_path)

    def achieve_parameter_groups(module_group, weight_decay, lr):
        named_parameters = []
        no_decay = ["bias", "LayerNorm.weight"]
        for module in module_group:
            named_parameters += module.named_parameters()
        grouped_parameters = [
            {
                "params": [p for n, p in named_parameters if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": weight_decay, 'lr': lr
            },
            {
                "params": [p for n, p in named_parameters if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0, 'lr': lr
            }
        ]
        return grouped_parameters

    optimizer_grouped_parameters = []
    module_group_num = len(module_groups)
    for idx, module_group in enumerate(module_groups):
        lr = learning_rate * (args.layer_wise_lr_decay ** (module_group_num - idx - 1))
        logging.info('group {} lr = {}'.format(idx, lr))
        grouped_parameters = achieve_parameter_groups(module_group=module_group,
                                                      weight_decay=args.weight_decay,
                                                      lr=lr)
        optimizer_grouped_parameters += grouped_parameters

    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate,
                      eps=args.adam_epsilon)
    return optimizer


def get_lr_with_optimizer(encoder, model, args):
    learning_rate = args.learning_rate
    if args.optimizer == 'Adam':
        if args.learning_rate_schema == 'fixed':
            optimizer = get_optimizer(encoder, model, args, learning_rate, remove_pooler=False)
        elif args.learning_rate_schema == 'group_decay':
            optimizer = get_group_decay_lr_optimizer(hgn_encoder=encoder, hgn_model=model, args=args, learning_rate=learning_rate)
        elif args.learning_rate_schema == 'layer_decay':
            optimizer = get_layer_wised_lr_optimizer(hgn_encoder=encoder, hgn_model=model, args=args, learning_rate=learning_rate)
        else:
            raise 'Wrong lr setting method = {}'.format(args.learning_rate_schema)
    else:
        optimizer = get_rec_adam_optimizer(pretrained_model=encoder, new_model=model, args=args)
    return optimizer

def get_rec_adam_optimizer(pretrained_model, new_model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    if args.optimizer == 'RecAdam':
        # Prepare for the grouped parameters for RecAdam optimizer.
        # Since the classifier layer is not pretrained, it is not penalized during optimization.
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in new_model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": args.weight_decay,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": args.weight_decay,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": 0.0,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": 0.0,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            }
        ]
        optimizer = RecAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                            anneal_fun=args.recadam_anneal_fun, anneal_k=args.recadam_anneal_k,
                            anneal_t0=args.recadam_anneal_t0, pretrain_cof=args.recadam_pretrain_cof)
    else:
        raise 'error oprimizer {}'.format(args.optimizer)

    return optimizer


def compute_sent_loss(args, batch, sent):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())
    loss = loss_sup
    return [loss]


def supp_fact_eval_model(args, encoder, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, dev_gold_file):
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}

    # dataloader.refresh()

    thresholds = np.arange(0.1, 1.0, 0.05)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            inputs = {'input_ids':      batch['context_idxs'],
                      'attention_mask': batch['context_mask'],
                      'token_type_ids': batch['segment_idxs'] if args.model_type in ['bert', 'xlnet', 'electra'] else None}  # XLM don't use segment_ids
            outputs = encoder(**inputs)

            ####++++++++++++++++++++++++++++++++++++++
            if args.model_type == 'electra':
                batch['context_encoding'] = outputs.last_hidden_state
            else:
                batch['context_encoding'] = outputs[0]
            ####++++++++++++++++++++++++++++++++++++++
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            sent = model(batch, return_yp=True)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]
            ##++++
            answer_dict[cur_id] = 'yes'
            answer_type_dict[cur_id] = 'yes'
            ##++++

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break

                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []

                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['sp_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['sp_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold
