from __future__ import absolute_import, division, print_function
import logging
import sys
from utils.gpu_utils import gpu_setting
from plmodels.jd_argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from plmodels.lightningHGN import lightningHGN
import torch.nn.functional as F
from csr_mhqa.utils import convert_to_tokens
import pytorch_lightning as pl
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
from tqdm import tqdm
import torch
import os
import json
import numpy as np
import shutil
from utils.jdutils import log_metrics
from time import time
from os.path import join
from jdevaluation.devdataHelper import DataHelper as DevDataHelper
from envs import OUTPUT_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
########################################################################################################################
def parse_args():
    parser = default_train_parser()
    logger.info("IN CMD MODE")
    args_config_provided = parser.parse_args(sys.argv[1:])
    if args_config_provided.config_file is not None:
        argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args = complete_default_train_parser(args)

    logger.info('-' * 100)
    logger.info('Input Argument Information')
    logger.info('-' * 100)
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))
    return args
########################################################################################################################
def device_setting(args):
    if torch.cuda.is_available():
        free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
        print('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
        if args.gpus > 0:
            gpu_ids = free_gpu_ids
            device = torch.device("cuda:%d" % gpu_ids[0])
            print('Single GPU setting')
        else:
            device = torch.device("cpu")
            print('Single cpu setting')
    else:
        device = torch.device("cpu")
        print('Single cpu setting')
    return device
########################################################################################################################
def dev_data_loader(args):
    dev_helper = DevDataHelper(gz=True, config=args)
    dev_data_loader = dev_helper.hotpot_val_dataloader
    dev_example_dict = dev_helper.dev_example_dict
    dev_feature_dict = dev_helper.dev_feature_dict
    return dev_data_loader, dev_feature_dict, dev_example_dict
########################################################################################################################
def batch2device(batch, device):
    for key, value in batch.items():
        if key not in {'ids'}:
            batch[key] = value.to(device)
    return batch
########################################################################################################################
def lightnHGN_test_procedure(model, test_data_loader, dev_feature_dict, dev_example_dict, args, device):
    model.freeze()
    out_puts = []
    start_time = time()
    total_steps = len(test_data_loader)
    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(test_data_loader)):
            batch = batch2device(batch=batch, device=device)
            start, end, q_type, paras, sents, ents, yp1, yp2 = model.forward(batch=batch)
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
            answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(dev_example_dict,
                                                                                        dev_feature_dict,
                                                                                        batch['ids'],
                                                                                        yp1.data.cpu().numpy().tolist(),
                                                                                        yp2.data.cpu().numpy().tolist(),
                                                                                        type_prob)
            predict_support_np = torch.sigmoid(sents[:, :, 1]).data.cpu().numpy()
            valid_dict = {'answer': answer_dict_, 'ans_type': answer_type_dict_, 'ids': batch['ids'],
                          'ans_type_pro': answer_type_prob_dict_, 'supp_np': predict_support_np}
            # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if (batch_idx + 1) % args.eval_batch_size == 0:
                print('Evaluating the model... {}/{} in {:.4f} seconds'.format(batch_idx + 1, total_steps, time()-start_time))
            out_puts.append(valid_dict)
            del batch
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

    thresholds = np.arange(0.1, 1.0, 0.025)
    N_thresh = len(thresholds)
    total_sp_dict = [{} for _ in range(N_thresh)]
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for batch_idx, valid_dict in tqdm(enumerate(out_puts)):
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = valid_dict['answer'], valid_dict['ans_type'], \
                                                                  valid_dict['ans_type_pro']
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = valid_dict['supp_np']
        batch_ids = valid_dict['ids']
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch_ids[i]

            for j in range(predict_support_np.shape[1]):
                if j >= len(dev_example_dict[cur_id].sent_names):
                    break
                for thresh_i in range(N_thresh):
                    if predict_support_np[i, j] > thresholds[thresh_i]:
                        cur_sp_pred[thresh_i].append(dev_example_dict[cur_id].sent_names[j])

            for thresh_i in range(N_thresh):
                if cur_id not in total_sp_dict[thresh_i]:
                    total_sp_dict[thresh_i][cur_id] = []
                total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

    def choose_best_threshold(ans_dict, pred_file):
        best_joint_f1 = 0
        best_metrics = None
        best_threshold = 0
        #################
        metric_dict = {}
        #################
        for thresh_i in range(N_thresh):
            prediction = {'answer': ans_dict,
                          'sp': total_sp_dict[thresh_i],
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, args.dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)
            #######
            metric_dict[thresh_i] = (
                metrics['em'], metrics['f1'], metrics['joint_em'], metrics['joint_f1'], metrics['sp_em'],
                metrics['sp_f1'])
            #######
        return best_metrics, best_threshold, metric_dict

    output_pred_file = os.path.join(args.exp_name, f'pred.json')
    output_eval_file = os.path.join(args.exp_name, f'eval.txt')
    ####+++++
    best_metrics, best_threshold, metric_dict = choose_best_threshold(answer_dict, output_pred_file)
    ####++++++
    logging.info('Leader board evaluation completed with threshold = {}'.format(best_threshold))
    log_metrics(mode='Evaluation', metrics=best_metrics)
    logging.info('*' * 75)
    ####++++++
    for key, value in metric_dict.items():
        logging.info('threshold {}: \t metrics: {}'.format(key, value))
    ####++++++
    json.dump(best_metrics, open(output_eval_file, 'w'))
    #############################################################################
    return best_metrics, best_threshold
########################################################################################################################
def main(args):
    device = device_setting(args=args)
    model_ckpt = join(OUTPUT_FOLDER, args.exp_name, 'HGN_hotpotQA-epoch=00-joint_f1=0.6507.ckpt')
    lighthgn_model = lightningHGN.load_from_checkpoint(checkpoint_path=model_ckpt)
    lighthgn_model = lighthgn_model.to(device)
    print('Model Parameter Configuration:')
    for name, param in lighthgn_model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    dev_data, dev_feature_dict, dev_example_dict = dev_data_loader(args=args)

    lightnHGN_test_procedure(model=lighthgn_model, test_data_loader=dev_data, dev_feature_dict=dev_feature_dict,
                             dev_example_dict=dev_example_dict, args=args, device=device)

if __name__ == '__main__':
    args = parse_args()
    main(args)
