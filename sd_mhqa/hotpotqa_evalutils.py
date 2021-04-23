from torch import nn
from sd_mhqa.hotpotqa_data_loader import IGNORE_INDEX
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from csr_mhqa.utils import MODEL_CLASSES
from sd_mhqa.hotpotqaUtils import case_to_features
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
import shutil

import torch
import os
import json
logger = logging.getLogger(__name__)

def compute_loss(args, batch, start, end, para, sent, q_type):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())

    loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))

    loss = loss_span + loss_type + loss_sup + loss_para

    if loss_span > 1000:
        logging.info('Hhhhhhhhhhhhhhhhh {}'.format((loss_span, loss_type, loss_sup, loss_para)))
        start_list = batch['y1'].tolist()
        mask = batch['context_mask']
        for x_idx, x in enumerate(start_list):
            print(x, start[x_idx][x], mask[x_idx][x])
        # logging.info(start)
        # logging.info(batch['y1'])
        # logging.info(criterion(start, batch['y1']))
        logging.info('*' * 10)
        # logging.info(end)
        end_list = batch['y2'].tolist()
        for x_idx, x in enumerate(end_list):
            print(x, end[x_idx][x], mask[x_idx][x])
        # logging.info(batch['y2'])
        # logging.info(criterion(end, batch['y2']))

    return loss, loss_span, loss_type, loss_sup, loss_para

def convert_to_tokens(tokenizer, examples, ids, y1, y2, q_type_prob):
    answer_dict, answer_type_dict = {}, {}
    answer_type_prob_dict = {}

    q_type = np.argmax(q_type_prob, 1)

    def get_ans_from_pos(qid, y1, y2):
        example = examples[qid]
        feature_list = case_to_features(case=example, train_dev=True)
        doc_input_ids = feature_list[0]

        final_text = " "
        if y1 < len(doc_input_ids) and y2 < len(doc_input_ids):
            answer_input_ids = doc_input_ids[y1:y2]
            final_text = tokenizer.decode(answer_input_ids)
        return final_text

    for i, qid in enumerate(ids):
        if q_type[i] == 2:
            answer_text = get_ans_from_pos(qid, y1[i], y2[i])
        elif q_type[i] == 0:
            answer_text = 'yes'
        elif q_type[i] == 1:
            answer_text = 'no'
        else:
            raise ValueError("question type error")

        answer_dict[qid] = answer_text
        answer_type_prob_dict[qid] = q_type_prob[i].tolist()
        answer_type_dict[qid] = q_type[i].item()

    return answer_dict, answer_type_dict, answer_type_prob_dict

def jd_hotpotqa_eval_model(args, encoder, model, dataloader, example_dict, prediction_file, eval_file, dev_gold_file, output_score_file=None):
    _, _, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case)
    # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    encoder.eval()
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}

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
            start, end, q_type, paras, sent, yp1, yp2 = model(encoder, batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(tokenizer, example_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)

        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = [[] for _ in range(N_thresh)]
            cur_id = batch['ids'][i]

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
                          'type': answer_type_dict,
                          'type_prob': answer_type_prob_dict}
            tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp.json')
            with open(tmp_file, 'w') as f:
                json.dump(prediction, f)
            metrics = hotpot_eval(tmp_file, dev_gold_file)
            if metrics['joint_f1'] >= best_joint_f1:
                best_joint_f1 = metrics['joint_f1']
                best_threshold = thresholds[thresh_i]
                best_metrics = metrics
                shutil.move(tmp_file, pred_file)

        return best_metrics, best_threshold

    best_metrics, best_threshold = choose_best_threshold(answer_dict, prediction_file)
    json.dump(best_metrics, open(eval_file, 'w'))

    return best_metrics, best_threshold
