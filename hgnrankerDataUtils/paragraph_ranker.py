import numpy as np
from tqdm import tqdm
import torch
import json
import itertools

def para_ranker_model(args, encoder, model, dataloader, example_dict, topk=2, gold_file=None):
    #### model_type, device
    encoder.eval()
    model.eval()
    ##++++++
    prediction_para_dict = {}
    ##++++++
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
            batch['context_encoding'] = outputs[0]
            ####++++++++++++++++++++++++++++++++++++++
            batch['context_mask'] = batch['context_mask'].float().to(args.device)
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb= model(batch, return_yp=True, return_cls=True)
        ####################################################################
        predict_support_para_np = torch.sigmoid(paras[:, :, 1]).data.cpu().numpy()
        support_para_mask_np = batch['para_mask'].data.cpu().numpy()
        ####################################################################
        for i in range(predict_support_para_np.shape[0]):
            cur_id = batch['ids'][i]
            para_names_i = example_dict[cur_id].para_names
            para_score_i = predict_support_para_np[i]
            para_mask_i = support_para_mask_np[i]
            para_num = para_mask_i.sum()
            print(para_num)
            para_score_i[para_mask_i == 0] = -1e6
            sorted_idxes = np.argsort(para_score_i)[::-1]
            print(sorted_idxes)
            sorted_idxes = sorted_idxes.tolist()[:para_num]
            if topk == 2:
                sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [])
            elif topk == 3:
                sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [para_names_i[sorted_idxes[2]]])
            else:
                sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [para_names_i[sorted_idxes[2]], para_names_i[sorted_idxes[3]]])
            prediction_para_dict[cur_id] = sel_paras

    recall_list = []
    if gold_file is not None:
        with open(gold_file) as f:
            gold = json.load(f)
        for idx, case in enumerate(gold):
            key = case['_id']
            supp_title_set = set([x[0] for x in case['supporting_facts']])
            pred_paras = prediction_para_dict[key]
            sel_para_names = set(itertools.chain.from_iterable(pred_paras))
            if supp_title_set.issubset(sel_para_names):
                recall_list.append(1)
            else:
                recall_list.append(0)
        print('Recall = {}'.format(sum(recall_list)*1.0/len(prediction_para_dict)))

    return prediction_para_dict