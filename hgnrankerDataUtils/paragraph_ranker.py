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
    rank_paras_dict = {}
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
            # print('original para names {}'.format(para_names_i))
            para_score_i = predict_support_para_np[i]
            para_mask_i = support_para_mask_np[i]
            para_num = int(para_mask_i.sum())
            ##+++++++++
            ctx_titles = para_names_i[:para_num]
            predicted_scores = para_score_i[:para_num]
            title_score_pair_list = list(zip(ctx_titles, predicted_scores))
            title_score_pair_list.sort(key=lambda x: x[1], reverse=True)
            rank_paras_dict[cur_id] = title_score_pair_list
            ##+++++++++
            selected_idxes = [0] * len(para_names_i)
            # print('para num = {}'.format(para_num))
            para_score_i[para_mask_i == 0] = -1e6
            sorted_idxes = np.argsort(para_score_i)[::-1].tolist()
            if para_num < 2:
                selected_idxes[sorted_idxes[0]] = 1
            else:
                if topk >= para_num:
                    topk_sorted_idxes = sorted_idxes[:para_num]
                else:
                    topk_sorted_idxes = sorted_idxes[:topk]
                # print(topk_sorted_idxes, para_num)
                for x in topk_sorted_idxes:
                    selected_idxes[x] = 1
            selected_para_names = []
            for x_idx, x in enumerate(selected_idxes):
                if x == 1:
                    selected_para_names.append(para_names_i[x_idx])
            print('selected para names = {}'.format(selected_para_names))
            sel_paras = []
            if len(selected_para_names) < 2:
                sel_paras.append([selected_para_names[0], selected_para_names[0]])
                sel_paras.append([])
                sel_paras.append([])
            else:
                sel_paras.append(selected_para_names[:2])
                sel_paras.append([])
                sel_paras.append(selected_para_names[2:])
            print('tuple result = {}'.format(sel_paras))
            # if para_num < 2:
            #     sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[0]]], [], [])
            # else:
            #     topk_paras = []
            #     for i in range(para_num):
            #         sel_idx = sorted_idxes[i]
            #         if len(topk_paras) < topk:
            #             topk_paras.append(para_names_i[sel_idx])
            #     assert len(topk_paras) <= topk and len(topk_paras) <= para_num and para_num <=4
            #     sel_paras=[topk_paras[:2], [], topk_paras[2:]]

                # if topk == 2 and para_num >=2:
                #     sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [])
                # elif topk == 3:
                #     if para_num > 2:
                #         sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [para_names_i[sorted_idxes[2]]])
                #     else:
                #         sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [])
                # else:
                #     if
                #     sel_paras = ([para_names_i[sorted_idxes[0]], para_names_i[sorted_idxes[1]]], [], [para_names_i[sorted_idxes[2]], para_names_i[sorted_idxes[3]]])
            prediction_para_dict[cur_id] = (sel_paras[0], sel_paras[1], sel_paras[2])

    recall_list = []
    if gold_file is not None:
        with open(gold_file) as f:
            gold = json.load(f)
        for idx, case in enumerate(gold):
            key = case['_id']
            supp_title_set = set([x[0] for x in case['supporting_facts']])
            pred_paras = prediction_para_dict[key]
            # print('selected para {}'.format(pred_paras))
            sel_para_names = set(itertools.chain.from_iterable(pred_paras))

            # print('Gold para {}'.format(supp_title_set))
            if supp_title_set.issubset(sel_para_names) and len(supp_title_set) == 2:
                recall_list.append(1)
            else:
                recall_list.append(0)
        print('Recall = {}'.format(sum(recall_list)*1.0/len(prediction_para_dict)))

    return prediction_para_dict, rank_paras_dict