import json
import numpy as np
from tqdm import tqdm
import torch
import shutil
import os
from eval.hotpot_evaluate_v1 import normalize_answer, eval as hotpot_eval
from csr_mhqa.utils import convert_to_tokens
import torch.nn.functional as F

def jd_unified_test_model(args, model, dataloader, example_dict, feature_dict, prediction_file, eval_file, threshold=0.45, dev_gold_file=None, score_file=None):
    model.eval()

    answer_dict = {}
    answer_type_dict = {}
    answer_type_prob_dict = {}
    sp_dict = {}

    for batch in tqdm(dataloader):
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in {'ids'}:
                batch[key] = value.to(args.device)
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        with torch.no_grad():
            start, end, q_type, paras, sent, ent, yp1, yp2, cls_emb = model(batch, return_yp=True)

        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        answer_type_dict.update(answer_type_dict_)
        answer_type_prob_dict.update(answer_type_prob_dict_)
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sent[:, :, 1]).data.cpu().numpy()

        for i in range(predict_support_np.shape[0]):
            cur_id = batch['ids'][i]
            sp_dict[cur_id] = []

            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if predict_support_np[i, j] > threshold:
                    sp_dict[cur_id].append(example_dict[cur_id].sent_names[j])

    prediction = {'answer': answer_dict,
                  'sp': sp_dict,
                  'type': answer_type_dict,
                  'type_prob': answer_type_prob_dict}
    tmp_file = os.path.join(os.path.dirname(prediction_file), 'tmp.json')
    with open(tmp_file, 'w') as f:
        json.dump(prediction, f)
    metrics = hotpot_eval(tmp_file, dev_gold_file)
    json.dump(metrics, open(eval_file, 'w'))

    return metrics