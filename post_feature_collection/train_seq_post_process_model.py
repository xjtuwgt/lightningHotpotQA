from post_feature_collection.post_process_data_helper import RangeSeqDataset
from post_feature_collection.post_process_argument_parser import train_parser
from torch.utils.data import DataLoader
from os.path import join
import torch
import json
from leaderboardscripts.lb_postprocess_model import RangeSeqModel, seq_loss_computation
from tqdm import tqdm, trange
from adaptive_threshold.atutils import get_optimizer, get_scheduler
import random
import numpy as np
from utils.gpu_utils import single_free_cuda
from torch import Tensor
import torch
from post_feature_collection.post_process_feature_extractor import get_threshold_category, np_sigmoid, load_json_score_data, score_row_supp_f1_computation

def batch_analysis(x_feat: Tensor):
    p2dist = torch.cdist(x1=x_feat, x2=x_feat, p=2)
    print(p2dist)

def train(args):
    train_feat_file_name = join(args.output_dir, args.exp_name, args.train_feat_json_name)
    dev_feat_file_name = join(args.output_dir, args.exp_name, args.dev_feat_json_name)
    dev_score_file_name = join(args.output_dir, args.exp_name, args.dev_score_name)
    threshold_category = get_threshold_category(interval_num=args.interval_number)

    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')

    for key, value in vars(args).items():
        print('{}: {}'.format(key, value))

    ##+++++++++
    random_seed = args.rand_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    ##+++++++++
    train_data = RangeSeqDataset(json_file_name=train_feat_file_name)
    dev_data = RangeSeqDataset(json_file_name=dev_feat_file_name)
    train_data_loader = DataLoader(dataset=dev_data,
                                   shuffle=True,
                                   collate_fn=RangeSeqDataset.collate_fn,
                                   batch_size=args.train_batch_size)
    dev_data_loader = DataLoader(dataset=dev_data,
                                 shuffle=False,
                                 collate_fn=RangeSeqDataset.collate_fn,
                                 batch_size=args.eval_batch_size)
    dev_score_dict = load_json_score_data(json_score_file_name=dev_score_file_name)
    t_total_steps = len(train_data_loader) * args.num_train_epochs
    model = RangeSeqModel(args=args)
    model.to(device)

    model.zero_grad()
    model.train()
    optimizer = get_optimizer(model=model, args=args)
    scheduler = get_scheduler(optimizer=optimizer, args=args, total_steps=t_total_steps)
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)

    ###++++++++++++++++++++++++++++++++++++++++++
    total_batch_num = len(train_data_loader)
    print('Total number of batches = {}'.format(total_batch_num))
    eval_batch_interval_num = int(total_batch_num * args.eval_interval_ratio) + 1
    print('Evaluate the model by = {} batches'.format(eval_batch_interval_num))
    ###++++++++++++++++++++++++++++++++++++++++++

    start_epoch = 0
    best_em_ratio = 0.0
    best_f1 = 0.0
    dev_loss = 0.0
    dev_prediction_dict = None
    for epoch in range(start_epoch, start_epoch + int(args.num_train_epochs)):
        epoch_iterator = train_data_loader
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #+++++++
            for key, value in batch.items():
                if key not in ['id']:
                    batch[key] = value.to(device)
            #+++++++
            # batch_analysis(batch['x_feat'])
            start_scores, end_scores = model(batch['x_feat'])
            loss = seq_loss_computation(start=start_scores, end=end_scores, batch=batch)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step % 10 == 0:
                print('Epoch={}\tstep={}\tloss={:.5f}\teval_em={:.6f}\teval_f1={:.6f}\teval_loss={:.5f}\n'.format(epoch, step, loss.data.item(), best_em_ratio, best_f1, dev_loss))
            if (step + 1) % eval_batch_interval_num == 0:
                em_count, dev_f1, total_count, dev_loss_i, pred_dict = eval_model(model=model, data_loader=dev_data_loader,
                                                                          device=device, alpha=args.alpha, threshold_category=threshold_category, dev_score_dict=dev_score_dict)
                dev_loss = dev_loss_i
                em_ratio = em_count * 1.0/total_count
                if em_ratio > best_em_ratio:
                    best_em_ratio = em_ratio
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                               join(args.output_dir, args.exp_name, f'seq_threshold_pred_model.pkl'))
                    dev_prediction_dict = pred_dict
                if best_f1 < dev_f1:
                    best_f1 = dev_f1

    print('Best em ratio = {:.5f}'.format(best_em_ratio))
    print('Best f1 = {:.5f}'.format(best_f1))
    return best_em_ratio, best_f1, dev_prediction_dict

def eval_model(model, data_loader, dev_score_dict, threshold_category, alpha, device):
    model.eval()
    em_count = 0
    total_count = 0
    pred_score_dict = {}
    # for batch in tqdm(data_loader):
    dev_loss_list = []
    dev_f1_list = []
    for batch in data_loader:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in ['id']:
                batch[key] = value.to(device)
        with torch.no_grad():
            start_scores, end_scores, y1, y2 = model(batch['x_feat'], return_yp=True)
            loss = seq_loss_computation(start=start_scores, end=end_scores, batch=batch)
            dev_loss_list.append(loss.data.item())
            y_min_np = batch['y_min'].data.cpu().numpy()
            y_max_np = batch['y_max'].data.cpu().numpy()
            y_flag_np = batch['flag'].data.cpu().numpy()
            start_indexes = y1.data.cpu().numpy()
            end_indexes = y2.data.cpu().numpy()

            for i in range(y_min_np.shape[0]):
                key = batch['id'][i]
                total_count = total_count + 1
                start_i = int(start_indexes[i])
                end_i = int(end_indexes[i])
                score_i = (threshold_category[start_i][1] * (1 - alpha) + threshold_category[end_i][0] * alpha)
                y_min_i = np_sigmoid(y_min_np[i])
                y_max_i = np_sigmoid(y_max_np[i])
                y_flag_i = y_flag_np[i]
                score_i = y_min_i
                # print('pred', start_i, end_i)
                # print('gold', batch['y_1'][i], batch['y_2'][i])
                if key in dev_score_dict:
                    score_row = dev_score_dict[key]
                    f1_i = score_row_supp_f1_computation(row=score_row, threshold=score_i)
                    print(f1_i)
                    dev_f1_list.append(f1_i)
                else:
                    dev_f1_list.append(0.0)

                # print(score_i, y_min_i, y_max_i)
                if score_i >= y_min_i and score_i <= y_max_i and y_flag_i == 1:
                    em_count = em_count + 1
                pred_score_dict[key] = float(score_i)
    # print(em_count, total_count)
    avg_dev_loss = sum(dev_loss_list)/len(dev_loss_list)
    dev_f1 = sum(dev_f1_list)/len(dev_f1_list)
    return em_count, dev_f1, total_count, avg_dev_loss, pred_score_dict

if __name__ == '__main__':

    args = train_parser()
    best_em_ratio, best_f1, dev_prediction_dict = train(args)
    predict_threshold_file_name = join(args.output_dir, args.exp_name, args.pred_threshold_json_name)
    json.dump(dev_prediction_dict, open(predict_threshold_file_name, 'w'))
    print('Saving {} records into {}'.format(len(dev_prediction_dict), predict_threshold_file_name))