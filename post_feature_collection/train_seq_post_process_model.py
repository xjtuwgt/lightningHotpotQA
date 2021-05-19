from post_feature_collection.post_process_data_helper import RangeSeqDataset
from post_feature_collection.post_process_argument_parser import train_parser
from torch.utils.data import DataLoader
from utils.jdutils import seed_everything
from os.path import join
import torch
import json
from leaderboardscripts.lb_postprocess_model import RangeSeqModel, RangeSeqScoreModel, seq_loss_computation
from tqdm import tqdm, trange
from adaptive_threshold.atutils import get_optimizer, get_scheduler
import random
import numpy as np
from utils.gpu_utils import single_free_cuda
from torch import Tensor
import torch
from post_feature_collection.post_process_feature_extractor import get_threshold_category, np_sigmoid, \
    load_json_score_data, score_row_supp_f1_computation

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
    print('device = {}'.format(device))

    ##+++++++++
    random_seed = args.rand_seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    ##+++++++++
    train_data = RangeSeqDataset(json_file_name=train_feat_file_name, span_window_size=args.span_window_size, trim_drop_ratio=args.trim_drop_ratio)
    dev_data = RangeSeqDataset(json_file_name=dev_feat_file_name, span_window_size=args.span_window_size, trim_drop_ratio=0.0)
    train_data_loader = DataLoader(dataset=train_data,
                                   shuffle=True,
                                   collate_fn=RangeSeqDataset.collate_fn,
                                   num_workers=args.cpu_number,
                                   batch_size=args.train_batch_size)
    dev_data_loader = DataLoader(dataset=dev_data,
                                 shuffle=False,
                                 collate_fn=RangeSeqDataset.collate_fn,
                                 batch_size=args.eval_batch_size)
    dev_score_dict = load_json_score_data(json_score_file_name=dev_score_file_name)
    t_total_steps = len(train_data_loader) * args.num_train_epochs
    model = RangeSeqModel(args=args)
    # model = RangeSeqScoreModel(args=args)
    #++++++++++++++++++++++++++++++++++++++++++++
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
    early_stop_step = 0

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
            loss = seq_loss_computation(start=start_scores, end=end_scores, batch=batch, weight=args.weighted_loss)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step % 10 == 0:
                print('Epoch={}\tstep={}\tloss={:.5f}\teval_em={:.6f}\teval_f1={:.6f}\teval_loss={:.5f}\n'.format(epoch, step, loss.data.item(), best_em_ratio, best_f1, dev_loss))
            if (step + 1) % eval_batch_interval_num == 0:
                em_count, dev_f1, total_count, dev_loss_i, pred_dict = eval_model(model=model, data_loader=dev_data_loader, weighted_loss=args.weighted_loss,
                                                                          device=device, alpha=args.alpha, threshold_category=threshold_category, dev_score_dict=dev_score_dict)
                dev_loss = dev_loss_i
                em_ratio = em_count * 1.0/total_count
                # if em_ratio > best_em_ratio:
                #     best_em_ratio = em_ratio
                #     torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                #                join(args.output_dir, args.exp_name, f'seq_threshold_pred_model.pkl'))
                #     dev_prediction_dict = pred_dict
                if best_f1 < dev_f1:
                    best_f1 = dev_f1
                    early_stop_step = 0
                    best_em_ratio = em_ratio
                    best_f1_em = 'f1_{:.4f}_em_{:.4f}'.format(best_f1, best_em_ratio)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                               join(args.output_dir, args.exp_name, f'seq_pred_model_{epoch + 1}.step_{step + 1}.{best_f1_em}.pkl'))
                    dev_prediction_dict = pred_dict
                else:
                    early_stop_step += 1
    print('Best em ratio = {:.5f}'.format(best_em_ratio))
    print('Best f1 = {:.5f}'.format(best_f1))
    return best_em_ratio, best_f1, dev_prediction_dict

def eval_model(model, data_loader, dev_score_dict, threshold_category, alpha, weighted_loss, device):
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
            loss = seq_loss_computation(start=start_scores, end=end_scores, batch=batch, weight=weighted_loss)
            dev_loss_list.append(loss.data.item())
            y_min_np = batch['y_min'].data.cpu().numpy()
            y_max_np = batch['y_max'].data.cpu().numpy()
            y_flag_np = batch['flag'].data.cpu().numpy()
            start_indexes = y1.data.cpu().numpy()
            end_indexes = y2.data.cpu().numpy()
            # gold_y_1 = batch['y_1'].data.cpu().numpy()
            # gold_y_2 = batch['y_2'].data.cpu().numpy()

            for i in range(y_min_np.shape[0]):
                key = batch['id'][i]
                total_count = total_count + 1
                start_i = int(start_indexes[i])
                end_i = int(end_indexes[i])
                if start_i > end_i:
                    print('here')
                # print('start pred: {} \t true: {}'.format(start_i, gold_y_1[i]))
                # print('end pred: {} \t true: {}'.format(end_i, gold_y_2[i]))
                pred_idx_i = (start_i + end_i) // 2 + 1 ## better for EM
                score_i = (threshold_category[start_i][1] * (1 - alpha) + threshold_category[end_i][0] * alpha) ## better for F1
                score_i = (threshold_category[pred_idx_i][1] + score_i)/2
                y_min_i = np_sigmoid(y_min_np[i])
                y_max_i = np_sigmoid(y_max_np[i])
                y_flag_i = y_flag_np[i]

                # print('pred', start_i, end_i)
                # print('gold', batch['y_1'][i], batch['y_2'][i])
                if key in dev_score_dict:
                    score_row = dev_score_dict[key]
                    f1_i = score_row_supp_f1_computation(row=score_row, threshold=score_i)
                    dev_f1_list.append(f1_i)
                else:
                    f1_i = 0.0
                    dev_f1_list.append(f1_i)

                # print(score_i, y_min_i, y_max_i)
                if score_i > y_min_i and score_i < y_max_i and y_flag_i == 1:
                    em_count = em_count + 1
                # else:
                #     print(f1_i)
                pred_score_dict[key] = float(score_i)
    # print(em_count, total_count)
    avg_dev_loss = sum(dev_loss_list)/len(dev_loss_list)
    dev_f1 = sum(dev_f1_list)/len(dev_f1_list)
    return em_count, dev_f1, total_count, avg_dev_loss, pred_score_dict

if __name__ == '__main__':

    learning_rate_array = [0.003, 0.005, 0.01]
    decoder_span_window_size_pair = [(170, 180)]
    encoder_drop_out = [0.25]
    trim_drop_ratio = [0.1]
    alpha_array = [0.05, 0.1]

    encoder_array = ['ff']
    expriment_num = 0
    best_res_metrics = []
    for lr in learning_rate_array:
        for win_pair in decoder_span_window_size_pair:
            for encode_dr in encoder_drop_out:
                for alpha in alpha_array:
                    for t_dr in trim_drop_ratio:
                        for encoder in encoder_array:
                            experiment_id = encoder + '_' + str(lr) + '_' + str(win_pair[0]) + '_' + \
                                            str(win_pair[1]) + '_' + str(encode_dr) + '_' + str(t_dr) + '_' + str(alpha)
                            print('training post process via {}'.format(experiment_id))
                            args = train_parser()
                            args.rand_seed = args.rand_seed + 100
                            seed_everything(seed=args.rand_seed)
                            args.encoder_type = encoder
                            args.decoder_window_size = win_pair[1]
                            args.span_window_size = win_pair[0]
                            args.trim_drop_ratio = t_dr
                            args.alpha = alpha
                            args.learning_rate = lr
                            best_em_ratio, best_f1, dev_prediction_dict = train(args)
                            best_res_metrics.append((expriment_num, experiment_id, best_em_ratio, best_f1))
                            predict_threshold_file_name = join(args.output_dir, args.exp_name, args.pred_threshold_json_name)
                            json.dump(dev_prediction_dict, open(predict_threshold_file_name, 'w'))
                            print('Saving {} records into {}'.format(len(dev_prediction_dict), predict_threshold_file_name))
                            expriment_num = expriment_num + 1
                            print('Experiment {} completed'.format(experiment_id))

    for res in best_res_metrics:
        print(res)