from post_feature_collection.post_process_data_helper import RangeDataset
from post_feature_collection.post_process_argument_parser import train_parser
from torch.utils.data import DataLoader
from os.path import join
from utils.jdutils import seed_everything
import torch
import json
from leaderboardscripts.lb_postprocess_model import RangeModel, loss_computation
from leaderboardscripts.lb_postprocess_utils import get_threshold_category
from post_feature_collection.post_process_feature_extractor import np_sigmoid, load_json_score_data, \
    score_row_supp_f1_computation, row_f1_computation
from tqdm import tqdm, trange
from adaptive_threshold.atutils import get_optimizer, get_scheduler
import random
import numpy as np
from utils.gpu_utils import single_free_cuda
from torch import Tensor
import torch

def batch_analysis(x_feat: Tensor):
    p2dist = torch.cdist(x1=x_feat, x2=x_feat, p=2)
    print(p2dist)

def train(args):
    train_feat_file_name = join(args.output_dir, args.exp_name, args.train_feat_json_name)
    dev_feat_file_name = join(args.output_dir, args.exp_name, args.dev_feat_json_name)
    dev_score_file_name = join(args.output_dir, args.exp_name, args.dev_score_name)

    raw_dev_data_file_name = join(args.input_dir, args.raw_dev_data)

    raw_data = load_json_score_data(json_score_file_name=raw_dev_data_file_name)
    raw_data_dict = dict([(row['_id'], row)for row in raw_data])

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
    train_data = RangeDataset(json_file_name=train_feat_file_name)
    train_data_loader = DataLoader(dataset=train_data,
                                   shuffle=True,
                                   collate_fn=RangeDataset.collate_fn,
                                   batch_size=args.train_batch_size)

    dev_data = RangeDataset(json_file_name=dev_feat_file_name)
    dev_data_loader = DataLoader(dataset=dev_data,
                                 shuffle=False,
                                 collate_fn=RangeDataset.collate_fn,
                                 batch_size=args.eval_batch_size)
    dev_score_dict = load_json_score_data(json_score_file_name=dev_score_file_name)
    t_total_steps = len(train_data_loader) * args.num_train_epochs
    model = RangeModel(args=args)
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
    train_iterator = trange(start_epoch, start_epoch + int(args.num_train_epochs), desc="Epoch")
    best_em_ratio = 0.0
    best_f1 = 0.0
    dev_loss = 0.0
    dev_prediction_dict = None
    # for epoch in train_iterator:
    for epoch in range(start_epoch, start_epoch + int(args.num_train_epochs)):
        # epoch_iterator = tqdm(train_data_loader, desc="Iteration")
        epoch_iterator = train_data_loader
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #+++++++
            for key, value in batch.items():
                if key not in ['id']:
                    batch[key] = value.to(device)
            #+++++++
            # batch_analysis(batch['x_feat'])
            scores = model(batch['x_feat'])
            if args.weighted_loss:
                loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'], weight=batch['weight'])
            else:
                loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            if step % 10 == 0:
                print('Epoch={}\tstep={}\tloss={:.5f}\teval_em={:.6f}\teval_f1={:.6f}\teval_loss={:.5f}\n'.format(epoch, step, loss.data.item(), best_em_ratio, best_f1, dev_loss))
            if (step + 1) % eval_batch_interval_num == 0:
                em_ratio, dev_f1, total_count, dev_loss_i, pred_dict = eval_model(model=model, data_loader=dev_data_loader, device=device,
                                                                                  dev_score_dict=dev_score_dict, weigted_loss=args.weighted_loss,
                                                                                  raw_dev_dict=raw_data_dict)
                dev_loss = dev_loss_i
                # em_ratio = em_count * 1.0/total_count
                # if em_ratio > best_em_ratio:
                #     best_em_ratio = em_ratio
                #     torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                #                join(args.output_dir, args.exp_name, f'threshold_pred_model.pkl'))
                #     dev_prediction_dict = pred_dict
                if best_f1 < dev_f1:
                    best_f1 = dev_f1
                    best_em_ratio = em_ratio
                    best_f1_em = 'f1_{:.4f}_em_{:.4f}'.format(best_f1, best_em_ratio)
                    torch.save({k: v.cpu() for k, v in model.state_dict().items()},
                               join(args.output_dir, args.exp_name, f'seq_pred_model_{epoch + 1}.step_{step + 1}.{best_f1_em}.pkl'))
                    dev_prediction_dict = pred_dict

    print('Best em ratio = {:.5f}'.format(best_em_ratio))
    print('Best f1 = {:.5f}'.format(best_f1))
    return best_em_ratio, best_f1, dev_prediction_dict

def eval_model(model, data_loader, dev_score_dict, device, weigted_loss, raw_dev_dict):
    model.eval()
    em_count = 0
    total_count = 0
    pred_score_dict = {}
    # for batch in tqdm(data_loader):
    dev_loss_list = []
    dev_f1_list = []
    dev_em_list = []
    for batch in data_loader:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            if key not in ['id']:
                batch[key] = value.to(device)
        with torch.no_grad():
            scores = model(batch['x_feat'])
            if weigted_loss:
                loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'], weight=batch['weight'])
            else:
                loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'])
            dev_loss_list.append(loss.data.item())
            scores = scores.squeeze(-1)
            scores = torch.sigmoid(scores)
            score_np = scores.data.cpu().numpy()
            # y_min_np = batch['y_min'].data.cpu().numpy()
            # y_max_np = batch['y_max'].data.cpu().numpy()
            # y_flag_np = batch['flag'].data.cpu().numpy()

            for i in range(score_np.shape[0]):
                key = batch['id'][i]
                total_count = total_count + 1
                score_i = score_np[i]
                if key in dev_score_dict:
                    score_row = dev_score_dict[key]
                    raw_row = raw_dev_dict[key]
                    em_i, f1_i = row_f1_computation(row=score_row, raw_row=raw_row, threshold=score_i)
                else:
                    f1_i = 0.0
                    em_i = 0.0
                dev_em_list.append(em_i)
                dev_f1_list.append(f1_i)
                # if key in dev_score_dict:
                #     score_row = dev_score_dict[key]
                #     f1_i = score_row_supp_f1_computation(row=score_row, threshold=np_sigmoid(score_i))
                #     dev_f1_list.append(f1_i)
                # else:
                #     dev_f1_list.append(0.0)
                # y_min_i = y_min_np[i]
                # y_max_i = y_max_np[i]
                # y_flag_i = y_flag_np[i]
                # # print(score_i, y_min_i, y_max_i)
                # if score_i >= y_min_i and score_i <= y_max_i and y_flag_i == 1:
                #     em_count = em_count + 1
                pred_score_dict[key] = float(score_i)
    # print(em_count, total_count)
    avg_dev_loss = sum(dev_loss_list)/len(dev_loss_list)
    dev_f1 = sum(dev_f1_list)/len(dev_f1_list)
    em_ratio = sum(dev_em_list)/len(dev_em_list)
    return em_ratio, dev_f1, total_count, avg_dev_loss, pred_score_dict

if __name__ == '__main__':

    # args = train_parser()
    # best_em_ratio, best_f1, dev_prediction_dict = train(args)
    # predict_threshold_file_name = join(args.output_dir, args.exp_name, args.pred_threshold_json_name)
    # json.dump(dev_prediction_dict, open(predict_threshold_file_name, 'w'))
    # print('Saving {} records into {}'.format(len(dev_prediction_dict), predict_threshold_file_name))

    learning_rate_array = [0.001, 0.003]
    encoder_drop_out = [0.25, 0.3]
    encoder_array = ['ff']
    expriment_num = 0
    best_res_metrics = []
    for lr in learning_rate_array:
        for encode_dr in encoder_drop_out:
            for encoder in encoder_array:
                experiment_id = encoder + '_' + str(lr) + '_' + str(encode_dr)
                print('training post process via {}'.format(experiment_id))
                args = train_parser()
                args.rand_seed = args.rand_seed + 1
                seed_everything(seed=args.rand_seed)
                args.encoder_type = encoder
                args.learning_rate = lr
                best_em_ratio, best_f1, dev_prediction_dict = train(args)
                best_res_metrics.append((expriment_num, experiment_id, best_em_ratio, best_f1))
                predict_threshold_file_name = join(args.output_dir, args.exp_name,
                                                   args.pred_threshold_json_name)
                json.dump(dev_prediction_dict, open(predict_threshold_file_name, 'w'))
                print('Saving {} records into {}'.format(len(dev_prediction_dict), predict_threshold_file_name))
                expriment_num = expriment_num + 1
                print('Experiment {} completed'.format(experiment_id))

    for res in best_res_metrics:
        print(res)