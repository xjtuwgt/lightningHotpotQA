from post_feature_collection.post_process_data_helper import RangeDataset
from post_feature_collection.post_process_argument_parser import train_parser
from torch.utils.data import DataLoader
from leaderboardscripts.lb_postprocess_utils import load_json_score_data
from os.path import join
import torch
from adaptive_threshold.RangeModel import RangeModel, loss_computation
from tqdm import tqdm, trange
from adaptive_threshold.atutils import get_optimizer
import random
import numpy as np
from utils.gpu_utils import single_free_cuda

def train(args):
    train_feat_file_name = join(args.output_dir, args.exp_name, args.train_feat_json_name)
    dev_feat_file_name = join(args.output_dir, args.exp_name, args.dev_feat_json_name)

    if torch.cuda.is_available():
        device_ids, _ = single_free_cuda()
        device = torch.device('cuda:{}'.format(device_ids[0]))
    else:
        device = torch.device('cpu')

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

    model = RangeModel(args=args)
    model.to(device)

    model.zero_grad()
    model.train()
    optimizer = get_optimizer(model=model, args=args)
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
    # for epoch in train_iterator:
    for epoch in range(start_epoch, start_epoch + int(args.num_train_epochs)):
        # epoch_iterator = tqdm(train_data_loader, desc="Iteration")
        epoch_iterator = train_data_loader
        for step, batch in enumerate(epoch_iterator):
            model.train()
            #+++++++
            for key, value in batch.items():
                batch[key] = value.to(device)
            #+++++++
            scores = model(batch['x_feat']).squeeze(-1)
            loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            model.zero_grad()

            if step % 10 == 0:
                print('{}\t{}\t{:.5f}\n'.format(epoch, step, loss.data.item()))
            if (step + 1) % eval_batch_interval_num == 0:
                em_count, total_count = eval_model(model=model, data_loader=dev_data_loader, device=device)
                em_ratio = em_count * 1.0/total_count
                print('*' * 35)
                print('{}\t{}\t{:.5f}\n'.format(epoch, step, em_ratio))
                print('*' * 35)
                if em_ratio > best_em_ratio:
                    best_em_ratio = em_ratio
    print('Best em ratio = {:.5f}'.format(best_em_ratio))
    return best_em_ratio

def eval_model(model, data_loader, device):
    model.eval()
    em_count = 0
    total_count = 0
    # for batch in tqdm(data_loader):
    for batch in data_loader:
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        for key, value in batch.items():
            batch[key] = value.to(device)
        with torch.no_grad():
            scores = model(batch['x_feat']).squeeze(-1)
            scores = torch.sigmoid(scores)
            score_np = scores.data.cpu().numpy()
            y_min_np = batch['y_min'].data.cpu().numpy()
            y_max_np = batch['y_max'].data.cpu().numpy()
            y_flag_np = batch['flag'].data.cpu().numpy()

            for i in range(score_np.shape[0]):
                total_count = total_count + 1
                score_i = score_np[i]
                y_min_i = y_min_np[i]
                y_max_i = y_max_np[i]
                y_flag_i = y_flag_np[i]
                if score_i >= y_min_i and score_i <= y_max_i and y_flag_i == 1:
                    em_count = em_count + 1
    # print(em_count, total_count)
    return em_count, total_count

if __name__ == '__main__':

    args = train_parser()
    train(args)