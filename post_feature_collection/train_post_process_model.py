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
    for x in train_data:
        print(x)
    train_data_loader = DataLoader(dataset=train_data,
                                   shuffle=True,
                                   collate_fn=RangeDataset.collate_fn,
                                   num_workers=1,
                                   batch_size=args.train_batch_size)

    # dev_data = RangeDataset(json_file_name=dev_feat_file_name)
    # dev_data_loader = DataLoader(dataset=dev_data,
    #                              shuffle=False,
    #                              collate_fn=RangeDataset.collate_fn,
    #                              num_workers=args.cpu_number // 2,
    #                              batch_size=args.eval_batch_size)

    model = RangeModel(args=args)
    model.to(device)

    model.zero_grad()
    model.train()
    optimizer = get_optimizer(model=model, args=args)
    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)

    for batch in train_data_loader:
        print(batch['x_feat'].shape)

if __name__ == '__main__':

    args = train_parser()
    train(args)