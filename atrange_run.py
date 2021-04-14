from adaptive_threshold.range_argument_parser import train_parser
from torch.utils.data import DataLoader
from adaptive_threshold.RangeDataLoader import RangeDataset
from adaptive_threshold.atutils import get_optimizer
from os.path import join
from adaptive_threshold.RangeModel import RangeModel, loss_computation
from tqdm import tqdm

def run(args):
    if args.train_filter:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, 'filter_' + args.train_feat_name)
    else:
        train_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.train_feat_name)
    train_npz_data = RangeDataset(npz_file_name=train_npz_file_name)
    train_data_loader = DataLoader(dataset=train_npz_data,
                                   shuffle=True,
                                   collate_fn=RangeDataset.collate_fn,
                                   num_workers=args.cpu_number//2,
                                   batch_size=args.train_batch_size)

    # dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    # dev_npz_data = RangeDataset(npz_file_name=dev_npz_file_name)
    # dev_data_loader = DataLoader(dataset=dev_npz_data,
    #                                shuffle=False,
    #                                collate_fn=RangeDataset.collate_fn,
    #                                num_workers=args.cpu_number // 2,
    #                                batch_size=args.train_batch_size)

    model = RangeModel(args=args)
    model.zero_grad()
    model.train()

    optimizer = get_optimizer(model=model, args=args)

    for name, param in model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))

    for batch_idx, batch in tqdm(enumerate(train_data_loader)):
        scores = model(batch['x_feat']).squeeze(-1)
        loss = loss_computation(scores=scores, y_min=batch['y_min'], y_max=batch['y_max'])
        print(batch_idx, scores.shape, loss)
        loss.backward()
        optimizer.step()
        model.zero_grad()
    return

if __name__ == '__main__':
    args = train_parser()
    for key, value in vars(args).items():
        print(key, value)
    print('*' * 50)
    run(args=args)