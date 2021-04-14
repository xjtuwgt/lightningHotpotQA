from adaptive_threshold.range_argument_parser import train_parser
from torch.utils.data import DataLoader
from adaptive_threshold.RangeDataLoader import RangeDataset
from os.path import join

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
    for idx, x in enumerate(train_data_loader):
        for key, value in x.items():
            print(key, value.shape)

    dev_npz_file_name = join(args.pred_dir, args.model_name_or_path, args.dev_feat_name)
    dev_npz_data = RangeDataset(npz_file_name=dev_npz_file_name)
    dev_data_loader = DataLoader(dataset=dev_npz_data,
                                   shuffle=False,
                                   collate_fn=RangeDataset.collate_fn,
                                   num_workers=args.cpu_number // 2,
                                   batch_size=args.train_batch_size)
    return

if __name__ == '__main__':
    args = train_parser()
    for key, value in vars(args).items():
        print(key, value)

    run(args=args)