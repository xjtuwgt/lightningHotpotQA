from jdqamodel.hotpotqa_data_loader import HotpotDataset
from torch.utils.data import DataLoader
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def hotpot_train_dataloader(train_examples, args) -> DataLoader:
    train_data = HotpotDataset(examples=train_examples,
                              max_para_num=args.max_para_num,
                              max_sent_num=args.max_sent_num,
                              max_seq_num=args.max_seq_length,
                              sep_token_id=args.sep_token_id,
                              sent_drop_ratio=args.sent_drop_ratio)
    ####++++++++++++
    dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDataset.collate_fn)
    return dataloader

def hotpot_val_dataloader(dev_examples, sep_token_id, args) -> DataLoader:
    dev_data = HotpotDataset(examples=dev_examples,
                              max_para_num=args.max_para_num,
                              max_sent_num=args.max_sent_num,
                              max_seq_num=args.max_seq_length,
                              sep_token_id=sep_token_id,
                              sent_drop_ratio=-1.0)
    ####++++++++++++
    dataloader = DataLoader(
        dataset=dev_data,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(1, args.cpu_num // 2),
        collate_fn=HotpotDataset.collate_fn)
    return dataloader