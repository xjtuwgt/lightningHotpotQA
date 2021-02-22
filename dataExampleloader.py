import sys

from plmodels.jd_argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from plmodels.pldata_processing import Example, InputFeatures, DataHelper, HotpotDataset
import logging
from model_envs import MODEL_CLASSES
from torch.utils.data import DataLoader

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
#########################################################################
# Initialize arguments
##########################################################################
parser = default_train_parser()

logger.info("IN CMD MODE")
args_config_provided = parser.parse_args(sys.argv[1:])
if args_config_provided.config_file is not None:
    argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
else:
    argv = sys.argv[1:]
args = parser.parse_args(argv)
#########################################################################
for key, value in vars(args).items():
    print('Hype-parameter\t{} = {}'.format(key, value))
#########################################################################
args = complete_default_train_parser(args)

logger.info('-' * 100)
logger.info('Input Argument Information')
logger.info('-' * 100)
args_dict = vars(args)
for a in args_dict:
    logger.info('%-28s  %s' % (a, args_dict[a]))

#########################################################################
# Read Data
##########################################################################
helper = DataHelper(gz=True, config=args)

# Set datasets
# train_dataloader = helper.train_loader
dev_example_dict = helper.dev_example_dict
dev_feature_dict = helper.dev_feature_dict
dev_dataloader = helper.dev_loader

_, _, tokenizer_class = MODEL_CLASSES[args.model_type]
tokenizer = tokenizer_class.from_pretrained(args.encoder_name_or_path,
                                            do_lower_case=args.do_lower_case)

def get_data_loader(dataset):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=16,
        shuffle=False,
        num_workers=max(1, 6),
        collate_fn=HotpotDataset.collate_fn
    )
    return dataloader

dev_loader = get_data_loader(dataset=dev_dataloader)
for batch_idx, batch in enumerate(dev_loader):
    ids = batch['ids']
    for key, value in batch.items():
        print(type(value))

    for idx, id in enumerate(ids):
        print(dev_example_dict[id].question_tokens)
        print('query', tokenizer.decode(batch['context_idxs'][idx] * batch['query_mapping'][idx], skip_special_tokens=True))
        print('context', tokenizer.decode(batch['context_idxs'][idx] * batch['context_mask'][idx], skip_special_tokens=True))
        print('-' * 75)
        para_num = batch['para_mapping'].shape[-1]
        sent_num = batch['sent_mapping'].shape[-1]
        ent_num = batch['ent_mapping'].shape[-1]
        for j in range(para_num):
            para_mask_j = batch['para_mapping'][idx][:,j]
            print('doc {} = {}'.format(j, tokenizer.decode(batch['context_idxs'][idx] * para_mask_j, skip_special_tokens=True)))
        print('+' * 75)
        for j in range(sent_num):
            sent_mask_j = batch['sent_mapping'][idx][:,j]
            print('sent {} = {}'.format(j, tokenizer.decode(batch['context_idxs'][idx] * sent_mask_j, skip_special_tokens=True)))
        print('+' * 75)
        for j in range(ent_num):
            ent_mask_j = batch['ent_mapping'][idx][:,j]
            print('ent {} = {}'.format(j, tokenizer.decode(batch['context_idxs'][idx] * ent_mask_j, skip_special_tokens=True)))
        # print(batch['para_mapping'][idx].sum(dim=0), batch['para_mapping'][idx].shape)
        # print(batch['sent_mapping'][idx].sum(dim=0), batch['sent_mapping'][idx].shape)
        # print(batch['ent_mapping'][idx].sum(dim=0), batch['ent_mapping'].shape)
        # print(batch['query_mapping'][idx])
    print('*' * 75)
    print(batch['context_lens'])
    break
    # print(row['ids'])
