from __future__ import absolute_import, division, print_function
import logging
import sys
from utils.gpu_utils import gpu_setting
from plmodels.jd_argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from plmodels.lightningHGN import lightningHGN
import pytorch_lightning as pl
import torch
from os.path import join
from jdevaluation.devdataHelper import DataHelper as DevDataHelper
from envs import OUTPUT_FOLDER

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)
########################################################################################################################
def parse_args():
    parser = default_train_parser()
    logger.info("IN CMD MODE")
    args_config_provided = parser.parse_args(sys.argv[1:])
    if args_config_provided.config_file is not None:
        argv = json_to_argv(args_config_provided.config_file) + sys.argv[1:]
    else:
        argv = sys.argv[1:]
    args = parser.parse_args(argv)
    args = complete_default_train_parser(args)

    logger.info('-' * 100)
    logger.info('Input Argument Information')
    logger.info('-' * 100)
    args_dict = vars(args)
    for a in args_dict:
        logger.info('%-28s  %s' % (a, args_dict[a]))
    return args
########################################################################################################################
def device_setting(args):
    if torch.cuda.is_available():
        free_gpu_ids, used_memory = gpu_setting(num_gpu=args.gpus)
        print('{} gpus with used memory = {}, gpu ids = {}'.format(len(free_gpu_ids), used_memory, free_gpu_ids))
        if args.gpus > 0:
            gpu_ids = free_gpu_ids
            device = torch.device("cuda:%d" % gpu_ids[0])
            print('Single GPU setting')
        else:
            device = torch.device("cpu")
            print('Single cpu setting')
    else:
        device = torch.device("cpu")
        print('Single cpu setting')
    return device
########################################################################################################################
def dev_data_loader(args):
    dev_helper = DevDataHelper(gz=True, config=args)
    dev_data_loader = dev_helper.hotpot_val_dataloader
    return dev_data_loader
########################################################################################################################
def main(args):
    device = device_setting(args=args)
    model_ckpt = join(OUTPUT_FOLDER, args.exp_name, 'test.ckpt')
    train_model = lightningHGN(args=args)

    trainer = pl.Trainer(checkpoint_callback=False)
    trainer.fit(train_model)
    trainer.save_checkpoint(model_ckpt)


    # # hyper_parameters = join(OUTPUT_FOLDER, args.exp_name, 'default/version_4/hparams.yaml')
    # print('model checkpoint {}'.format(model_ckpt))
    lighthgn_model = lightningHGN.load_from_checkpoint(checkpoint_path=model_ckpt)
    lighthgn_model = lighthgn_model.to(device)
    print('Model Parameter Configuration:')
    for name, param in lighthgn_model.named_parameters():
        print('Parameter {}: {}, require_grad = {}'.format(name, str(param.size()), str(param.requires_grad)))
    print('*' * 75)
    # print("Model hype-parameter information...")
    # for key, value in vars(args).items():
    #     print('Hype-parameter\t{} = {}'.format(key, value))
    # print('*' * 75)
    # dev_data = dev_data_loader(args=args)

if __name__ == '__main__':
    args = parse_args()
    main(args)
