from __future__ import absolute_import, division, print_function
import logging
import sys
from utils.gpu_utils import gpu_setting
from plmodels.jd_argument_parser import default_train_parser, complete_default_train_parser, json_to_argv
from plmodels.lightningHGN import lightningHGN
import torch
from jdevaluation.devdataHelper import DataHelper as DevDataHelper

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

########################################################################################################################
def main(args):
    device = device_setting(args=args)
    dev_helper = DevDataHelper(gz=True, config=args)
    dev_data_loader = DevDataHelper.hotpot_val_dataloader
    lighthgn = lightningHGN.load_from_checkpoint()



if __name__ == '__main__':
    args = parse_args()
    main(args)
