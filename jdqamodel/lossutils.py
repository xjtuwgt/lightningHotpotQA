from torch import nn
from jdqamodel.hotpotqa_data_loader import IGNORE_INDEX
import logging
logger = logging.getLogger(__name__)

def compute_loss(args, batch, start, end, para, sent, q_type):
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')
    loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
    loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

    sent_pred = sent.view(-1, 2)
    sent_gold = batch['is_support'].long().view(-1)
    loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())

    loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))

    loss = loss_span + loss_type + loss_sup + loss_para

    if loss_span > 1000:
        logging.info('Hhhhhhhhhhhhhhhhh {}'.format((loss_span, loss_type, loss_sup, loss_para)))
        start_list = batch['y1'].tolist()
        mask = batch['context_mask']
        for x_idx, x in enumerate(start_list):
            print(x, start[x_idx][x], mask[x_idx][x])
        # logging.info(start)
        # logging.info(batch['y1'])
        # logging.info(criterion(start, batch['y1']))
        logging.info('*' * 10)
        # logging.info(end)
        end_list = batch['y2'].tolist()
        for x_idx, x in enumerate(end_list):
            print(x, end[x_idx][x], mask[x_idx][x])
        # logging.info(batch['y2'])
        # logging.info(criterion(end, batch['y2']))

    return loss, loss_span, loss_type, loss_sup, loss_para