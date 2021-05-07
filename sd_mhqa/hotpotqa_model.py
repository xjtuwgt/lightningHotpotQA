from torch import nn
import torch
from sd_mhqa.transformer import TransformerLayer
from csr_mhqa.utils import load_encoder_model
from sd_mhqa.hotpotqa_data_loader import IGNORE_INDEX
import logging
from os.path import join
from hgntransformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.optimizerutils import RecAdam
from sd_mhqa.hotpotqaUtils import ParaSentPredictionLayer, PredictionLayer, para_sent_state_feature_extractor, GatedAttention

##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class ReaderModel(nn.Module):
    def __init__(self, config):
        super(ReaderModel, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.transformer_hidden_dim
        self.head_num = config.transformer_head_num
        self.linear_map = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=False)
        self.transformer_layer = TransformerLayer(d_model=self.hidden_dim, ffn_hidden=4 * self.hidden_dim,
                                                  n_head=self.head_num)
        self.second_transformer_layer = TransformerLayer(d_model=self.hidden_dim, ffn_hidden=4 * self.hidden_dim,
                                                         n_head=self.head_num)  # two layer transformer
        self.ctx_attention = GatedAttention(input_dim=self.hidden_dim,
                                            memory_dim=self.hidden_dim * 2,
                                            hid_dim=self.hidden_dim,
                                            dropout=self.config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)
        self.para_sent_predict_layer = ParaSentPredictionLayer(self.config, hidden_dim=2 * self.hidden_dim)
        self.predict_layer = PredictionLayer(self.config)

    def forward(self, batch, return_yp=False, return_cls=False):
        context_encoding = batch['context_encoding']
        ####++++++++++++++++++++++++++++++++++++++
        input_state = self.linear_map(context_encoding)
        batch_mask = batch['context_mask'].unsqueeze(1)
        input_state = self.transformer_layer.forward(x=input_state, src_mask=batch_mask)
        input_state = self.second_transformer_layer.forward(x=input_state, src_mask=batch_mask)  ##two layer transformer
        ####++++++++++++++++++++++++++++++++++++++
        para_sent_state_dict = para_sent_state_feature_extractor(batch=batch, input_state=input_state, co_atten=True)
        ####++++++++++++++++++++++++++++++++++++++
        # print(para_sent_state_dict['para_sent_state'].shape, para_sent_state_dict['para_sent_mask'].shape)
        input_state, _ = self.ctx_attention(input_state, para_sent_state_dict['para_sent_state'],
                                            para_sent_state_dict['para_sent_mask'].squeeze(-1))
        para_sent_state_dict = para_sent_state_feature_extractor(batch=batch, input_state=input_state, co_atten=False)
        ####++++++++++++++++++++++++++++++++++++++
        para_predictions, sent_predictions = self.para_sent_predict_layer.forward(state_dict=para_sent_state_dict)
        query_mapping = batch['query_mapping']
        if self.training:
            predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
                                                 packing_mask=query_mapping, return_yp=False)
        else:
            predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
                                                     packing_mask=query_mapping, return_yp=True)
        if not self.training:
            start, end, q_type, yp1, yp2 = predictions
            if return_yp:
                return start, end, q_type, para_predictions, sent_predictions, yp1, yp2
            else:
                return start, end, q_type, para_predictions, sent_predictions
        else:
            start, end, q_type = predictions
            loss_list = self.compute_loss(self.config, batch, start, end, para_predictions, sent_predictions, q_type)
            return loss_list

    def compute_loss(self, args, batch, start, end, para, sent, q_type):
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
        print(start.shape, end.shape, batch['y1'].shape, batch['y2'].shape)
        loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
        loss_type = args.type_lambda * criterion(q_type, batch['q_type'])

        sent_pred = sent.view(-1, 2)
        sent_gold = batch['is_support'].long().view(-1)
        loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())

        loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))

        loss = loss_span + loss_type + loss_sup + loss_para
        return loss, loss_span, loss_type, loss_sup, loss_para

class UnifiedSDModel(nn.Module):
    def __init__(self, config):
        super(UnifiedSDModel, self).__init__()
        self.config = config
        self.encoder, _ = load_encoder_model(self.config.encoder_name_or_path, self.config.model_type)
        self.model = ReaderModel(config=self.config)
        if self.config.fine_tuned_encoder is not None:
            encoder_path = join(self.config.fine_tuned_encoder_path, self.config.fine_tuned_encoder, 'encoder.pkl')
            logging.info("Loading encoder from: {}".format(encoder_path))
            self.encoder.load_state_dict(torch.load(encoder_path))
            logging.info("Loading encoder completed")

    def forward(self, batch, return_yp=False, return_cls=False):
        ###############################################################################################################
        inputs = {'input_ids': batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if self.config.model_type in ['bert',
                                                                                        'xlnet'] else None}  # XLM don't use segment_ids
        ####++++++++++++++++++++++++++++++++++++++
        outputs = self.encoder(**inputs)
        batch['context_encoding'] = outputs[0]
        ####++++++++++++++++++++++++++++++++++++++
        batch['context_mask'] = batch['context_mask'].float().to(self.config.device)
        return self.model.forward(batch=batch, return_yp=return_yp, return_cls=return_cls)

    def fixed_learning_rate_optimizers(self, total_steps):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate, eps=self.config.adam_epsilon)

        if self.config.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return optimizer, scheduler

    def rec_adam_learning_optimizer(self, total_steps):
        no_decay = ["bias", "LayerNorm.weight"]
        new_model = self.model
        args = self.config
        pretrained_model = self.encoder
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in new_model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": args.weight_decay,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           not any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": args.weight_decay,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    not any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type in n],
                "weight_decay": 0.0,
                "anneal_w": args.recadam_anneal_w,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type in p_n]
            },
            {
                "params": [p for n, p in new_model.named_parameters() if
                           any(nd in n for nd in no_decay) and args.model_type not in n],
                "weight_decay": 0.0,
                "anneal_w": 0.0,
                "pretrain_params": [p_p for p_n, p_p in pretrained_model.named_parameters() if
                                    any(nd in p_n for nd in no_decay) and args.model_type not in p_n]
            }
        ]
        optimizer = RecAdam(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon,
                            anneal_fun=args.recadam_anneal_fun, anneal_k=args.recadam_anneal_k,
                            anneal_t0=args.recadam_anneal_t0, pretrain_cof=args.recadam_pretrain_cof)
        if self.config.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.config.warmup_steps,
                                                        num_training_steps=total_steps)
        elif self.config.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.config.warmup_steps,
                                                                           num_training_steps=total_steps)
        else:
            raise '{} is not supported'.format(self.config.lr_scheduler)
        return optimizer, scheduler
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
##++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# class SDModel(nn.Module):
#     """
#     Packing Query Version
#     """
#     def __init__(self, config):
#         super(SDModel, self).__init__()
#         self.config = config
#         self.input_dim = config.input_dim
#         self.hidden_dim = config.transformer_hidden_dim
#         self.head_num = config.transformer_head_num
#         self.encoder, _ = load_encoder_model(self.config.encoder_name_or_path, self.config.model_type)
#         if self.config.fine_tuned_encoder is not None:
#             encoder_path = join(self.config.fine_tuned_encoder_path, self.config.fine_tuned_encoder, 'encoder.pkl')
#             logging.info("Loading encoder from: {}".format(encoder_path))
#             self.encoder.load_state_dict(torch.load(encoder_path))
#             logging.info("Loading encoder completed")
#         self.linear_map = nn.Linear(in_features=self.input_dim, out_features=self.hidden_dim, bias=False)
#         self.transformer_layer = TransformerLayer(d_model=self.hidden_dim, ffn_hidden=4 * self.hidden_dim,
#                                                   n_head=self.head_num)
#         self.para_sent_predict_layer = ParaSentPredictionLayer(self.config, hidden_dim=2 * self.hidden_dim)
#         self.predict_layer = PredictionLayer(self.config)
#
#     def forward(self, batch, return_yp=False, return_cls=False):
#         ####++++++++++++++++++++++++++++++++++++++
#         ###############################################################################################################
#         inputs = {'input_ids': batch['context_idxs'],
#                   'attention_mask': batch['context_mask'],
#                   'token_type_ids': batch['segment_idxs'] if self.config.model_type in ['bert', 'xlnet'] else None}  # XLM don't use segment_ids
#         ####++++++++++++++++++++++++++++++++++++++
#         outputs = self.encoder(**inputs)
#         batch['context_encoding'] = outputs[0]
#         ####++++++++++++++++++++++++++++++++++++++
#         batch['context_mask'] = batch['context_mask'].float().to(self.config.device)
#         context_encoding = batch['context_encoding']
#         input_state = self.linear_map(context_encoding)
#         batch_mask = batch['context_mask'].unsqueeze(1)
#         input_state = self.transformer_layer.forward(x=input_state, src_mask=batch_mask)
#         ####++++++++++++++++++++++++++++++++++++++
#         ####++++++++++++++++++++++++++++++++++++++
#         para_sent_state_dict = para_sent_state_feature_extractor(batch=batch, input_state=input_state)
#         para_predictions, sent_predictions = self.para_sent_predict_layer.forward(state_dict=para_sent_state_dict)
#         query_mapping = batch['query_mapping']
#         if self.training:
#             predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
#                                                  packing_mask=query_mapping, return_yp=False)
#         else:
#             predictions = self.predict_layer.forward(batch=batch, context_input=input_state,
#                                                      packing_mask=query_mapping, return_yp=True)
#         if not self.training:
#             start, end, q_type, yp1, yp2 = predictions
#             if return_yp:
#                 return start, end, q_type, para_predictions, sent_predictions, yp1, yp2
#             else:
#                 return start, end, q_type, para_predictions, sent_predictions
#         else:
#             start, end, q_type = predictions
#             loss_list = self.compute_loss(self.config, batch, start, end, para_predictions, sent_predictions, q_type)
#             return loss_list
#
#     def compute_loss(self, args, batch, start, end, para, sent, q_type):
#         criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)
#         loss_span = args.ans_lambda * (criterion(start, batch['y1']) + criterion(end, batch['y2']))
#         loss_type = args.type_lambda * criterion(q_type, batch['q_type'])
#
#         sent_pred = sent.view(-1, 2)
#         sent_gold = batch['is_support'].long().view(-1)
#         loss_sup = args.sent_lambda * criterion(sent_pred, sent_gold.long())
#
#         loss_para = args.para_lambda * criterion(para.view(-1, 2), batch['is_gold_para'].long().view(-1))
#
#         loss = loss_span + loss_type + loss_sup + loss_para
#
#         if loss_span > 1000:
#             logging.info('Hhhhhhhhhhhhhhhhh {}'.format((loss_span, loss_type, loss_sup, loss_para)))
#             start_list = batch['y1'].tolist()
#             mask = batch['context_mask']
#             for x_idx, x in enumerate(start_list):
#                 print(x, start[x_idx][x], mask[x_idx][x])
#             # logging.info(start)
#             # logging.info(batch['y1'])
#             # logging.info(criterion(start, batch['y1']))
#             logging.info('*' * 10)
#             # logging.info(end)
#             end_list = batch['y2'].tolist()
#             for x_idx, x in enumerate(end_list):
#                 print(x, end[x_idx][x], mask[x_idx][x])
#             # logging.info(batch['y2'])
#             # logging.info(criterion(end, batch['y2']))
#
#         return loss, loss_span, loss_type, loss_sup, loss_para