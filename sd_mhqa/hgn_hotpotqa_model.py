from torch import nn
import logging
import torch
from csr_mhqa.utils import load_encoder_model
from models.HGN import HierarchicalGraphNetwork
from os.path import join
from csr_mhqa.utils import compute_loss
from hgntransformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from utils.optimizerutils import RecAdam

class UnifiedHGNModel(nn.Module):
    def __init__(self, config):
        super(UnifiedHGNModel, self).__init__()
        self.config = config
        self.encoder, _ = load_encoder_model(self.config.encoder_name_or_path, self.config.model_type)
        self.model = HierarchicalGraphNetwork(config=self.config)
        if self.config.fine_tuned_encoder is not None:
            encoder_path = join(self.config.fine_tuned_encoder_path, self.config.fine_tuned_encoder, 'encoder.pkl')
            logging.info("Loading encoder from: {}".format(encoder_path))
            self.encoder.load_state_dict(torch.load(encoder_path))
            logging.info("Loading encoder completed")

    def forward(self, batch, return_yp=False):
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
        if self.training:
            start, end, q_type, paras, sents, ents = self.model.forward(batch, return_yp=False)
            loss_list = compute_loss(self.config, batch, start, end, paras, sents, ents, q_type)
            return loss_list
        else:
            start, end, q_type, paras, sents, ents, y1, y2 = self.model.forward(batch, return_yp=True)
            return start, end, q_type, paras, sents, ents, y1, y2

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