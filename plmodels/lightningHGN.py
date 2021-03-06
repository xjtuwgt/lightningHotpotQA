import torch
import os
from os.path import join
import numpy as np
import json
from torch.utils.data import DataLoader
from hgntransformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)

from csr_mhqa.utils import load_encoder_model, compute_loss, convert_to_tokens
from models.HGN import HierarchicalGraphNetwork
import pytorch_lightning as pl
import torch.nn.functional as F
from eval.hotpot_evaluate_v1 import eval as hotpot_eval
import shutil
from argparse import Namespace
from utils.jdutils import log_metrics
from plmodels.pldata_processing import HotpotDataset, DataHelper
import logging
from utils.optimizerutils import RecAdam


MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    # 'unifiedqa': (T5Config, T5ForConditionalGeneration, AutoTokenizer)
}

class lightningHGN(pl.LightningModule):
    def __init__(self, hparams: Namespace):
        super().__init__()
        self.hparams = hparams
        self.save_hyperparameters()
        cached_config_file = join(self.hparams.exp_name, 'cached_config.bin')
        if os.path.exists(cached_config_file):
            cached_config = torch.load(cached_config_file)
            encoder_path = join(self.hparams.exp_name, cached_config['encoder'])
            model_path = join(self.hparams.exp_name, cached_config['model'])
        else:
            model_path = None
            if self.hparams.fine_tuned_encoder is not None:
                encoder_path = join(self.hparams.fine_tuned_encoder_path, self.hparams.fine_tuned_encoder, 'encoder.pkl')
            else:
                encoder_path = None

        _, _, tokenizer_class = MODEL_CLASSES[self.hparams.model_type]
        self.tokenizer = tokenizer_class.from_pretrained(self.hparams.encoder_name_or_path,
                                                    do_lower_case=self.hparams.do_lower_case)
        # Set Encoder and Model
        self.encoder, _ = load_encoder_model(self.hparams.encoder_name_or_path, self.hparams.model_type)
        self.model = HierarchicalGraphNetwork(config=self.hparams)
        if encoder_path is not None:
            self.encoder.load_state_dict(torch.load(encoder_path))
            logging.info('Initialize parameter with {}'.format(encoder_path))
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
        logging.info('Loading encoder and model completed')
        ##########

    def prepare_data(self):
        helper = DataHelper(gz=True, config=self.hparams)
        self.train_data = helper.train_loader
        self.dev_example_dict = helper.dev_example_dict
        self.dev_feature_dict = helper.dev_feature_dict
        self.dev_data = helper.dev_loader

    def setup(self, stage: str = 'fit'):
        if stage == 'fit':
            # Get dataloader by calling it - train_dataloader() is called after setup() by default
            train_loader = self.train_dataloader()
            # Calculate total steps
            if self.hparams.max_steps > 0:
                self.total_steps = self.hparams.max_steps
                self.hparams.num_train_epochs = self.hparams.max_steps // (
                            len(train_loader) // self.hparams.gradient_accumulation_steps) + 1
            else:
                self.total_steps = len(train_loader) // self.hparams.gradient_accumulation_steps * self.hparams.num_train_epochs
            print('total steps = {}'.format(self.total_steps))

    def train_dataloader(self):
        dataloader = DataLoader(dataset=self.train_data,
                                batch_size=self.hparams.per_gpu_train_batch_size,
                                shuffle=True,
                                drop_last=True,
                                pin_memory=True,
                                num_workers=max(1, self.hparams.cpu_num // 2),
                                collate_fn=HotpotDataset.collate_fn)
        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(dataset=self.dev_data,
                                batch_size=self.hparams.eval_batch_size,
                                shuffle=False,
                                drop_last=False,
                                pin_memory=True,
                                num_workers=max(1, self.hparams.cpu_num // 2),
                                collate_fn=HotpotDataset.collate_fn)
        return dataloader

    def forward(self, batch):
        inputs = {'input_ids':      batch['context_idxs'],
                  'attention_mask': batch['context_mask'],
                  'token_type_ids': batch['segment_idxs'] if self.hparams.model_type in ['bert', 'xlnet', 'electra'] else None}  # XLM don't use segment_ids
        ####++++++++++++++++++++++++++++++++++++++
        if self.hparams.model_type == 'electra':
            batch['context_encoding'] = self.encoder(**inputs).last_hidden_state
        else:
            batch['context_encoding'] = self.encoder(**inputs)[0]
        ####++++++++++++++++++++++++++++++++++++++
        batch['context_mask'] = batch['context_mask'].float().to(batch['context_encoding'].device)
        start, end, q_type, paras, sents, ents, yp1, yp2 = self.model(batch, return_yp=True)
        return start, end, q_type, paras, sents, ents, yp1, yp2

    def training_step(self, batch, batch_idx):
        start, end, q_type, paras, sents, ents, _, _ = self.forward(batch=batch)
        loss_list = compute_loss(self.hparams, batch, start, end, paras, sents, ents, q_type)
        ##################################################################################
        loss, loss_span, loss_type, loss_sup, loss_ent, loss_para = loss_list
        dict_for_progress_bar = {'span_loss': loss_span, 'type_loss': loss_type,
                                 'sent_loss': loss_sup, 'ent_loss': loss_ent,
                                 'para_loss': loss_para}
        dict_for_log = dict_for_progress_bar.copy()
        dict_for_log['step'] = batch_idx + 1
        ##################################################################################
        output = {'loss': loss, 'log': dict_for_log, 'progress_bar': dict_for_progress_bar}
        return output

    def validation_step(self, batch, batch_idx):
        start, end, q_type, paras, sents, ents, yp1, yp2 = self.forward(batch=batch)
        loss_list = compute_loss(self.hparams, batch, start, end, paras, sents, ents, q_type)
        loss, loss_span, loss_type, loss_sup, loss_ent, loss_para = loss_list
        dict_for_log = {'span_loss': loss_span, 'type_loss': loss_type,
                                 'sent_loss': loss_sup, 'ent_loss': loss_ent,
                                 'para_loss': loss_para,
                        'step': batch_idx + 1}
        #######################################################################
        type_prob = F.softmax(q_type, dim=1).data.cpu().numpy()
        answer_dict_, answer_type_dict_, answer_type_prob_dict_ = convert_to_tokens(self.dev_example_dict,
                                                                                    self.dev_feature_dict,
                                                                                    batch['ids'],
                                                                                    yp1.data.cpu().numpy().tolist(),
                                                                                    yp2.data.cpu().numpy().tolist(),
                                                                                    type_prob)
        predict_support_np = torch.sigmoid(sents[:, :, 1]).data.cpu().numpy()
        valid_dict = {'answer': answer_dict_, 'ans_type': answer_type_dict_, 'ids': batch['ids'],
                      'ans_type_pro': answer_type_prob_dict_, 'supp_np': predict_support_np}
        #######################################################################
        output = {'valid_loss': loss, 'log': dict_for_log, 'valid_dict_output': valid_dict}
        # output = {'valid_dict_output': valid_dict}
        return output

    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['valid_loss'] for x in validation_step_outputs]).mean()
        # print(avg_loss, type(avg_loss), avg_loss.device)
        # self.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        answer_dict = {}
        answer_type_dict = {}
        answer_type_prob_dict = {}

        thresholds = np.arange(0.1, 1.0, 0.02)
        N_thresh = len(thresholds)
        total_sp_dict = [{} for _ in range(N_thresh)]
        total_record_num = 0
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        valid_dict_outputs = [x['valid_dict_output'] for x in validation_step_outputs]
        for batch_idx, valid_dict in enumerate(valid_dict_outputs):
            answer_dict_, answer_type_dict_, answer_type_prob_dict_ = valid_dict['answer'], valid_dict['ans_type'], valid_dict['ans_type_pro']
            answer_type_dict.update(answer_type_dict_)
            answer_type_prob_dict.update(answer_type_prob_dict_)
            answer_dict.update(answer_dict_)

            predict_support_np = valid_dict['supp_np']
            batch_ids = valid_dict['ids']
            ###
            total_record_num = total_record_num + predict_support_np.shape[0]
            ###
            for i in range(predict_support_np.shape[0]):
                cur_sp_pred = [[] for _ in range(N_thresh)]
                cur_id = batch_ids[i]

                for j in range(predict_support_np.shape[1]):
                    if j >= len(self.dev_example_dict[cur_id].sent_names):
                        break
                    for thresh_i in range(N_thresh):
                        if predict_support_np[i, j] > thresholds[thresh_i]:
                            cur_sp_pred[thresh_i].append(self.dev_example_dict[cur_id].sent_names[j])

                for thresh_i in range(N_thresh):
                    if cur_id not in total_sp_dict[thresh_i]:
                        total_sp_dict[thresh_i][cur_id] = []
                    total_sp_dict[thresh_i][cur_id].extend(cur_sp_pred[thresh_i])

        def choose_best_threshold(ans_dict, pred_file):
            best_joint_f1 = 0
            best_metrics = None
            best_threshold = 0
            #################
            metric_dict = {}
            #################
            for thresh_i in range(N_thresh):
                prediction = {'answer': ans_dict,
                              'sp': total_sp_dict[thresh_i],
                              'type': answer_type_dict,
                              'type_prob': answer_type_prob_dict}
                tmp_file = os.path.join(os.path.dirname(pred_file), 'tmp_{}.json'.format(self.trainer.root_gpu))
                with open(tmp_file, 'w') as f:
                    json.dump(prediction, f)
                metrics = hotpot_eval(tmp_file, self.hparams.dev_gold_file)
                if metrics['joint_f1'] >= best_joint_f1:
                    best_joint_f1 = metrics['joint_f1']
                    best_threshold = thresholds[thresh_i]
                    best_metrics = metrics
                    shutil.move(tmp_file, pred_file)
                #######
                metric_dict[thresh_i] = (metrics['em'], metrics['f1'], metrics['sp_em'], metrics['sp_f1'], metrics['joint_em'], metrics['joint_f1'])
                #######
            return best_metrics, best_threshold, metric_dict

        output_pred_file = os.path.join(self.hparams.exp_name, f'pred.epoch_{self.current_epoch + 1}.gpu_{self.trainer.root_gpu}.json')
        output_eval_file = os.path.join(self.hparams.exp_name, f'eval.epoch_{self.current_epoch + 1}.gpu_{self.trainer.root_gpu}.txt')
        ####+++++
        best_metrics, best_threshold, metric_dict = choose_best_threshold(answer_dict, output_pred_file)
        ####++++++
        logging.info('Leader board evaluation completed over {} records with threshold = {:.4f}'.format(total_record_num, best_threshold))
        log_metrics(mode='Evaluation epoch {} gpu {}'.format(self.current_epoch, self.trainer.root_gpu), metrics=best_metrics)
        logging.info('*' * 75)
        ####++++++
        for key, value in metric_dict.items():
            str_value = ['{:.4f}'.format(_) for _ in value]
            logging.info('threshold {:.4f}: \t metrics: {}'.format(thresholds[key], str_value))
        ####++++++
        json.dump(best_metrics, open(output_eval_file, 'w'))
        #############################################################################
        # self.log('valid_loss', avg_loss, 'joint_f1', best_metrics['joint_f1'], on_epoch=True, prog_bar=True, sync_dist=True)
        joint_f1 = torch.Tensor([best_metrics['joint_f1']])[0].to(avg_loss.device)
        self.log('joint_f1', joint_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        #############################################################################
        return best_metrics, best_threshold

    def configure_optimizers(self):
        # "Prepare optimizer and schedule (linear warmup and decay)"
        if self.hparams.optimizer == 'Adam':
            if self.hparams.learning_rate_schema == 'fixed':
                return self.fixed_learning_rate_optimizers()
            elif self.hparamself.learning_rate_schema == 'layer_decay':
                return self.layer_wise_learning_rate_optimizer()
            else:
                raise 'Wrong lr setting method = {}'.format(self.hparams.learning_rate_schema)
        else:
            return self.rec_adam_learning_optimizer()

    def fixed_learning_rate_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if
                           (p.requires_grad) and (any(nd in n for nd in no_decay))],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        # )
        if self.hparams.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.hparams.warmup_steps,
                                                                           num_training_steps=self.total_steps)
        else:
            raise '{} is not supported'.format(self.hparams.lr_scheduler)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def layer_wise_learning_rate_optimizer(self):
        "Prepare optimizer and schedule (linear warmup and decay)"
        encoder_layer_number_dict = {'roberta-large': 24, 'albert-xxlarge-v2': 1}
        assert self.hparams.encoder_name_or_path in encoder_layer_number_dict

        def achieve_module_groups(encoder, number_of_layer, number_of_groups):
            layer_num_each_group = number_of_layer // number_of_groups
            number_of_divided_groups = number_of_groups + 1 if number_of_layer % number_of_groups > 0 else number_of_groups
            groups = []
            groups.append([encoder.embeddings, *encoder.encoder.layer[:layer_num_each_group]])
            for group_id in range(1, number_of_divided_groups):
                groups.append(
                    [*encoder.encoder.layer[(group_id * layer_num_each_group):((group_id + 1) * layer_num_each_group)]])
            return groups, number_of_divided_groups

        if self.hparams.encoder_name_or_path == 'roberta-large':
            encoder_layer_number = encoder_layer_number_dict[self.hparams.encoder_name_or_path]
            encoder_group_number = encoder_layer_number
            module_groups, encoder_group_number = achieve_module_groups(encoder=self.encoder,
                                                                        number_of_layer=encoder_layer_number,
                                                                        number_of_groups=encoder_group_number)
            module_groups.append([self.model])
            assert len(module_groups) == encoder_group_number + 1
        elif self.hparams.encoder_name_or_path == 'albert-xxlarge-v2':
            module_groups = []
            module_groups.append([self.encoder])
            module_groups.append([self.model])
            assert len(module_groups) == 2
        else:
            raise 'Not supported {}'.format(self.hparams.encoder_name_or_path)

        def achieve_parameter_groups(module_group, weight_decay, lr):
            named_parameters = []
            no_decay = ["bias", "LayerNorm.weight"]
            for module in module_group:
                named_parameters += module.named_parameters()
            grouped_parameters = [
                {
                    "params": [p for n, p in named_parameters if
                               (p.requires_grad) and (not any(nd in n for nd in no_decay))],
                    "weight_decay": weight_decay, 'lr': lr
                },
                {
                    "params": [p for n, p in named_parameters if
                               (p.requires_grad) and (any(nd in n for nd in no_decay))],
                    "weight_decay": 0.0, 'lr': lr
                }
            ]
            return grouped_parameters

        optimizer_grouped_parameters = []
        for idx, module_group in enumerate(module_groups):
            lr = self.hparams.learning_rate * (10.0 ** idx)
            logging.info('group {} lr = {}'.format(idx, lr))
            grouped_parameters = achieve_parameter_groups(module_group=module_group,
                                                          weight_decay=self.hparams.weight_decay,
                                                          lr=lr)
            optimizer_grouped_parameters += grouped_parameters

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        # )
        if self.hparams.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.hparams.warmup_steps,
                                                                           num_training_steps=self.total_steps)
        else:
            raise '{} is not supported'.format(self.hparams.lr_scheduler)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    def rec_adam_learning_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        new_model = self.model
        args = self.hparams
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
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        # )
        if self.hparams.lr_scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer=optimizer,
                                                        num_warmup_steps=self.hparams.warmup_steps,
                                                        num_training_steps=self.total_steps)
        elif self.hparams.lr_scheduler == 'cosine_restart':
            scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer=optimizer,
                                                                           num_warmup_steps=self.hparams.warmup_steps,
                                                                           num_training_steps=self.total_steps)
        else:
            raise '{} is not supported'.format(self.hparams.lr_scheduler)
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    # def configure_optimizers(self):
    #     # "Prepare optimizer and schedule (linear warmup and decay)"
    #     if self.hparams.learning_rate_schema == 'fixed':
    #         return self.fixed_learning_rate_optimizers()
    #     else:
    #         return self.layer_wise_learning_rate_optimizer()
    #
    # def fixed_learning_rate_optimizers(self):
    #     "Prepare optimizer and schedule (linear warmup and decay)"
    #     no_decay = ["bias", "LayerNorm.weight"]
    #     optimizer_grouped_parameters = [
    #         {
    #             "params": [p for n, p in self.named_parameters() if
    #                        (p.requires_grad) and (not any(nd in n for nd in no_decay))],
    #             "weight_decay": self.hparams.weight_decay,
    #         },
    #         {
    #             "params": [p for n, p in self.named_parameters() if
    #                        (p.requires_grad) and (any(nd in n for nd in no_decay))],
    #             "weight_decay": 0.0,
    #         }
    #     ]
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
    #     )
    #     scheduler = {
    #         'scheduler': scheduler,
    #         'interval': 'step',
    #         'frequency': 1
    #     }
    #     return [optimizer], [scheduler]
    #
    # def layer_wise_learning_rate_optimizer(self):
    #     "Prepare optimizer and schedule (linear warmup and decay)"
    #     encoder_layer_number_dict = {'roberta-large': 24, 'albert-xxlarge-v2': 1}
    #     assert self.hparams.encoder_name_or_path in encoder_layer_number_dict
    #
    #     def achieve_module_groups(encoder, number_of_layer, number_of_groups):
    #         layer_num_each_group = number_of_layer // number_of_groups
    #         number_of_divided_groups = number_of_groups + 1 if number_of_layer % number_of_groups > 0 else number_of_groups
    #         groups = []
    #         groups.append([encoder.embeddings, *encoder.encoder.layer[:layer_num_each_group]])
    #         for group_id in range(1, number_of_divided_groups):
    #             groups.append(
    #                 [*encoder.encoder.layer[(group_id * layer_num_each_group):((group_id + 1) * layer_num_each_group)]])
    #         return groups, number_of_divided_groups
    #     if self.hparams.encoder_name_or_path == 'roberta-large':
    #         encoder_layer_number = encoder_layer_number_dict[self.hparams.encoder_name_or_path]
    #         encoder_group_number = 2
    #         module_groups, encoder_group_number = achieve_module_groups(encoder=self.encoder, number_of_layer=encoder_layer_number,
    #                                               number_of_groups=encoder_group_number)
    #         module_groups.append([self.model])
    #         assert len(module_groups) == encoder_group_number + 1
    #     elif self.hparams.encoder_name_or_path == 'albert-xxlarge-v2':
    #         module_groups = []
    #         module_groups.append([self.encoder])
    #         module_groups.append([self.model])
    #         assert len(module_groups) == 2
    #     else:
    #         raise 'Not supported {}'.format(self.hparams.encoder_name_or_path)
    #
    #     def achieve_parameter_groups(module_group, weight_decay, lr):
    #         named_parameters = []
    #         no_decay = ["bias", "LayerNorm.weight"]
    #         for module in module_group:
    #             named_parameters += module.named_parameters()
    #         grouped_parameters = [
    #             {
    #                 "params": [p for n, p in named_parameters if
    #                            (p.requires_grad) and (not any(nd in n for nd in no_decay))],
    #                 "weight_decay": weight_decay, 'lr': lr
    #             },
    #             {
    #                 "params": [p for n, p in named_parameters if
    #                            (p.requires_grad) and (any(nd in n for nd in no_decay))],
    #                 "weight_decay": 0.0, 'lr': lr
    #             }
    #         ]
    #         return grouped_parameters
    #
    #     optimizer_grouped_parameters = []
    #     for idx, module_group in enumerate(module_groups):
    #         lr = self.hparams.learning_rate * (10.0**idx)
    #         logging.info('group {} lr = {}'.format(idx, lr))
    #         grouped_parameters = achieve_parameter_groups(module_group=module_group,
    #                                                       weight_decay=self.hparams.weight_decay,
    #                                                       lr=lr)
    #         optimizer_grouped_parameters += grouped_parameters
    #
    #     optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
    #     )
    #     scheduler = {
    #         'scheduler': scheduler,
    #         'interval': 'step',
    #         'frequency': 1
    #     }
    #     return [optimizer], [scheduler]