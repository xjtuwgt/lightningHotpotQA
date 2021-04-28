from torch import nn
import torch
import logging
from csr_mhqa.utils import load_encoder_model
from models.HGN import HierarchicalGraphNetwork

class UnifiedHGNModel(nn.Module):
    def __init__(self, config):
        super(UnifiedHGNModel, self).__init__()
        self.config = config
        self.encoder, _ = load_encoder_model(self.config.encoder_name_or_path, self.config.model_type)
        self.model = HierarchicalGraphNetwork(config=self.config)
        if self.config.encoder_path is not None:
            self.initialize_model()

    def initialize_model(self):
        if self.config.encoder_path is not None:
            logging.info("Loading encoder from: {}".format(self.config.encoder_path))
            self.encoder.load_state_dict(torch.load(self.config.encoder_path))
            logging.info("Loading encoder completed")
        else:
            raise 'The encoder name is none {}'.format(self.config.model)
        if self.config.model_path is not None:
            logging.info("Loading model from: {}".format(self.config.model_path))
            self.model.load_state_dict(torch.load(self.config.model_path))
            logging.info("Loading model completed")
        else:
            raise 'The model name is none'.format(self.config.model)

    def forward(self, batch, return_yp=True, return_cls=True):
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
        start, end, q_type, paras, sents, ents, y1, y2, cls_emb = self.model.forward(batch, return_yp=return_yp, return_cls=return_cls)
        return start, end, q_type, paras, sents, ents, y1, y2, cls_emb