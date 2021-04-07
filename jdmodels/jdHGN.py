from models.layers import mean_pooling, BiAttention, LSTMWrapper, GatedAttention
from jdmodels.jdlayers import GraphBlock, init_encoder_graph_node_feature, ParaSentEntPredictionLayer, PredictionLayer
from torch import nn

class HierarchicalGraphNetwork(nn.Module):
    """
    Packing Query Version
    """
    def __init__(self, config):
        super(HierarchicalGraphNetwork, self).__init__()
        self.config = config
        self.max_query_length = self.config.max_query_length
        self.bi_attention = BiAttention(input_dim=config.input_dim,
                                        memory_dim=config.input_dim,
                                        hid_dim=config.hidden_dim,
                                        dropout=config.bi_attn_drop)
        self.bi_attn_linear = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.hidden_dim = config.hidden_dim
        self.sent_lstm = LSTMWrapper(input_dim=config.hidden_dim,
                                     hidden_dim=config.hidden_dim,
                                     n_layer=config.lstm_layer,
                                     dropout=config.lstm_drop) ### output: 2 * self.hidden_dim
        #####the output of lstm is 2 * self.hidden_dim, the dimension is same as hidden in GAT

        self.q_map = nn.Linear(in_features=config.input_dim, out_features=self.hidden_dim * 2)
        if self.q_map:
            nn.init.xavier_uniform_(self.q_map.weight, gain=1.414)

        self.graph_blocks = nn.ModuleList()
        self.graph_blocks.append(GraphBlock(self.config.q_attn, config, input_dim=self.hidden_dim * 2, hidden_dim=self.hidden_dim))
        for _ in range(self.config.num_gnn_layers-1):
            self.graph_blocks.append(GraphBlock(self.config.q_attn, config, input_dim=self.hidden_dim, hidden_dim=self.hidden_dim))
        self.ctx_attention = GatedAttention(input_dim=config.hidden_dim*2,
                                            memory_dim=config.hidden_dim,
                                            hid_dim=self.config.ctx_attn_hidden_dim,
                                            dropout=config.bi_attn_drop,
                                            gate_method=self.config.ctx_attn)
        self.para_sent_ent_predict_layer = ParaSentEntPredictionLayer(self.config, hidden_dim=self.hidden_dim)
        self.predict_layer = PredictionLayer(self.config)

    def forward(self, batch, return_yp):
        query_mapping = batch['query_mapping']
        context_encoding = batch['context_encoding']
        print('context encode', context_encoding.shape)
        # extract query encoding
        trunc_query_mapping = query_mapping[:, :self.max_query_length].contiguous()
        trunc_query_state = (context_encoding * query_mapping.unsqueeze(2))[:, :self.max_query_length, :].contiguous()
        # bert encoding query vec
        query_vec = mean_pooling(trunc_query_state, trunc_query_mapping)
        attn_output, trunc_query_state = self.bi_attention(context_encoding,
                                                           trunc_query_state,
                                                           trunc_query_mapping)
        input_state = self.bi_attn_linear(attn_output) # N x L x d
        input_state = self.sent_lstm(input_state, batch['context_lens'])
        ###############################################################################################################
        query_vec = self.q_map(query_vec)
        ################################################################################################################
        graph_state_dict = init_encoder_graph_node_feature(batch=batch, input_state=input_state, hidden_dim=self.hidden_dim)
        graph_state, graph_mask = None, None
        ################################################################################################################
        for l in range(self.config.num_gnn_layers):
            graph_state, graph_state_dict, graph_mask, query_vec = self.graph_blocks[l](batch=batch,
                                                                                        graph_state_dict=graph_state_dict,
                                                                                        query_vec=query_vec)
        input_state, _ = self.ctx_attention(input_state, graph_state, graph_mask.squeeze(-1))
        print('input state', input_state.shape)
        para_predictions, sent_predictions, ent_predictions = self.para_sent_ent_predict_layer.forward(batch=batch,
                                                                                     graph_state_dict=graph_state_dict)
        predictions = self.predict_layer(batch, input_state, packing_mask=query_mapping,
                                         return_yp=return_yp)
        if return_yp:
            start, end, q_type, yp1, yp2 = predictions
            return start, end, q_type, para_predictions, sent_predictions, ent_predictions, yp1, yp2
        else:
            start, end, q_type = predictions
            return start, end, q_type, para_predictions, sent_predictions, ent_predictions