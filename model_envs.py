from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)
from hgntransformers import (BertModel, XLNetModel, RobertaModel)
from electramodels.modeling_electra import ElectraModel, ElectraForPreTraining
from electramodels.tokenization_electra import ElectraTokenizer
from electramodels.configuration_electra import ElectraConfig


############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'electra': (ElectraConfig, ElectraModel, ElectraTokenizer)
    # 'unifiedqa': (T5Config, T5ForConditionalGeneration, AutoTokenizer)
}