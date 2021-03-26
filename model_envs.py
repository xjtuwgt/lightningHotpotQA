# from transformers import T5ForConditionalGeneration, AutoTokenizer, T5Config
from hgntransformers import (BertConfig, BertTokenizer, BertModel,
                             RobertaConfig, RobertaTokenizer, RobertaModel,
                             AlbertConfig, AlbertTokenizer, AlbertModel)
from hgntransformers import (BertModel, XLNetModel, RobertaModel)

from transformers import (ElectraConfig, ElectraTokenizer, ElectraModel, ElectraForMaskedLM)


############################################################
# Model Related Global Varialbes
############################################################

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, RobertaConfig, AlbertConfig)), ())

MODEL_CLASSES = {
    'bert': (BertConfig, BertModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertModel, AlbertTokenizer),
    'electra': (ElectraConfig, ElectraForMaskedLM, ElectraTokenizer)
    # 'unifiedqa': (T5Config, T5ForConditionalGeneration, AutoTokenizer)
}