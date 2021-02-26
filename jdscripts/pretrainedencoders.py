###Roberta-large on squad2 --> ahotrod/roberta_large_squad2
###AL-BERT-large on squad2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from hgntransformers import RobertaTokenizer, RobertaConfig, RobertaModel, RobertaForQuestionAnswering

from envs import PRETRAINED_MODEL_FOLDER
from model_envs import MODEL_CLASSES

model_name = join(PRETRAINED_MODEL_FOLDER, 'ahotrod')

model = RobertaForQuestionAnswering.from_pretrained(model_name)

# tokenizer
#
# config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
#
# tokenizer = tokenizer_class.from_pretrained('t'



