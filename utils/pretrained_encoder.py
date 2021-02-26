###Roberta-large on squad2 --> ahotrod/roberta_large_squad2
###AL-BERT-large on squad2
from os.path import join


from envs import PRETRAINED_MODEL_FOLDER
from model_envs import MODEL_CLASSES

config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
model_name = join(PRETRAINED_MODEL_FOLDER, 'ahotrod')

model = model_class.from_pretrained(model_name)

# tokenizer
#
# config_class, model_class, tokenizer_class = MODEL_CLASSES['roberta']
#
# tokenizer = tokenizer_class.from_pretrained('t'