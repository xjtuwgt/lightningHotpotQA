###Roberta-large on squad2 --> ahotrod/roberta_large_squad2
###AL-BERT-large on squad2
from os.path import join
from envs import PRETRAINED_MODEL_FOLDER, OUTPUT_FOLDER
from model_envs import MODEL_CLASSES
import torch
import os

def load_model(model_type, model_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model_name = join(PRETRAINED_MODEL_FOLDER, model_name)
    model = model_class.from_pretrained(model_name)
    return model

def save_model_as_pkl(encoder, model_name):
    pickle_model_name = join(PRETRAINED_MODEL_FOLDER, model_name, f'encoder.pkl')
    torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, pickle_model_name)
    print('Saved pickle name = {}'.format(pickle_model_name))

def load_hotpotqa_model(model_type, model_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name)
    encoder_pickle_name = join(OUTPUT_FOLDER, model_type, 'encoder.pkl')
    print('loading ')
    model.load_state_dict(torch.load(encoder_pickle_name))
    return model

def save_hotpotqa_model(encoder, model_type, model_name):
    model_type_folder = join(PRETRAINED_MODEL_FOLDER, model_type)
    if not os.path.exists(model_type_folder):
        os.makedirs(model_type_folder)
    model_type_name_folder = join(model_type_folder, model_name + '_hotpotqa')
    if not os.path.exists(model_type_name_folder):
        os.makedirs(model_type_name_folder)
    pickle_model_name = join(model_type_name_folder, f'encoder.pkl')
    torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, pickle_model_name)
    print('Saved pickle name = {}'.format(pickle_model_name))

def load_model_with_enconder(model_type, model_name, encoder_model_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name)
    encoder_pickle_name = join(PRETRAINED_MODEL_FOLDER, encoder_model_name, 'encoder.pkl')
    print('loading ')
    model.load_state_dict(torch.load(encoder_pickle_name))
    return model

def load_hgn_hotpotqa_model(model_type, model_name, exp_name, encoder_pkl_name):
    config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
    model = model_class.from_pretrained(model_name)
    encoder_pickle_name = join(OUTPUT_FOLDER, exp_name, encoder_pkl_name)
    print('loading parameter from {}'.format(encoder_pickle_name))
    model.load_state_dict(torch.load(encoder_pickle_name))
    return model

def save_hgn_hotpotqa_model(encoder, model_type, model_name):
    model_type_folder = join(PRETRAINED_MODEL_FOLDER, model_type)
    if not os.path.exists(model_type_folder):
        os.makedirs(model_type_folder)
    model_type_name_folder = join(model_type_folder, model_name + '_hgn')
    if not os.path.exists(model_type_name_folder):
        os.makedirs(model_type_name_folder)
    pickle_model_name = join(model_type_name_folder, f'encoder.pkl')
    torch.save({k: v.cpu() for k, v in encoder.state_dict().items()}, pickle_model_name)
    print('Saved pickle name = {}'.format(pickle_model_name))