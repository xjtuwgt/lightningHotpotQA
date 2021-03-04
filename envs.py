import os
import logging
import sys
from os.path import join

# Add submodule path into import paths
# is there a better way to handle the sub module path append problem?
PROJECT_FOLDER = os.path.dirname(__file__)
print('Project folder = {}'.format(PROJECT_FOLDER))
sys.path.append(join(PROJECT_FOLDER))

# Define the dataset folder and model folder based on environment
# HOME_DATA_FOLDER = '/ssd/HGN/data'
HOME_DATA_FOLDER = join(PROJECT_FOLDER, 'data')
print('data folder = {}'.format(HOME_DATA_FOLDER))
DATASET_FOLDER = join(HOME_DATA_FOLDER, 'dataset')
print('hotpotqa data folder = {}'.format(DATASET_FOLDER))
MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
print('pretrained model folder = {}'.format(MODEL_FOLDER))
KNOWLEDGE_FOLDER = join(HOME_DATA_FOLDER, 'knowledge')
print('knowledge folder = {}'.format(KNOWLEDGE_FOLDER))
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'outputs')
print('output result folder = {}'.format(OUTPUT_FOLDER))
PRETRAINED_MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models/pretrained')
print('pretrained model with finetuned folder = {}'.format(PRETRAINED_MODEL_FOLDER))
os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = join(HOME_DATA_FOLDER, 'models', 'pretrained_cache')