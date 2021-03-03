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
DATASET_FOLDER = join(HOME_DATA_FOLDER, 'dataset')
MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models')
KNOWLEDGE_FOLDER = join(HOME_DATA_FOLDER, 'knowledge')
OUTPUT_FOLDER = join(HOME_DATA_FOLDER, 'outputs')
PRETRAINED_MODEL_FOLDER = join(HOME_DATA_FOLDER, 'models/pretrained')

os.environ['PYTORCH_PRETRAINED_BERT_CACHE'] = join(HOME_DATA_FOLDER, 'models', 'pretrained_cache')