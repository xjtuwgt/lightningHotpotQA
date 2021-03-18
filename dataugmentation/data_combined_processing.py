import gzip
import pickle
import os
from os.path import join
import itertools
from collections import ChainMap
from plmodels.pldata_processing import get_cached_filename

def get_pickle_file(file_name, gz=True):
    if gz:
        return gzip.open(file_name, 'rb')
    else:
        return open(file_name, 'rb')

def get_or_load(file):
    with get_pickle_file(file) as fin:
        print('loading', file)
        return pickle.load(fin)

def get_feature_file(data_dir, tag, f_type, config):
    cached_filename = get_cached_filename('{}_features'.format(f_type), config)
    return join(data_dir, tag, cached_filename)

def get_example_file(data_dir, tag, f_type, config):
    cached_filename = get_cached_filename('{}_examples'.format(f_type), config)
    return join(data_dir, tag, cached_filename)

def get_graph_file(data_dir, tag, f_type, config):
    cached_filename = get_cached_filename('{}_graphs'.format(f_type), config)
    return join(data_dir, tag, cached_filename)

def load_data_graph_feat_examples(data_folder, tag, f_type, config):
    feature_data_file_name = get_feature_file(data_dir=data_folder, tag=tag, f_type=f_type, config=config)
    example_data_file_name = get_example_file(data_dir=data_folder, tag=tag, f_type=f_type, config=config)
    graph_data_file_name = get_graph_file(data_dir=data_folder, tag=tag, f_type=f_type, config=config)

    feature_data = get_or_load(feature_data_file_name)
    example_data = get_or_load(example_data_file_name)
    graph_data = get_or_load(graph_data_file_name)
    print('Loading data from {}'.format(data_folder))
    print('Features {}, Examples {}, Graphs {}'.format(len(feature_data), len(example_data), len(graph_data)))
    return feature_data, example_data, graph_data

def combine_data_graph_feat_examples(data_folder, config, tag_f_type_list):
    feature_data_list = []
    example_data_list = []
    graph_data_list = []
    for tag, f_type in tag_f_type_list:
        feature_data, example_data, graph_data = load_data_graph_feat_examples(data_folder=data_folder, config=config, tag=tag, f_type=f_type)
        feature_data_list.append(feature_data)
        example_data_list.append(example_data)
        graph_data_list.append(graph_data)
    feature_data_array = list(itertools.chain(*feature_data_list))
    example_data_array = list(itertools.chain(*example_data_list))
    graph_data_array = dict(ChainMap(*graph_data_list))
    assert len(feature_data_array) == len(example_data_array) and len(feature_data_array) == len(graph_data_array)
    print('Combine data in {} by {}'.format(data_folder, tag_f_type_list))
    print('Features {}, Examples {}, Graphs {}'.format(len(feature_data_array), len(example_data_array), len(graph_data_array)))
    return feature_data_array, example_data_array, graph_data_array

def save_data_graph_feat(out_folder, features, examples, graphs, f_type, config):
    cached_features_file = os.path.join(out_folder,
                                        get_cached_filename('{}_features'.format(f_type), config))
    with gzip.open(cached_features_file, 'wb') as fout:
        pickle.dump(features, fout)
    print('Save {} features into {}'.format(len(features), cached_features_file))

    cached_examples_file = os.path.join(out_folder,
                                        get_cached_filename('{}_examples'.format(f_type), config))
    with gzip.open(cached_examples_file, 'wb') as fout:
        pickle.dump(examples, fout)
    print('Save {} examples into {}'.format(len(examples), cached_examples_file))

    cached_graph_file = os.path.join(out_folder,
                                       get_cached_filename('{}_graphs'.format(f_type), config))
    with gzip.open(cached_graph_file, 'wb') as fout:
        pickle.dump(graphs, fout)
    print('Save {} graphs into {}'.format(len(graphs), cached_graph_file))