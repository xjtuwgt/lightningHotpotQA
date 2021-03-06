import os
from numpy import random
import numpy as np
import shutil
import json
from envs import PROJECT_FOLDER

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def single_task_trial(search_space: dict, rand_seed=42):
    parameter_dict = {}
    for key, value in search_space.items():
        parameter_dict[key] = rand_search_parameter(value)
    parameter_dict['seed'] = rand_seed
    exp_name = 'train.graph.' + parameter_dict['model_type'] + '.bs' + str(parameter_dict['per_gpu_train_batch_size']) + '.as' + str(parameter_dict['gradient_accumulation_steps']) + \
               '.lr' + str(parameter_dict['learning_rate']) +'.lrs' + str(parameter_dict['learning_rate_schema']) + '.lrd' + str(parameter_dict['layer_wise_lr_decay']) + \
               '.gnn' + str(parameter_dict['gnn'].replace(':','').replace(',', '.')) +\
               '.data' +str(parameter_dict['daug_type']) + parameter_dict['optimizer'] + '.' + parameter_dict['lr_scheduler'] + '.seed' +str(rand_seed)
    parameter_dict['exp_name'] = exp_name
    return parameter_dict

def rand_search_parameter(space: dict):
    para_type = space['type']
    if para_type == 'fixed':
        return space['value']
    if para_type == 'choice':
        candidates = space['values']
        value = random.choice(candidates, 1).tolist()[0]
        return value
    if para_type == 'range':
        log_scale = space.get('log_scale', False)
        low, high = space['bounds']
        if log_scale:
            value = random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
            value = np.exp(value)
        else:
            value = random.uniform(low=low, high=high, size=1)[0]
        return value
    else:
        raise ValueError('Training batch mode %s not supported' % para_type)

def HypeParameterSpace():
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [3e-5, 2e-5, 1e-5, 1.5e-5]} #3e-5, 5e-5, 1e-4, 1.5e-4
    layer_wise_lr_decay = {'name': 'layer_wise_lr_decay', 'type': 'choice', 'values': [0.9]}
    gradient_accumulation_steps = {'name': 'gradient_accumulation_steps', 'type': 'choice', 'values': [1,2,4]}
    sent_lambda = {'name': 'sent_lambda', 'type': 'choice', 'values': [5]} ##
    frozen_layer_num = {'name': 'frozen_layer_number', 'type': 'choice', 'values': [0]}
    gnn_drop = {'name': 'gnn_drop', 'type': 'choice', 'values': [0.3]} #0.3
    gnn_attn_drop = {'name': 'gnn_attn_drop', 'type': 'choice', 'values': [0.3]}  # 0.3
    bi_attn_drop = {'name': 'bi_attn_drop', 'type': 'choice', 'values': [0.3]}
    trans_drop = {'name': 'trans_drop', 'type': 'choice', 'values': [0.3]}
    lstm_drop = {'name': 'lstm_drop', 'type': 'choice', 'values': [0.3]}
    num_train_epochs = {'name': 'num_train_epochs', 'type': 'choice', 'values': [8]}
    devf_type = {'name': 'devf_type', 'type': 'choice', 'values': ['long_low']}
    daug_type = {'name': 'daug_type', 'type': 'choice', 'values': ['hgn_long_docred_low']} #
    num_edge_type = {'name': 'num_edge_type', 'type': 'choice', 'values': [8]} # if SAE, then this number should be 9
    ctx_attn_hidden_dim = {'name': 'ctx_attn_hidden_dim', 'type': 'choice', 'values': [300]} # 300
    hidden_dim = {'name': 'hidden_dim', 'type': 'choice', 'values': [300]} # 300
    learning_rate_schema = {'name': 'learning_rate_schema', 'type': 'choice', 'values': ['layer_decay']}
    gnn = {'name': 'gnn', 'type': 'choice', 'values': ['gat:1,4']} ##'gat:1,2' 'gat:1,4'
    per_gpu_train_batch_size = {'name': 'per_gpu_train_batch_size', 'type': 'choice', 'values': [2]}
    model_type = {'name': 'model_type', 'type': 'choice', 'values': ['roberta']}
    fine_tuned_encoder = {'name': 'fine_tuned_encoder', 'type': 'choice', 'values': ['roberta/roberta-large_hgn']} #'ahotrod/roberta_large_squad2'
    encoder_name_or_path = {'name': 'encoder_name_or_path', 'type': 'choice', 'values': ['roberta-large']}
    # per_gpu_train_batch_size = {'name': 'per_gpu_train_batch_size', 'type': 'choice', 'values': [2]}
    # model_type = {'name': 'model_type', 'type': 'choice', 'values': ['albert']}
    # fine_tuned_encoder = {'name': 'fine_tuned_encoder', 'type': 'choice', 'values': ['albert/albert-xxlarge-v2_hotpotqa']} #'ahotrod/roberta_large_squad2'
    # encoder_name_or_path = {'name': 'encoder_name_or_path', 'type': 'choice', 'values': ['albert-xxlarge-v2']}
    optimizer = {'name': 'optimizer', 'type': 'choice', 'values': ['RecAdam']} #RecAdam
    lr_scheduler = {'name': 'lr_scheduler', 'type': 'choice', 'values': ['cosine']}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, per_gpu_train_batch_size, gradient_accumulation_steps, sent_lambda, frozen_layer_num, layer_wise_lr_decay,
                    lr_scheduler, gnn, fine_tuned_encoder, daug_type, devf_type, ctx_attn_hidden_dim, hidden_dim, learning_rate_schema,
                    gnn_attn_drop, optimizer,
                    gnn_drop, num_edge_type, bi_attn_drop, trans_drop, lstm_drop, num_train_epochs, model_type, encoder_name_or_path]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num, seed=42):
    relative_path = PROJECT_FOLDER + '/'
    json_file_path = 'configs/jdhgn/'
    job_path = 'jdhgn_jobs/'
    #================================================
    bash_save_path = relative_path + json_file_path
    jobs_path = relative_path + job_path
    if os.path.exists(jobs_path):
        remove_all_files(jobs_path)
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    if jobs_path and not os.path.exists(jobs_path):
        os.makedirs(jobs_path)
    ##################################################
    search_space = HypeParameterSpace()
    for i in range(task_num):
        rand_hype_dict = single_task_trial(search_space, seed+i)
        config_json_file_name = 'train.graph.' + rand_hype_dict['model_type'] + '.data.' + rand_hype_dict['daug_type'] \
                                +'.lr.'+ str(rand_hype_dict['learning_rate']) + '.lrs' + rand_hype_dict['learning_rate_schema'] + '.lrd' + \
                                str(rand_hype_dict['layer_wise_lr_decay']) + rand_hype_dict['optimizer'] + '.' + rand_hype_dict['lr_scheduler']+ \
                                rand_hype_dict['gnn'].replace(',', '.').replace(':', '') \
                                + '.seed' + str(rand_hype_dict['seed']) + '.json'
        with open(os.path.join(bash_save_path, config_json_file_name), 'w') as fp:
            json.dump(rand_hype_dict, fp)
        print('{}\n{}'.format(rand_hype_dict, config_json_file_name))
        with open(jobs_path + 'jd_hgn_' + config_json_file_name +'.sh', 'w') as rsh_i:
            command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 jdgraphtrain.py --config_file " + \
                        json_file_path + config_json_file_name
            # command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 jdtrain.py --config_file " + \
            #             json_file_path + config_json_file_name
            rsh_i.write(command_i)
            print('saving jobs at {}'.format(jobs_path + 'jd_hgn_' + config_json_file_name +'.sh'))
    print('{} jobs have been generated'.format(task_num))
