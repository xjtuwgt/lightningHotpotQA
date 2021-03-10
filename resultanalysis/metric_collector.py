from envs import OUTPUT_FOLDER
import os
import json

def list_all_folders(d):
    folder_names = [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))]
    return folder_names

def list_all_txt_files(path):
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    eval_file_names = [i for i in files_txt if i.startswith('eval.epoch')]
    eval_file_names = [i for i in eval_file_names if 'gpu' not in i]
    eval_file_names = [i for i in eval_file_names if 'albert' not in i]
    return eval_file_names

def best_metric_collection():
    best_metric_dict = None
    best_joint_f1 = -1
    best_setting = None
    folder_names = list_all_folders(d=OUTPUT_FOLDER)
    for folder_idx, folder_name in enumerate(folder_names):
        eval_file_names = list_all_txt_files(path=folder_name)
        trim_folder_name = folder_name[(len(OUTPUT_FOLDER)+1):]
        for file_idx, file_name in enumerate(eval_file_names):
            print('{} | {} | {} | {}'.format(folder_idx, file_idx, trim_folder_name, file_name))
            with open(os.path.join(folder_name, file_name)) as fp:
                lines = fp.readlines()
                for line in lines:
                    metric_dict = json.loads(line)
                    for key, value in metric_dict.items():
                        metric_dict[key] = float(value)
                    if metric_dict['joint_f1'] > best_joint_f1:
                        best_joint_f1 = metric_dict['joint_f1']
                        best_setting = os.path.join(folder_name, file_name)
                        best_metric_dict = metric_dict
    print('*' * 75)
    print('Best joint F1 = {}\nSetting = {}'.format(best_joint_f1, best_setting))
    for key, value in best_metric_dict.items():
        print('{}: {}'.format(key, value))
    print('*' * 75)