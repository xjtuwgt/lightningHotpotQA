from envs import OUTPUT_FOLDER
import os

def list_all_folders(d):
    folder_names = [os.path.join(d, o) for o in os.listdir(d)
     if os.path.isdir(os.path.join(d, o))]
    return folder_names

def list_all_txt_files(path):
    files = os.listdir(path)
    files_txt = [i for i in files if i.endswith('.txt')]
    eval_file_names = [i for i in files_txt if i.startswith('eval.epoch')]
    return eval_file_names

def best_metric_collection():
    folder_names = list_all_folders(d=OUTPUT_FOLDER)
    for x in folder_names:
        print(x)