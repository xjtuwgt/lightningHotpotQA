from utils.gpu_utils import gpu_id_setting
import torch
if torch.cuda.is_available():
    gpus_count = torch.cuda.device_count()
    gpu_id_list, true_gpus_count = gpu_id_setting(gpus=gpus_count)
    print('Number of available gpus = {} with device id = {}'.format(true_gpus_count, gpu_id_list))
else:
    print('cuda is not available')