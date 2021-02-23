import json
import sys

from tqdm import tqdm
assert len(sys.argv) == 4
raw_data = json.load(open(sys.argv[1], 'r'))
para_file = sys.argv[2]
with open(para_file, 'r', encoding='utf-8') as reader:
    para_data = json.load(reader)
#################################
reverse_output_file = sys.argv[3]
################################
selected_para_dict_reverse = {}
################################
for case in tqdm(raw_data):
    guid = case['_id']
    ##############################################
    ir_selected_paras = para_data[guid]
    selected_para_dict_reverse[guid] = []
    assert len(ir_selected_paras) == 3
    if len(ir_selected_paras[0]) == 2:
        reverse_ir_paras_1st = [ir_selected_paras[0][1], ir_selected_paras[0][0]]
    else:
        reverse_ir_paras_1st = ir_selected_paras[0]
    selected_para_dict_reverse[guid].append(reverse_ir_paras_1st)
    selected_para_dict_reverse[guid].append(ir_selected_paras[1])
    if len(ir_selected_paras[2]) == 2:
        reverse_ir_paras_3rd = [ir_selected_paras[2][1], ir_selected_paras[2][0]]
    else:
        reverse_ir_paras_3rd = ir_selected_paras[2]
    selected_para_dict_reverse[guid].append(reverse_ir_paras_3rd)
json.dump(selected_para_dict_reverse, open(reverse_output_file, 'w'))