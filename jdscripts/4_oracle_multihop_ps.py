import json
import sys

from tqdm import tqdm
assert len(sys.argv) == 3

raw_data = json.load(open(sys.argv[1], 'r'))
output_file = sys.argv[2]

para_num = []
selected_para_dict = {}

def build_dict(title_list):
    title_to_id, id_to_title = {}, {}

    for idx, title in enumerate(title_list):
        id_to_title[idx] = title
        title_to_id[title] = idx

    return title_to_id, id_to_title

for case in tqdm(raw_data):
    guid = case['_id']
    context = dict(case['context'])
    support_facts = case['supporting_facts']
    selected_para_dict[guid] = []

    title_to_id, id_to_title = build_dict(context.keys())
    sel_para_idx = [0] * len(context)
    for fact in support_facts:
        sel_para_idx[title_to_id[fact[0]]] = 1

    selected_para_dict[guid].append([id_to_title[i] for i in range(len(context)) if sel_para_idx[i] == 1])
    # second hop: use hyperlink
    second_hop_titles = []
    selected_para_dict[guid].append(second_hop_titles)

    # others, keep a high recall
    other_titles = []
    selected_para_dict[guid].append(other_titles)
    para_num.append(sum(sel_para_idx))

json.dump(selected_para_dict, open(output_file, 'w'))
