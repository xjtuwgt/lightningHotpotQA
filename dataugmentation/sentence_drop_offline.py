import json
import sys

from tqdm import tqdm
assert len(sys.argv) == 3
raw_data = json.load(open(sys.argv[1], 'r')) ## original json data
#################################
sentence_drop_output_file = sys.argv[2]
################################
selected_para_dict_reverse = {}
################################
for case in tqdm(raw_data):
    guid = case['_id']
    ##############################################
