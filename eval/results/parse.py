import json
from pathlib import Path

DATA_FILE = "samples_when2call-qwen2_5_2026-11-11T09-34-28.476972.jsonl"
DIR_PATH = Path(__file__).resolve().parent
FILE_PATH = DIR_PATH / DATA_FILE

data = []
with open(FILE_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            data.append(json.loads(line))

# Map the predicted index back to its string label
id_to_label = {
    0: 'direct',
    1: 'tool_call',
    2: 'request_for_info',
    3: 'cannot_answer'
}

# Map full labels to your dictionary short keys
short_keys = {
    'direct': 'direct',
    'tool_call': 'tc',
    'request_for_info': 'rfi',
    'cannot_answer': 'ca'
}

tc_missed = {'ca': [], 'rfi': [], 'direct': []}
ca_missed = {'tc': [], 'rfi': [], 'direct': []}
rfi_missed = {'ca': [], 'tc': [], 'direct': []}

for line in data:
    # 1. Determine the predicted index
    resp = [float(r[0]) for r in line['filtered_resps']]
    pred_idx = max(range(len(resp)), key=resp.__getitem__)
    
    # 2. Extract ground truth and prediction as strings
    target_label = line['doc']['correct_answer']
    pred_label = id_to_label[pred_idx]
    
    # 3. Route to the correct bucket if it's a miss
    if target_label != pred_label:
        pred_k = short_keys[pred_label]
        
        if target_label == 'tool_call':
            tc_missed[pred_k].append(line)
        elif target_label == 'request_for_info':
            rfi_missed[pred_k].append(line)
        elif target_label == 'cannot_answer':
            ca_missed[pred_k].append(line)

missed = {
    'tc': tc_missed,
    'ca': ca_missed,
    'rfi': rfi_missed,
}

with open(DIR_PATH / 'missed_cases.json', 'w', encoding='utf-8') as f:
    json.dump(missed, f, indent=2)