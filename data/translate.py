import json

def convert_to_HFChat(xLAM_example: dict) -> tuple[list[dict], list[dict]]:

    tools = json.loads(xLAM_example['tools']) if isinstance(xLAM_example['tools'], str) else xLAM_example['tools']
    answers = json.loads(xLAM_example['answers']) if isinstance(xLAM_example['answers'], str) else xLAM_example['answers']
    
    hf_tool_calls = []
    for call in answers:
        hf_tool_calls.append({
            "type": "function",
            "function": {
                "name": call["name"],
                # Hugging Face templates typically expect arguments as a JSON string, not a dict
                "arguments": json.dumps(call["arguments"], ensure_ascii=False)
            }
        })

    # Construct the messages list using the `tool_calls` key instead of formatting `content`
    messages = [
        {"role": "user", "content": xLAM_example['query']},
        {"role": "assistant", "content": "", "tool_calls": hf_tool_calls}
    ]
    
    return messages, tools