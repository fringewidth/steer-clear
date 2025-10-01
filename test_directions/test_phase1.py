import torch
import json
import os
import sys
sys.path.append(os.path.abspath("../train"))
from train_commons import PromptDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

phase = 1

directions = os.listdir(f"../directions/phase{phase}")
directions = [os.path.join(f"../directions/phase{phase}", ckpt) for ckpt in directions]

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_csv = pd.read_csv("../datasets/test.csv")
full_prompts = PromptDataset(test_csv, prompt_column='full_prompt')
original_completions = test_csv['original_completion']


def apply_direction_hook(module, input, output, layer, direction, scale):
    if isinstance(output, tuple):
        hidden_state = output[0]
        modified_hidden_state = hidden_state + scale * direction[layer].unsqueeze(0).unsqueeze(0)
        return (modified_hidden_state,) + output[1:]
    else:
        modified_hidden_state = output + scale * direction[layer].unsqueeze(0).unsqueeze(0)
        return modified_hidden_state

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
)

hidden_size = model.config.hidden_size


scaling = 2

for direction_name in directions:
    direction = torch.load(direction_name).to(device, dtype=torch.bfloat16)

    output_file = f"../model_outputs/phase{phase}/direction/ckpt_{direction_name.split('_')[-1].split('.')[0]}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(
            lambda module, input, output, l=i: apply_direction_hook(module, input, output, l, direction, scaling)
        )
        hook_handles.append(handle)

    model_outputs = []
    for i, (prompt, completion) in enumerate(zip(full_prompts, original_completions)):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(
            **inputs, 
            max_new_tokens=100, 
            eos_token_id=im_end_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
        generated_tokens = output[0][inputs['input_ids'].shape[1]:]
        output_txt = tokenizer.decode(generated_tokens, skip_special_tokens=True, eos_token_id=im_end_token_id)
        model_output = {
            "prompt_id": i,
            "prompt": prompt,
            "original_completion": ''.join(completion.split(prompt.split("<|im_start|>")[2])[1:]),
            "output": output_txt
        }
        model_outputs.append(model_output)

    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)

    for handle in hook_handles:
        handle.remove()

del model


