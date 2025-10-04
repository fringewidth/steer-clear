import torch
import json
import os
sys.path.append(os.path.abspath("../train"))
from train_commons import SharedLoRA, PromptDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd

phase = 2

ckpts = os.listdir(f"../divergence_adapters_phase{phase}")
ckpts = [os.path.join(f"../divergence_adapters_phase{phase}", ckpt) for ckpt in ckpts]

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
rank = 2
scaling = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_csv = pd.read_csv("../datasets/test.csv")
full_prompts = PromptDataset(test_csv, prompt_column='full_prompt')
original_completions = test_csv['original_completion']


def apply_adapter_hook(module, input, output):
    if isinstance(output, tuple):
        hidden_state = output[0]
        modified_hidden_state = shared_adapter(hidden_state)
        return (modified_hidden_state,) + output[1:]
    else:
        modified_hidden_state = shared_adapter(output)
        return modified_hidden_state

for ckpt in ckpts:
    output_file = f"../model_outputs/phase{phase}/lora/ckpt_{ckpt.split('_')[-1].split('.')[0]}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        dtype=torch.bfloat16,
    )

    hidden_size = model.config.hidden_size
    shared_adapter = SharedLoRA(hidden_size, rank=rank, scaling=scaling).to(device, dtype=torch.bfloat16)
    shared_adapter.load_state_dict(torch.load(ckpt, map_location=device))

    hook_handles = []
    for layer in model.model.layers:
        handle = layer.register_forward_hook(apply_adapter_hook)
        hook_handles.append(handle)

    model_outputs = []
    for i, (prompt, completion) in enumerate(zip(full_prompts, original_completions)):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=50, eos_token_id=im_end_token_id)
        generated_tokens = output[0][inputs['input_ids'].shape[1]:]
        output_txt = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        model_output = {
            "prompt_id": i,
            "prompt": prompt,
            "original_completion": completion,
            "output": output_txt
        }
        model_outputs.append(model_output)
    
    with open(output_file, "w") as f:
        json.dump(model_outputs, f, indent=2)

    for handle in hook_handles:
        handle.remove()
    del model


