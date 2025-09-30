import sys
import os

sys.path.append(os.path.abspath('../train/'))
from train_commons import SharedLoRA, PromptDataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd

class BatchedSharedLoRA(torch.nn.Module):
    """
    Batched version of SharedLoRA that processes multiple adapters in parallel
    by concatenating their parameters and using batch operations.
    """
    def __init__(self, hidden_size, rank, scaling, num_adapters):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scaling = scaling
        self.num_adapters = num_adapters
        self.batched_lora_A = torch.nn.Parameter(torch.randn(num_adapters, hidden_size, rank))
        self.batched_lora_B = torch.nn.Parameter(torch.zeros(num_adapters, rank, hidden_size))
    
    def load_from_individual_adapters(self, adapter_paths, device):
        """Load parameters from individual adapter checkpoints"""
        lora_A_list = []
        lora_B_list = []
        
        for ckpt_path in adapter_paths:
            state_dict = torch.load(ckpt_path, map_location=device)
            lora_A_list.append(state_dict['lora_A'])
            lora_B_list.append(state_dict['lora_B'])
        
        self.batched_lora_A.data = torch.stack(lora_A_list, dim=0)
        self.batched_lora_B.data = torch.stack(lora_B_list, dim=0)
    
    def forward(self, x):
        x_expanded = x.unsqueeze(0).expand(self.num_adapters, -1, -1, -1)
        intermediate = torch.einsum('nbsh,nhr->nbsr', x_expanded, self.batched_lora_A)
        
        updates = torch.einsum('nbsr,nrh->nbsh', intermediate, self.batched_lora_B)
        update_norms = torch.norm(updates, p=2, dim=-1, keepdim=True) + 1e-8
        updates = updates / update_norms * self.scaling
        
        modified_outputs = x_expanded + updates

        return modified_outputs

def save_direction_vectors(accumulated_diffs, token_counts, phase, ckpt_idx_to_name, hidden_size, num_layers=48):
    """Extract and save final steering vectors for each checkpoint"""

    directions_dir = f"../directions/phase{phase}"
    os.makedirs(directions_dir, exist_ok=True)
    
    saved_files = []

    for ckpt_idx in range(len(ckpt_idx_to_name)):
        ckpt_name = ckpt_idx_to_name[ckpt_idx]
        layer_directions = []
        for layer_idx in range(num_layers):
            if token_counts[layer_idx] > 0:
                avg_direction = accumulated_diffs[layer_idx][:, ckpt_idx] / token_counts[layer_idx]
                layer_directions.append(avg_direction)
            else:
                layer_directions.append(torch.zeros(hidden_size, device=accumulated_diffs[0].device))
        all_directions = torch.stack(layer_directions, dim=0)

        filename = f"model_phase{phase}_{ckpt_name}.pt"
        filepath = os.path.join(directions_dir, filename)
        torch.save(all_directions.cpu(), filepath)
        
        print(f"Saved direction vectors: {filepath}")
        saved_files.append(filepath)
    
    return saved_files

phase = 1
checkpoint_dir = f"../divergence_adapters_phase{phase}"

if not os.path.exists(checkpoint_dir):
    raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")

ckpts = os.listdir(checkpoint_dir)
ckpts = [f for f in ckpts if f.endswith('.pth')]
if not ckpts:
    raise FileNotFoundError(f"No .pth checkpoint files found in {checkpoint_dir}")

ckpts = [os.path.join(checkpoint_dir, ckpt) for ckpt in ckpts]

model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
rank = 2
scaling = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Check if dataset exists
dataset_path = "../datasets/full_dataset.csv"
if not os.path.exists(dataset_path):
    raise FileNotFoundError(f"Dataset not found: {dataset_path}")

csv = pd.read_csv(dataset_path)
if csv.empty:
    raise ValueError("Dataset is empty")
if 'full_prompt' not in csv.columns:
    raise ValueError("Dataset missing required 'full_prompt' column")

full_prompts = PromptDataset(csv, prompt_column='full_prompt')

print(f"Found {len(ckpts)} checkpoints to process in parallel")
print(f"Processing {len(full_prompts)} prompts")

# Load the base model once
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
)
hidden_size = model.config.hidden_size
num_layers = len(model.model.layers)

ckpt_idx_to_name = {}
for idx, ckpt_path in enumerate(ckpts):
    ckpt_name = os.path.basename(ckpt_path).replace('.pth', '')
    ckpt_idx_to_name[idx] = ckpt_name

print("Checkpoint mapping:")
for idx, name in ckpt_idx_to_name.items():
    print(f"  Index {idx}: {name}")

batched_adapter = BatchedSharedLoRA(hidden_size, rank=rank, scaling=scaling, num_adapters=len(ckpts))
batched_adapter.load_from_individual_adapters(ckpts, device)
batched_adapter = batched_adapter.to(device, dtype=torch.bfloat16)

print(f"Loaded {len(ckpts)} adapters into batched processor")

accumulated_diffs = []
token_counts = []

for layer_idx in range(num_layers):
    # Initialize accumulation tensor
    accumulated_diffs.append(torch.zeros(hidden_size, len(ckpts), device=device, dtype=torch.bfloat16))
    token_counts.append(0)

current_layer_idx = 0

def apply_batched_adapter_hook(module, input, output):
    """Hook that applies ALL adapters in parallel and accumulates differences efficiently"""
    global current_layer_idx
    
    if isinstance(output, tuple):
        hidden_state = output[0]
    else:
        hidden_state = output
    modified_outputs = batched_adapter(hidden_state)
    hidden_state_expanded = hidden_state.unsqueeze(0).expand(len(ckpts), -1, -1, -1)

    diff_batch = modified_outputs - hidden_state_expanded
    diff_sum = torch.sum(diff_batch, dim=(1, 2)).T  # (hidden_size, num_adapters)

    accumulated_diffs[current_layer_idx] += diff_sum

    num_tokens = hidden_state.shape[0] * hidden_state.shape[1]  # batch_size * seq_len
    token_counts[current_layer_idx] += num_tokens
    
    return output

hook_handles = []
for layer_idx, layer in enumerate(model.model.layers):
    def make_hook(layer_index):
        def hook_fn(module, input, output):
            global current_layer_idx
            current_layer_idx = layer_index
            return apply_batched_adapter_hook(module, input, output)
        return hook_fn
    
    handle = layer.register_forward_hook(make_hook(layer_idx))
    hook_handles.append(handle)

print("Registered hooks for all layers")
print(f"Memory allocated: {accumulated_diffs[0].element_size() * accumulated_diffs[0].numel() * len(accumulated_diffs) / 1e9:.2f} GB")
print("Starting parallel processing of all adapters...")
for i, prompt in enumerate(full_prompts):
    if i % 10 == 0:
        print(f"Processing prompt {i}/{len(full_prompts)}")
        
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        generated_ids = model.generate(
            inputs['input_ids'],
            max_new_tokens=50,
            eos_token_id=im_end_token_id,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.9
        )
    
    with torch.no_grad():
        model(generated_ids)

print("Completed processing all prompts for all adapters!")
print(f"Total tokens processed per layer: {token_counts[0]}")

for handle in hook_handles:
    handle.remove()

print("Extracting and saving direction vectors for all checkpoints...")
saved_directions = save_direction_vectors(accumulated_diffs, token_counts, phase, ckpt_idx_to_name, hidden_size)

del model
del accumulated_diffs
del batched_adapter
torch.cuda.empty_cache()

print("Analysis complete! Extracted direction vectors for all checkpoints.")
print(f"Saved {len(saved_directions)} direction files to ../directions/ directory:")
for direction_path in saved_directions:
    print(f"  - {direction_path}")