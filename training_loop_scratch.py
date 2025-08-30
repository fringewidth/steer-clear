# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import os
import textwrap
import gc
# ==============================================================================

# -----------------------------
# Custom Shared LoRA Module
# -----------------------------
class SharedLoRA(nn.Module):
    """
    A single, shared LoRA module that will be applied to the output of every transformer block.
    This is a highly parameter-efficient way to introduce a global change to the model's behavior.
    """
    def __init__(self, hidden_size, rank, scaling=1.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(hidden_size, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        self.scaling = scaling
        
        # Initialize A with Kaiming uniform for better stability
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    def forward(self, x):
        """Applies the low-rank update to the input hidden state."""
        # Input x has shape (batch, seq_len, hidden_size)
        update = (x @ self.lora_A @ self.lora_B) * self.scaling
        return x + update

# -----------------------------
# Dataset
# -----------------------------
class PromptDataset(Dataset):
    """
    A simple dataset to load prompts from a pandas DataFrame.
    """
    def __init__(self, df: pd.DataFrame, prompt_column: str):
        # Ensure the column exists
        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found in the DataFrame.")
        self.prompts = df[prompt_column].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]


# %%
def load_base_model_and_tokenizer(args):
    """Loads the objects that are constant across all training runs."""
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a CUDA-enabled GPU.")
    
    print("--- Loading Base Model and Tokenizer (once) ---")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model_base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model_base.eval()
    for param in model_base.parameters():
        param.requires_grad = False
        
    print("--- Base Model and Tokenizer Loaded ---")
    return model_base, tokenizer


# %%
def run_single_training_cycle(args, model_base, tokenizer, run_idx):
    """
    Runs one full cycle of training and evaluation.
    It loads a new model to be tuned each time it's called.
    """
    device = "cuda"
    run_output_path = os.path.join(args.output_dir, f"divergence_adapter_run_{run_idx}.pth")
    im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    
    print(f"Loading a new, randomly initialized 'model_tuned' for run {run_idx}...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_tuned = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model_tuned.train()

    # --- Create and Inject a new Shared LoRA Adapter ---
    hidden_size = model_tuned.config.hidden_size
    shared_adapter = SharedLoRA(hidden_size, rank=args.lora_rank).to(device, dtype=torch.bfloat16)

    hook_handles = []
    def apply_adapter_hook(module, input, output):
        if isinstance(output, tuple):
            hidden_state = output[0]
            modified_hidden_state = shared_adapter(hidden_state)
            return (modified_hidden_state,) + output[1:]
        else:
            modified_hidden_state = shared_adapter(output)
            return modified_hidden_state

    for layer in model_tuned.model.layers:
        handle = layer.register_forward_hook(apply_adapter_hook)
        hook_handles.append(handle)

    num_trainable_params = sum(p.numel() for p in shared_adapter.parameters() if p.requires_grad)
    print(f"Run {run_idx}: Shared LoRA adapter created with {num_trainable_params:,} parameters.")

    optimizer = torch.optim.AdamW(shared_adapter.parameters(), lr=args.learning_rate)
    
    df = pd.read_csv(args.dataset_path)
    dataset = PromptDataset(df, prompt_column='full_prompt')
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # --- Training Loop ---
    print(f"Run {run_idx}: Starting divergence training...")
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Run {run_idx} Epoch {epoch+1}")
        for batch in pbar:
            prompt_text = batch[0]
            input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
            
            total_loss_for_prompt = 0
            
            for step in range(args.max_new_tokens):
                with torch.no_grad():
                    outputs_base = model_base(input_ids)
                    logits_base = outputs_base.logits[:, -1, :]
                    logprobs_p = F.log_softmax(logits_base, dim=-1)

                outputs_tuned = model_tuned(input_ids)
                logits_tuned = outputs_tuned.logits[:, -1, :]
                logprobs_q = F.log_softmax(logits_tuned, dim=-1)
                
                probs_p = logprobs_p.exp().detach()
                kl_div = (probs_p * (logprobs_p.detach() - logprobs_q)).sum(dim=-1)
                
                loss = -kl_div
                total_loss_for_prompt += loss.item()

                loss.backward()

                with torch.no_grad():
                    next_token = torch.multinomial(probs_p, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=-1)

                if next_token.item() in [tokenizer.eos_token_id, im_end_token_id]:
                    break
            
            optimizer.step()
            optimizer.zero_grad()
            avg_loss = total_loss_for_prompt / (step + 1)
            pbar.set_postfix({"avg_loss": f"{avg_loss:.4f}"})

    # --- Save ---
    print(f"\nRun {run_idx}: Training finished.")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(shared_adapter.state_dict(), run_output_path)
    print(f"Shared LoRA adapter weights saved to '{run_output_path}'")

    # --- IN-LINE EVALUATION ---
    print(f"\n--- Starting Evaluation for Run {run_idx} ---")
    model_tuned.eval()
    sample_prompts = df['full_prompt'].sample(n=args.num_eval_samples, random_state=42+run_idx).tolist()

    for i, prompt in enumerate(sample_prompts):
        print("\n" + "="*80)
        print(f"PROMPT:\n{textwrap.fill(prompt, 80)}")
        print("="*80)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        print("\n--- BASE MODEL OUTPUT ---")
        with torch.no_grad():
            outputs_base_gen = model_base.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id
            )
            base_text = tokenizer.decode(outputs_base_gen[0], skip_special_tokens=True)
            print(textwrap.fill(base_text.replace(prompt, "", 1).strip(), 80))

        print("\n--- DIVERGENT MODEL OUTPUT ---")
        with torch.no_grad():
            outputs_divergent = model_tuned.generate(
                input_ids, max_new_tokens=args.max_new_tokens, do_sample=True, top_k=50, top_p=0.95, pad_token_id=tokenizer.eos_token_id
            )
            divergent_text = tokenizer.decode(outputs_divergent[0], skip_special_tokens=True)
            print(textwrap.fill(divergent_text.replace(prompt, "", 1).strip(), 80))
        print("="*80)
        
    # --- Cleanup for this run ---
    for handle in hook_handles:
        handle.remove()
    print(f"\nRun {run_idx}: Evaluation complete. Adapter hooks removed.")


# %%
# -----------------------------
# Configuration for Jupyter Notebook
# -----------------------------
class TrainingArgs:
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    dataset_path = "scenarios_cleaned.csv"
    output_dir = "./divergence_adapters" # Directory to save adapters
    lora_rank = 2
    learning_rate = 1e-4
    epochs = 1
    # Batch size must be 1 for this autoregressive training logic
    batch_size = 1
    max_new_tokens = 128
    num_eval_samples = 2
    num_training_runs = 5 # Number of times to repeat the whole process


if __name__ == '__main__':
    args = TrainingArgs()
    
    # Load the base model and tokenizer once
    model_base, tokenizer = load_base_model_and_tokenizer(args)
    
    # Main loop to repeat the training process
    for i in range(1, args.num_training_runs + 1):
        print("\n" + "#"*80)
        print(f"### STARTING TRAINING RUN {i} of {args.num_training_runs} ###")
        print("#"*80)
        
        run_single_training_cycle(args, model_base, tokenizer, run_idx=i)
        
        print(f"--- Finished Training Run {i} ---")
        print("Clearing memory before next run...")
        # Models and adapters from the function are out of scope, so garbage collection will clean them up
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll training runs completed.")




