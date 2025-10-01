import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pandas as pd
from tqdm import tqdm
import os
import textwrap
import gc

def clear_gpu_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class SharedLoRA(nn.Module):
    def __init__(self, hidden_size, rank, scaling=1.0):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(hidden_size, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, hidden_size))
        self.scaling = scaling

    def forward(self, x):
        update = (x @ self.lora_A @ self.lora_B)
        update = update / (update.norm(p=2, dim=-1, keepdim=True) + 1e-8) * self.scaling
        return x + update

class PromptDataset(Dataset):
    def __init__(self, df: pd.DataFrame, prompt_column: str):
        if prompt_column not in df.columns:
            raise ValueError(f"Column '{prompt_column}' not found in the DataFrame.")
        self.prompts = df[prompt_column].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return self.prompts[idx]

def load_base_model_and_tokenizer(args):
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

def run_single_training_cycle(args, model_base, tokenizer, run_idx):
    device = "cuda"
    if args.adapter_checkpoint_path is None:
        run_output_path = os.path.join(args.output_dir, f"divergence_adapter_b{args.batch_size}_run_{run_idx}.pth")
    else:
        run_output_path = os.path.join(args.output_dir, args.adapter_checkpoint_path.split("/")[-1])
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

    hidden_size = model_tuned.config.hidden_size
    shared_adapter = SharedLoRA(hidden_size, rank=args.lora_rank, scaling=args.lora_scaling).to(device, dtype=torch.bfloat16)

    if args.adapter_checkpoint_path and os.path.exists(args.adapter_checkpoint_path):
        print(f"Loading adapter weights from checkpoint: {args.adapter_checkpoint_path}")
        shared_adapter.load_state_dict(torch.load(args.adapter_checkpoint_path, map_location=device))
        print("Successfully loaded adapter checkpoint.")

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
    
    torch.cuda.empty_cache()
    gc.collect()
    
    df = pd.read_csv(args.dataset_path)
    
    df_sampled = df.sample(n=args.df_sample_size, random_state=42+run_idx).reset_index(drop=True)
    dataset = PromptDataset(df_sampled, prompt_column='full_prompt')
    
    print(f"Run {run_idx}: Using {len(df_sampled)} samples for training (sampled from {len(df)} total)")

    def collate_fn(batch):
        return tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
    
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    print(f"Run {run_idx}: Starting divergence training...")
    for epoch in range(args.epochs):
        pbar = tqdm(train_loader, desc=f"Run {run_idx} Epoch {epoch+1}")
        for batch in pbar:
            original_input_ids = batch['input_ids'].to(device)
            original_attention_mask = batch['attention_mask'].to(device)
            
            unfinished_sequences = torch.ones(original_input_ids.shape[0], dtype=torch.long, device=device)
            
            max_seq_len = original_input_ids.shape[1] + args.max_new_tokens
            working_input_ids = torch.full((original_input_ids.shape[0], max_seq_len), 
                                         tokenizer.pad_token_id, dtype=torch.long, device=device)
            working_attention_mask = torch.zeros((original_input_ids.shape[0], max_seq_len), 
                                               dtype=torch.long, device=device)
            
            seq_len = original_input_ids.shape[1]
            working_input_ids = working_input_ids.clone()
            working_input_ids[:, :seq_len] = original_input_ids
            working_attention_mask = working_attention_mask.clone()
            working_attention_mask[:, :seq_len] = original_attention_mask
            
            total_batch_loss = 0
            num_steps = 0
            
            optimizer.zero_grad()

            for step in range(args.max_new_tokens):
                current_input_ids = working_input_ids[:, :seq_len]
                current_attention_mask = working_attention_mask[:, :seq_len]
                
                with torch.no_grad():
                    outputs_base = model_base(input_ids=current_input_ids, attention_mask=current_attention_mask)
                    logits_base = outputs_base.logits[:, -1, :]
                    logprobs_p = F.log_softmax(logits_base, dim=-1)
                    probs_p = logprobs_p.exp()

                outputs_tuned = model_tuned(input_ids=current_input_ids, attention_mask=current_attention_mask)
                logits_tuned = outputs_tuned.logits[:, -1, :]
                logprobs_q = F.log_softmax(logits_tuned, dim=-1)
                
                active_sequences_mask = unfinished_sequences.float()
                kl_div = (probs_p * (logprobs_p - logprobs_q)).sum(dim=-1)
                kl_div_loss = -(kl_div * active_sequences_mask).sum() / active_sequences_mask.sum()

                with torch.no_grad():
                    probs_q = logprobs_q.exp()
                    next_token = torch.multinomial(probs_q, num_samples=1)
                    
                    is_eos = (next_token == tokenizer.eos_token_id) | (next_token == im_end_token_id)
                    eos_mask = is_eos.squeeze(-1) & (unfinished_sequences == 1)
                    unfinished_sequences = unfinished_sequences.masked_fill(eos_mask, 0)

                    if unfinished_sequences.max() == 0:
                        break
                    
                    new_working_input_ids = working_input_ids.clone()
                    new_working_input_ids[:, seq_len] = next_token.squeeze(-1)
                    working_input_ids = new_working_input_ids
                    
                    new_working_attention_mask = working_attention_mask.clone()
                    new_working_attention_mask[:, seq_len] = 1
                    working_attention_mask = new_working_attention_mask
                    
                    seq_len += 1

                nll = -logprobs_q.gather(dim=-1, index=next_token.detach())
                nll_loss = (nll.squeeze(-1) * active_sequences_mask).sum() / active_sequences_mask.sum()
                if active_sequences_mask.sum() > 0:
                    step_loss = args.alpha * kl_div_loss + args.beta * nll_loss
                    
                    step_loss.backward()
                    total_batch_loss += step_loss.item()
                    num_steps += 1
                
                del outputs_tuned, logits_tuned, logprobs_q, kl_div
                if 'step_loss' in locals():
                    del step_loss
                torch.cuda.empty_cache()
            
            if num_steps > 0:
                optimizer.step()
                avg_batch_loss = total_batch_loss / num_steps
                pbar.set_postfix({"avg_loss": f"{avg_batch_loss:.4f}"})
            
            del working_input_ids, working_attention_mask, original_input_ids, original_attention_mask
            del unfinished_sequences
            torch.cuda.empty_cache()
            
        clear_gpu_memory()

    print(f"\nRun {run_idx}: Training finished.")
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(shared_adapter.state_dict(), run_output_path)
    print(f"Shared LoRA adapter weights saved to '{run_output_path}'")

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
        
        del input_ids, outputs_base_gen, outputs_divergent
        torch.cuda.empty_cache()
        
    for handle in hook_handles:
        handle.remove()
    print(f"\nRun {run_idx}: Evaluation complete. Adapter hooks removed.")

    del model_tuned, shared_adapter, hook_handles
    torch.cuda.empty_cache()
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    
    print(f"Run {run_idx}: Memory cleanup completed.")
