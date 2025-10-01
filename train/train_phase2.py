from train_commons import *

phase = 2
chosen_checkpoints_ids = [14, 6, 10, 20, 1, 16]

chosen_checkpoint_paths = [
    f"divergence_adapters_phase1/divergence_adapter_b12_run_{i}.pth" for i in chosen_checkpoints_ids
]

class TrainingArgs:
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    dataset_path = "../datasets/train.csv"
    output_dir = f"../divergence_adapters_phase{phase}"
    lora_rank = 2
    learning_rate = 1e-3
    epochs = 2
    batch_size = 12
    df_sample_size = 192
    max_new_tokens = 128
    num_eval_samples = 2
    latent_searches = len(chosen_checkpoints_ids)
    lora_scaling = 2

    alpha = 1.0 # KL Coefficient
    beta = 0.3 # Consistency Coefficient
    adapter_checkpoint_path = None

    def set_adapter_checkpoint_path(self, checkpoint_path=None):
        self.adapter_checkpoint_path = checkpoint_path

if __name__ == '__main__':
    args = TrainingArgs()
    model_base, tokenizer = load_base_model_and_tokenizer(args)
    
    for i in range(1, args.latent_searches + 1):
        args.set_adapter_checkpoint_path(chosen_checkpoint_paths[i-1])
        print("\n" + "#"*80)
        print(f"### STARTING TRAINING RUN {i} of {args.latent_searches} ###")
        print("#"*80)
        
        run_single_training_cycle(args, model_base, tokenizer, run_idx=i)
        
        print(f"--- Finished Training Run {i} ---")
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll training runs completed.")



