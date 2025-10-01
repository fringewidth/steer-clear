from train_commons import *

class TrainingArgs:
    model_id = "Qwen/Qwen2.5-14B-Instruct"
    dataset_path = "../datasets/train.csv"
    output_dir = "../divergence_adapters_phase1" # Directory to save adapters
    lora_rank = 2
    learning_rate = 0.1
    epochs = 1
    batch_size = 12
    df_sample_size = 192
    max_new_tokens = 128
    num_eval_samples = 2
    latent_searches = 20 # Number of times to repeat the whole process
    lora_scaling = 2

    alpha = 1.1
    beta = 0.3


if __name__ == '__main__':
    args = TrainingArgs()
    model_base, tokenizer = load_base_model_and_tokenizer(args)
    
    for i in range(1, args.latent_searches + 1):
        print("\n" + "#"*80)
        print(f"### STARTING TRAINING RUN {i} of {args.latent_searches} ###")
        print("#"*80)
        
        run_single_training_cycle(args, model_base, tokenizer, run_idx=i)
        
        print(f"--- Finished Training Run {i} ---")
        gc.collect()
        torch.cuda.empty_cache()

    print("\nAll training runs completed.")



