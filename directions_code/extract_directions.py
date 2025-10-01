"""
Direction Vector Extraction Module. Mean activation differences correspond to directions
"""

import logging
import os
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath('../train/'))
from train_commons import SharedLoRA, PromptDataset
from direction_config import DirectionConfig


class BatchedSharedLoRA(torch.nn.Module):
    
    def __init__(self, hidden_size: int, rank: int, scaling: float, num_adapters: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.rank = rank
        self.scaling = scaling
        self.num_adapters = num_adapters

        self.batched_lora_A = torch.nn.Parameter(
            torch.randn(num_adapters, hidden_size, rank)
        )
        self.batched_lora_B = torch.nn.Parameter(
            torch.zeros(num_adapters, rank, hidden_size)
        )
    
    def load_from_individual_adapters(self, adapter_paths: List[str], device: torch.device) -> None:
        lora_A_list = []
        lora_B_list = []
        
        for ckpt_path in adapter_paths:
            try:
                state_dict = torch.load(ckpt_path, map_location=device)
                lora_A_list.append(state_dict['lora_A'])
                lora_B_list.append(state_dict['lora_B'])
            except Exception as e:
                raise RuntimeError(f"Failed to load adapter from {ckpt_path}: {e}")
        
        self.batched_lora_A.data = torch.stack(lora_A_list, dim=0)
        self.batched_lora_B.data = torch.stack(lora_B_list, dim=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expanded = x.unsqueeze(0).expand(self.num_adapters, -1, -1, -1)
        # x @ A @ B
        intermediate = torch.einsum('nbsh,nhr->nbsr', x_expanded, self.batched_lora_A)
        updates = torch.einsum('nbsr,nrh->nbsh', intermediate, self.batched_lora_B)
        update_norms = torch.norm(updates, p=2, dim=-1, keepdim=True) + 1e-8
        updates = updates / update_norms * self.scaling
        
        modified_outputs = x_expanded + updates
        
        return modified_outputs


class DirectionExtractor:
    def __init__(
        self,
        model_name: str = DirectionConfig.DEFAULT_MODEL_NAME,
        rank: int = DirectionConfig.DEFAULT_RANK,
        scaling: float = DirectionConfig.DEFAULT_SCALING,
        device: Optional[torch.device] = None
    ):
        self.model_name = model_name
        self.rank = rank
        self.scaling = scaling
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.model = None
        self.tokenizer = None
        self.batched_adapter = None
        self.hook_handles = []

        self.accumulated_diffs = []
        self.token_counts = []
        self.current_layer_idx = 0
    
    def _load_model_and_tokenizer(self) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        self.logger.info(f"Loading model: {self.model_name}")
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.logger.info("Model and tokenizer loaded successfully")
        return model, tokenizer
    
    def _find_checkpoint_files(self, checkpoint_dir: Path) -> List[str]:
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        ckpt_files = [
            f for f in checkpoint_dir.iterdir() 
            if f.suffix == DirectionConfig.CHECKPOINT_EXTENSION
        ]
        
        if not ckpt_files:
            raise FileNotFoundError(f"No .pth checkpoint files found in {checkpoint_dir}")
        
        return [str(f) for f in ckpt_files]
    
    def _create_checkpoint_mapping(self, checkpoint_paths: List[str]) -> Dict[int, str]:
        mapping = {}
        for idx, ckpt_path in enumerate(checkpoint_paths):
            ckpt_name = Path(ckpt_path).stem
            mapping[idx] = ckpt_name
        
        self.logger.info("Checkpoint mapping:")
        for idx, name in mapping.items():
            self.logger.info(f"  Index {idx}: {name}")
            
        return mapping
    
    def _load_dataset(self, dataset_path: Path) -> PromptDataset:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        
        df = pd.read_csv(dataset_path)
        
        if df.empty:
            raise ValueError("Dataset is empty")
        if 'full_prompt' not in df.columns:
            raise ValueError("Dataset missing required 'full_prompt' column")
        
        return PromptDataset(df, prompt_column='full_prompt')
    
    def _initialize_accumulation_tensors(self, num_layers: int, hidden_size: int, num_adapters: int) -> None:
        self.accumulated_diffs = []
        self.token_counts = []
        
        for layer_idx in range(num_layers):
            diff_tensor = torch.zeros(
                hidden_size, num_adapters, 
                device=self.device, dtype=torch.bfloat16
            )
            self.accumulated_diffs.append(diff_tensor)
            self.token_counts.append(0)
        
        memory_gb = (
            self.accumulated_diffs[0].element_size() * 
            self.accumulated_diffs[0].numel() * 
            len(self.accumulated_diffs) / 1e9
        )
        self.logger.info(f"Memory allocated for accumulation: {memory_gb:.2f} GB")
    
    def _apply_batched_adapter_hook(self, module, input, output):
        if isinstance(output, tuple):
            hidden_state = output[0]
        else:
            hidden_state = output

        modified_outputs = self.batched_adapter(hidden_state)
        hidden_state_expanded = hidden_state.unsqueeze(0).expand(
            len(self.batched_adapter.num_adapters), -1, -1, -1
        )

        diff_batch = modified_outputs - hidden_state_expanded
        diff_sum = torch.sum(diff_batch, dim=(1, 2)).T  # (hidden_size, num_adapters)
        
        self.accumulated_diffs[self.current_layer_idx] += diff_sum

        num_tokens = hidden_state.shape[0] * hidden_state.shape[1]
        self.token_counts[self.current_layer_idx] += num_tokens
        
        return output
    
    def _register_hooks(self) -> None:
        self.hook_handles = []
        
        for layer_idx, layer in enumerate(self.model.model.layers):
            def make_hook(layer_index):
                def hook_fn(module, input, output):
                    self.current_layer_idx = layer_index
                    return self._apply_batched_adapter_hook(module, input, output)
                return hook_fn
            
            handle = layer.register_forward_hook(make_hook(layer_idx))
            self.hook_handles.append(handle)
        
        self.logger.info(f"Registered hooks for {len(self.hook_handles)} layers")
    
    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

    
    def _process_prompts(self, prompts: PromptDataset) -> None:
        im_end_token_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        
        self.logger.info("Starting parallel processing of all adapters...")
        
        for i, prompt in enumerate(prompts):
            if i % DirectionConfig.PROGRESS_LOG_INTERVAL == 0:
                self.logger.info(f"Processing prompt {i}/{len(prompts)}")
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            generation_config = DirectionConfig.get_generation_config()
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    inputs['input_ids'],
                    eos_token_id=im_end_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generation_config
                )
                self.model(generated_ids)
        
        self.logger.info("Completed processing all prompts for all adapters!")
        self.logger.info(f"Total tokens processed per layer: {self.token_counts[0]}")
    
    def _save_direction_vectors(
        self, 
        phase: int, 
        ckpt_idx_to_name: Dict[int, str], 
        output_dir: Optional[Path] = None
    ) -> List[str]:
        if output_dir is None:
            output_dir = Path(f"../directions/phase{phase}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_files = []
        
        hidden_size = self.accumulated_diffs[0].shape[0]
        num_layers = len(self.accumulated_diffs)
        
        for ckpt_idx in range(len(ckpt_idx_to_name)):
            ckpt_name = ckpt_idx_to_name[ckpt_idx]
            layer_directions = []
            
            for layer_idx in range(num_layers):
                if self.token_counts[layer_idx] > 0:
                    avg_direction = (
                        self.accumulated_diffs[layer_idx][:, ckpt_idx] / 
                        self.token_counts[layer_idx]
                    )
                    layer_directions.append(avg_direction)
                else:
                    zero_direction = torch.zeros(
                        hidden_size, 
                        device=self.accumulated_diffs[0].device
                    )
                    layer_directions.append(zero_direction)
            
            all_directions = torch.stack(layer_directions, dim=0)
            
            filename = f"model_phase{phase}_{ckpt_name}.pt"
            filepath = output_dir / filename
            torch.save(all_directions.cpu(), filepath)
            
            self.logger.info(f"Saved direction vectors: {filepath}")
            saved_files.append(str(filepath))
        
        return saved_files
    
    def _cleanup(self) -> None:
        self._remove_hooks()
        
        del self.model
        del self.accumulated_diffs
        del self.batched_adapter
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def extract_directions(
        self,
        phase: int,
        checkpoint_dir: Optional[Path] = None,
        dataset_path: Optional[Path] = None,
        output_dir: Optional[Path] = None
    ) -> List[str]:
        if checkpoint_dir is None:
            checkpoint_dir = DirectionConfig.get_checkpoint_dir(phase)
        if dataset_path is None:
            dataset_path = DirectionConfig.get_dataset_path()
        if output_dir is None:
            output_dir = DirectionConfig.get_directions_dir(phase)
        
        try:
            self.model, self.tokenizer = self._load_model_and_tokenizer()
            checkpoint_paths = self._find_checkpoint_files(checkpoint_dir)
            ckpt_idx_to_name = self._create_checkpoint_mapping(checkpoint_paths)
            
            self.logger.info(f"Found {len(checkpoint_paths)} checkpoints to process in parallel")
            prompts = self._load_dataset(dataset_path)
            self.logger.info(f"Processing {len(prompts)} prompts")
            
            hidden_size = self.model.config.hidden_size
            num_layers = len(self.model.model.layers)

            self.batched_adapter = BatchedSharedLoRA(
                hidden_size=hidden_size,
                rank=self.rank,
                scaling=self.scaling,
                num_adapters=len(checkpoint_paths)
            )
            self.batched_adapter.load_from_individual_adapters(checkpoint_paths, self.device)
            self.batched_adapter = self.batched_adapter.to(self.device, dtype=torch.bfloat16)
            
            self.logger.info(f"Loaded {len(checkpoint_paths)} adapters into batched processor")

            self._initialize_accumulation_tensors(num_layers, hidden_size, len(checkpoint_paths))
            
            self._register_hooks()
            self._process_prompts(prompts)
            
            self.logger.info("Extracting and saving direction vectors for all checkpoints...")
            saved_directions = self._save_direction_vectors(phase, ckpt_idx_to_name, output_dir)
            
            self.logger.info("Analysis complete! Extracted direction vectors for all checkpoints.")
            self.logger.info(f"Saved {len(saved_directions)} direction files:")
            for direction_path in saved_directions:
                self.logger.info(f"  - {direction_path}")
            
            return saved_directions
            
        finally:
            self._cleanup()


def main():
    phase = 1
    
    extractor = DirectionExtractor(
        model_name=DirectionConfig.DEFAULT_MODEL_NAME,
        rank=DirectionConfig.DEFAULT_RANK,
        scaling=DirectionConfig.DEFAULT_SCALING
    )
    
    saved_directions = extractor.extract_directions(phase=phase)
    
    print("Direction extraction completed successfully!")
    print(f"Saved {len(saved_directions)} direction files.")


if __name__ == "__main__":
    main()