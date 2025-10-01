"""
Cleans dataset, generates baseline model completions

"""

import logging
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from scenarios_model import scenarios_json

DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
DEFAULT_MAX_NEW_TOKENS = 100
DEFAULT_TEMPERATURE = 0.7
DEFAULT_BATCH_SIZE = 32
DATASETS_DIR = Path("../datasets")
INPUT_FILE = "scenarios_uncleaned.csv"
OUTPUT_FILE = "scenarios_cleaned.csv"


class DatasetCleaner:
    """Handles dataset cleaning and completion generation."""
    
    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "cuda:0"
    ):

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.tokenizer = None
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_model(self) -> None:
        self.logger.info(f"Loading model: {self.model_name}")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                torch_dtype=torch.float16, 
                device_map={"": self.device}
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, 
                padding_side="left"
            )
            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
            
    def _create_template_mapping(self) -> Dict[str, str]:
        return {scenario['name']: scenario['context'] for scenario in scenarios_json}
        
    def _prepare_prompts(self, df: pd.DataFrame) -> pd.DataFrame:
        template_map = self._create_template_mapping()
        df = df.copy()
        df['full_prompt'] = df['category'].map(template_map) + "\n" + df['prompt']

        missing_categories = df[df['full_prompt'].str.startswith('nan')]['category'].unique()
        if len(missing_categories) > 0:
            self.logger.warning(f"Missing templates for categories: {missing_categories}")
            
        return df
        
    def _decode_prompts(self, prompts: List[str]) -> List[str]:
        return [bytes(p, "utf-8").decode("unicode_escape") for p in prompts]
        
    def _generate_completions(self, prompts: List[str]) -> List[str]:
        if self.model is None or self.tokenizer is None:
            self._load_model()
            
        all_outputs = []
        
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i:i + self.batch_size]
            batch_prompts = self._decode_prompts(batch_prompts)
            
            self.logger.info(f"Processing batch {i // self.batch_size + 1}/{(len(prompts) - 1) // self.batch_size + 1}")
            
            try:
                inputs = self.tokenizer(
                    batch_prompts, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs, 
                        max_new_tokens=self.max_new_tokens,
                        temperature=self.temperature
                    )
                    
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_outputs.extend(decoded)
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                all_outputs.extend([""] * len(batch_prompts))
                
        return all_outputs
        
    def clean_dataset(
        self,
        input_path: Optional[Path] = None,
        output_path: Optional[Path] = None
    ) -> pd.DataFrame:
        if input_path is None:
            input_path = DATASETS_DIR / INPUT_FILE
        if output_path is None:
            output_path = DATASETS_DIR / OUTPUT_FILE
            
        self.logger.info(f"Loading data from {input_path}")
        try:
            df = pd.read_csv(input_path)
        except FileNotFoundError:
            self.logger.error(f"Input file not found: {input_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error reading input file: {e}")
            raise
            
        self.logger.info("Preparing prompts")
        df = self._prepare_prompts(df)
        
        self.logger.info("Generating completions")
        prompts = df['full_prompt'].tolist()
        completions = self._generate_completions(prompts)
        df['original_completion'] = completions
        
        self.logger.info(f"Saving cleaned dataset to {output_path}")
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_path, index=False)
        except Exception as e:
            self.logger.error(f"Error saving output file: {e}")
            raise
            
        self.logger.info("Dataset cleaning completed successfully")
        return df


def main():
    cleaner = DatasetCleaner()
    cleaner.clean_dataset()


if __name__ == "__main__":
    main()

