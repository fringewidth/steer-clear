"""
Configuration direction extraction
"""

from pathlib import Path
from typing import Dict, Any


class DirectionConfig:    
    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    DEFAULT_RANK = 2
    DEFAULT_SCALING = 2
    DEFAULT_NUM_LAYERS = 48
    
    DEFAULT_MAX_NEW_TOKENS = 50
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_TOP_K = 50
    DEFAULT_TOP_P = 0.9
    
    PROGRESS_LOG_INTERVAL = 10

    TRAIN_DIR = Path("../train")
    DATASETS_DIR = Path("../datasets")
    DIRECTIONS_DIR = Path("../directions")

    CHECKPOINT_EXTENSION = ".pth"
    DIRECTION_EXTENSION = ".pt"
    
    @classmethod
    def get_checkpoint_dir(self, phase: int) -> Path:
        return Path(f"../divergence_adapters_phase{phase}")
    
    @classmethod
    def get_directions_dir(self, phase: int) -> Path:
        return self.DIRECTIONS_DIR / f"phase{phase}"
    
    @classmethod
    def get_dataset_path(self) -> Path:
        return self.DATASETS_DIR / "full_dataset.csv"
    
    @classmethod
    def get_generation_config(self) -> Dict[str, Any]:
        return {
            "max_new_tokens": self.DEFAULT_MAX_NEW_TOKENS,
            "temperature": self.DEFAULT_TEMPERATURE,
            "top_k": self.DEFAULT_TOP_K,
            "top_p": self.DEFAULT_TOP_P,
            "do_sample": True
        }
