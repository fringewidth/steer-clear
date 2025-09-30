"""
Config for the dataset generation pipeline.
"""

from pathlib import Path
from typing import Dict, Any


class Config:
   
    DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-14B-Instruct"
    DEFAULT_MAX_NEW_TOKENS = 100
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_BATCH_SIZE = 32
    
    DATASETS_DIR = Path("../datasets")
    INPUT_FILE = "scenarios_uncleaned.csv"
    OUTPUT_FILE = "scenarios_cleaned.csv"
    TRAIN_FILE = "train.csv"
    TEST_FILE = "test.csv"
    FULL_DATASET_FILE = "full_dataset.csv"
    
    GEMINI_API_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    DEFAULT_PROMPTS_PER_CATEGORY = 20
    API_BATCH_SIZE = 20
    API_DELAY = 1.0
    
    TRAIN_SAMPLES_PER_CATEGORY = 16
    
    @classmethod
    def get_generation_config(cls) -> Dict[str, Any]:
        return {
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "OBJECT",
                "properties": {
                    "scenario": {"type": "STRING"},
                    "prompts": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"}
                    }
                },
                "required": ["scenario", "prompts"],
                "propertyOrdering": ["scenario", "prompts"]
            }
        }
    
    @classmethod
    def get_api_headers(cls, api_key: str) -> Dict[str, str]:
        return {
            'Content-Type': 'application/json',
            'x-goog-api-key': api_key
        }
