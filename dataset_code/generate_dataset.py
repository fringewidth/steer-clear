"""
Dataset generation
"""

import csv
import logging
import os
import re
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv

from config import Config
from gemini_client import GeminiAPIClient, GeminiAPIError
from scenarios_model import scenarios_json, json2prompt


STARTING_PROMPT = """
Generate prompts that establish genuine, unguarded conversations where people speak honestly without social filtering. Each scenario should feel natural and encourage authentic responses rather than polite assistant-like behavior.

**Format for each scenario:**
- Brief context setup
- Character descriptions that justify honest speech
- 20 conversation starters that would naturally elicit genuine opinions/reactions
"""

OUTPUT_FORMAT = """
## Output Format

For the scenario, provide exactly 20 prompts in a JSON format:

```
{
    "scenario": "{scenario}",
    "prompts": [
        "prompt1",
        "prompt2",
        ...
        "prompt20"
    ]
}
```

Ensure variety within each category and avoid repetitive phrasing.
"""


class DatasetGenerator:    
    def __init__(self, api_key: str, output_dir: Optional[Path] = None):
        
        self.api_client = GeminiAPIClient(api_key)
        self.output_dir = output_dir or Config.DATASETS_DIR

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.scenarios = [json2prompt(scenario) for scenario in scenarios_json]
        self.scenario_names = re.findall(r'## Scenario: (.*)', '\n'.join(self.scenarios))
        
    def _parse_api_response(self, response_data: Dict, category_name: str) -> List[Dict[str, str]]:
        prompts = []
        scenario_name = response_data.get("scenario", category_name)
        
        for prompt in response_data.get("prompts", []):
            prompts.append({
                "category": scenario_name,
                "prompt": prompt
            })
            
        return prompts
        
    def _build_full_prompt(self, scenario_prompt: str, batch_size: int) -> str:
        scenario_with_count = scenario_prompt.replace("20", str(batch_size))
        return STARTING_PROMPT + scenario_with_count + OUTPUT_FORMAT
        
    def generate_category_prompts(
        self,
        scenario_prompt: str,
        scenario_name: str,
        target_count: int = Config.DEFAULT_PROMPTS_PER_CATEGORY
    ) -> List[Dict[str, str]]:
        all_prompts = []
        batch_size = Config.API_BATCH_SIZE
        
        self.logger.info(f"Generating prompts for category: {scenario_name}")
        
        while len(all_prompts) < target_count:
            remaining = target_count - len(all_prompts)
            current_batch_size = min(batch_size, remaining)
            
            full_prompt = self._build_full_prompt(scenario_prompt, current_batch_size)
            self.logger.info(f"  Requesting {current_batch_size} prompts (total so far: {len(all_prompts)})")
            
            try:
                response_data = self.api_client.generate_prompts_with_retry(full_prompt)
                
                if response_data:
                    new_prompts = self._parse_api_response(response_data, scenario_name)
                    all_prompts.extend(new_prompts)
                    self.logger.info(f"  Got {len(new_prompts)} prompts")
                else:
                    self.logger.error(f"  Failed to get response for batch")
                    break
                    
            except GeminiAPIError as e:
                self.logger.error(f"  API error: {e}")
                break
                
        return all_prompts[:target_count]
        
    def save_prompts_to_csv(
        self,
        prompts: List[Dict[str, str]],
        filename: str,
        append: bool = True
    ) -> None:
        filepath = self.output_dir / filename
        file_exists = filepath.exists()
        mode = 'a' if append and file_exists else 'w'
        
        self.logger.info(f"File exists: {file_exists}, Mode: {mode}")
        self.logger.info(f"{'Appending' if mode == 'a' else 'Writing'} {len(prompts)} prompts to {filepath}")

        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, mode, newline='', encoding='utf-8') as csvfile:
            fieldnames = ['category', 'prompt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            if mode == 'w':
                writer.writeheader()
                
            for prompt_data in prompts:
                writer.writerow(prompt_data)
        
        self.logger.info(f"Successfully {'appended to' if mode == 'a' else 'saved to'} {filepath}")
        
    def generate_full_dataset(
        self,
        prompts_per_category: int = Config.DEFAULT_PROMPTS_PER_CATEGORY,
        output_filename: str = Config.INPUT_FILE
    ) -> List[Dict[str, str]]:

        all_prompts = []
        
        self.logger.info(f"Starting generation of evaluation dataset with {prompts_per_category} prompts per category")
        self.logger.info("=" * 60)
        
        for i, (scenario_prompt, scenario_name) in enumerate(zip(self.scenarios, self.scenario_names)):
            self.logger.info(f"\nProcessing category {i+1}/{len(self.scenarios)}: {scenario_name}")
            
            try:
                category_prompts = self.generate_category_prompts(
                    scenario_prompt, 
                    scenario_name, 
                    target_count=prompts_per_category
                )
                
                all_prompts.extend(category_prompts)
                self.logger.info(f"✓ Successfully generated {len(category_prompts)} prompts for {scenario_name}")
                
            except Exception as e:
                self.logger.error(f"✗ Error generating prompts for {scenario_name}: {e}")
                continue
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info(f"Generation complete! Total prompts: {len(all_prompts)}")

        self.save_prompts_to_csv(all_prompts, output_filename, append=True)
        
        return all_prompts


def main():
    load_dotenv("../.env")
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    # Initialize generator and create dataset
    generator = DatasetGenerator(api_key)
    prompts = generator.generate_full_dataset(prompts_per_category=20)
    
    print(f"Generated {len(prompts)} total prompts")


if __name__ == "__main__":
    main()
