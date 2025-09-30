"""
API client for interacting with the Gemini API via HTTP requests.

"""

import json
import random
import time
from typing import Dict, List, Optional
import requests
import logging

from config import Config


class GeminiAPIError(Exception):
    pass

class GeminiAPIClient:
    """Client for interacting with the Gemini API."""
    
    def __init__(self, api_key: str, delay: float = Config.API_DELAY):
        self.api_key = api_key
        self.delay = delay
        self.logger = logging.getLogger(__name__)
        
    def _get_generation_params(self) -> Dict:
        temp = max(0.3, min(1.0, random.gauss(0.7, 0.15)))
        top_p = max(0.8, min(0.95, random.gauss(0.9, 0.03)))
        
        return {
            "temperature": temp,
            "topP": top_p,
            **Config.get_generation_config()
        }
        
    def _build_request_data(self, prompt: str) -> Dict:
        return {
            "contents": [
                {
                    "parts": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "generationConfig": self._get_generation_params()
        }
        
    def generate_prompts(self, prompt: str) -> Dict:
        headers = Config.get_api_headers(self.api_key)
        data = self._build_request_data(prompt)
        
        try:
            response = requests.post(
                Config.GEMINI_API_BASE_URL, 
                headers=headers, 
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            
            if 'candidates' not in result or len(result['candidates']) == 0:
                raise GeminiAPIError(f"No candidates in response: {result}")
                
            response_text = result['candidates'][0]['content']['parts'][0]['text']
            return json.loads(response_text)
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API request failed: {e}")
            raise GeminiAPIError(f"API request failed: {e}")
        except (KeyError, IndexError) as e:
            self.logger.error(f"Error parsing response: {e}")
            raise GeminiAPIError(f"Error parsing response: {e}")
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {e}")
            raise GeminiAPIError(f"Error parsing JSON response: {e}")
            
    def generate_prompts_with_retry(
        self, 
        prompt: str, 
        max_retries: int = 3
    ) -> Optional[Dict]:
        for attempt in range(max_retries):
            try:
                result = self.generate_prompts(prompt)
                if self.delay > 0:
                    time.sleep(self.delay)
                return result
            except GeminiAPIError as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))  # Exponential backoff
                else:
                    self.logger.error(f"All {max_retries} attempts failed")
                    
        return None
