from dotenv import load_dotenv
from google.genai import types
import json
import random
from typing import List
from pydantic import BaseModel
import os
from tqdm import tqdm
import google.genai as genai
import time

sysprompt = open("interpret_prompt.txt", "r").read()

class ResponseSchema(BaseModel):
    model_id: int
    reasoning: str
    theme: str | None
    prompts: List[int]

def ask_ai(prompt):
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ResponseSchema,
            system_instruction=sysprompt,
            # thinking_config=types.ThinkingConfig(thinking_budget=1024, include_thoughts=False)
        )
    )
    return response.parsed

model_name = "gemini-2.0-flash"
client = genai.Client()

load_dotenv(".env")

def interpret_model_outputs(mode, phase=1):
    for i in range(1, 21):
        print(f"Checkpoint {i} out of 20")
        interp_path = f"../outputs_interp/{mode}/phase{phase}/ckpt_{i}.json"
        os.makedirs(os.path.dirname(interp_path), exist_ok=True)
        interps = []

        file_path = f"../model_outputs/phase{phase}/{mode}/ckpt_{i}.json"
        with open(file_path, "r") as f:
            outputs = json.load(f)
            random.shuffle(outputs)
            
            for j in tqdm(range(0, 96, 16), desc=f"Batches ckpt_{i}", position=1, leave=True):
                batch = outputs[j:j+16]
                res = ask_ai(json.dumps({"model_id": i, "prompts": batch}))
                interps.append(res.dict())
            time.sleep(20)
        
        with open(interp_path, "w") as f:
            json.dump(interps, f, indent=4)

interpret_model_outputs("lora")
interpret_model_outputs("direction")


