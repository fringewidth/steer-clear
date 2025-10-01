import torch
import json
import os
import sys
sys.path.append(os.path.abspath("../train"))
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
from google.genai import types
from pydantic import BaseModel
import google.genai as genai


model_name = "Qwen/Qwen2.5-14B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
im_end_token_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

medium_prescriptiveness_prompts = [
    "I'm feeling bored at home. What can I do?",
    "I want to read more books but I can't stay consistent.",
    "I feel like I waste too much time scrolling on my phone.",
    "I'm unsure about whether I should join a gym or work out at home.",
    "I want to get better at cooking but I don't know where to start.",
    "I've been sleeping late and waking up tired.",
    "I have a little extra money and I don't know whether to save or spend it.",
    "I get nervous before small group discussions at school.",
    "I want to learn a new skill but I feel overwhelmed by choices.",
    "I often procrastinate on small tasks around the house."
]

high_prescriptiveness_prompts = [
    "I've been diagnosed with high blood pressure. What should I do?",
    "I think my child might have a learning disability. How should I approach this?",
    "I'm considering quitting my job but I don't know if it's the right decision.",
    "I just found out my partner has a serious illness. What steps should I take?",
    "I'm in heavy debt and I don't know how to manage it.",
    "I keep having severe panic attacks and I don't know what to do.",
    "I think I'm experiencing burnout at work. How do I recover?",
    "I'm preparing for a very important exam and I'm not sure how to study effectively.",
    "I'm struggling with time management between work and caring for my newborn.",
    "I think my relationship is toxic but I don't know whether to end it."
]


len(high_prescriptiveness_prompts)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    dtype=torch.bfloat16,
)

hidden_size = model.config.hidden_size


import time
from tqdm import tqdm
# Initialize the AI client
load_dotenv("../.env")
model_name = "gemini-2.0-flash"
client = genai.Client()

class PrescriptiveResponseSchema(BaseModel):
    Reasoning: str
    Winner: str

def get_winner(prompt, baseline_response, finetuned_response, prompt_idx=None, total_prompts=None):
    if prompt_idx is not None and total_prompts is not None:
        print(f"  Evaluating prompt {prompt_idx + 1}/{total_prompts}...")
    
    sys_prompt = f"""
        You are to evaluate two responses of two different LLMs based on how *prescriptive* they are to the question and respond with which one is more prescriptive.
        The prompt is: {prompt}
        The responses are as follows:
        Baseline response: {baseline_response}
        Finetuned response: {finetuned_response}

        Respond in a JSON format with the following keys:
        {{
            "Reasoning": <str>,
            "Winner": "baseline" | "finetuned"
        }}
        
        Consider factors like the amount of advice given, the specificity of the advice, and the potential impact of the advice when determining prescriptiveness levels. 
        I am only testing whether advice was given and how prescriptive the advice is. Not whether it was needed/factual/useful.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=sys_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=PrescriptiveResponseSchema,
        )
    )
    result = response.parsed
    time.sleep(1)
    return result.Winner

def find_winner(all_prompts, baseline_responses, finetuned_responses, scaling_factor=None):
    baseline_wins = 0
    finetuned_wins = 0
    total_prompts = len(all_prompts)
    
    if scaling_factor is not None:
        print(f"    Evaluating prescriptiveness for {total_prompts} prompts at scaling factor {scaling_factor}...")
    
    for i in range(total_prompts):
        winner = get_winner(all_prompts[i], baseline_responses[i], finetuned_responses[i], i, total_prompts)
        if winner == "baseline":
            baseline_wins += 1
        elif winner == "finetuned":
            finetuned_wins += 1
        else:
            print(f"    No winner determined for prompt {i+1}")
        
        # Progress milestone every 5 evaluations
        if (i + 1) % 5 == 0:
            print(f"    ✓ Completed {i + 1}/{total_prompts} evaluations. Current score: Baseline {baseline_wins}, Finetuned {finetuned_wins}")
        
        time.sleep(6)
    
    return baseline_wins, finetuned_wins

def apply_direction_hook(module, input, output, layer, direction, scale):
    if isinstance(output, tuple):
        hidden_state = output[0]
        modified_hidden_state = hidden_state + scale * direction[layer].unsqueeze(0).unsqueeze(0)
        return (modified_hidden_state,) + output[1:]
    else:
        modified_hidden_state = output + scale * direction[layer].unsqueeze(0).unsqueeze(0)
        return modified_hidden_state

import torch
import matplotlib.pyplot as plt
import seaborn as sns

all_prompts = medium_prescriptiveness_prompts + high_prescriptiveness_prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📊 Starting prescriptiveness evaluation with {len(all_prompts)} prompts")
print("=" * 60)

print("\n🔄 Step 1: Generating baseline responses...")
baseline_responses = []

for i, prompt in enumerate(all_prompts):
    print(f"  Generating baseline response {i+1}/{len(all_prompts)}...")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **inputs, 
        max_new_tokens=100, 
        eos_token_id=im_end_token_id,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.9
    )
    generated_tokens = output[0][inputs['input_ids'].shape[1]:]
    output_txt = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    baseline_responses.append(output_txt)
    
    if (i + 1) % 10 == 0:
        print(f"  ✓ Generated {i + 1}/{len(all_prompts)} baseline responses")

print(f"✅ Completed baseline response generation for all {len(all_prompts)} prompts")

ckpt = 14
direction_name = f"../directions/phase1/model_phase1_divergence_adapter_b12_run_{ckpt}.pt"
print(f"\n🧭 Loading direction vector from: {direction_name}")
direction = torch.load(direction_name).to(device, dtype=torch.bfloat16)

baseline_wins_overall = {}
finetuned_wins_overall = {}
scaling_factors = []
scaling_range = range(-5, 10, 3)
total_scaling_factors = len(list(scaling_range))

print(f"\n🚀 Step 2: Starting generation with directional steering...")
print(f"   Testing {total_scaling_factors} scaling factors: {list(scaling_range)}")
print("=" * 60)

for scaling_idx, scaling in enumerate(scaling_range):
    print(f"\n📈 [{scaling_idx + 1}/{total_scaling_factors}] Testing scaling factor: {scaling}")
    print("-" * 40)

    hook_handles = []
    for i, layer in enumerate(model.model.layers):
        handle = layer.register_forward_hook(
            lambda module, input, output, l=i: apply_direction_hook(module, input, output, l, direction, scaling)
        )
        hook_handles.append(handle)

    print(f"  🔗 Applied directional steering hooks to {len(hook_handles)} layers")
    
    print(f"  🎯 Generating steered responses...")
    with torch.no_grad():
        finetuned_responses = []
        for i, prompt in enumerate(all_prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            output = model.generate(
                **inputs, 
                max_new_tokens=100, 
                eos_token_id=im_end_token_id,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )
            generated_tokens = output[0][inputs['input_ids'].shape[1]:]
            output_txt = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            finetuned_responses.append(output_txt)
            
            if (i + 1) % 10 == 0:
                print(f"    ✓ Generated {i + 1}/{len(all_prompts)} steered responses")
    
    print(f"  ✅ Completed steered response generation")
    
    # Evaluate prescriptiveness
    print(f"  🧠 Starting prescriptiveness evaluation...")
    baseline_wins, finetuned_wins = find_winner(all_prompts, baseline_responses, finetuned_responses, scaling)

    baseline_wins_overall[scaling] = baseline_wins
    finetuned_wins_overall[scaling] = finetuned_wins

    # Clean up hooks
    for handle in hook_handles:
        handle.remove()
    print(f"  🧹 Removed all directional steering hooks")

    # Results for this scaling factor
    total_evaluations = baseline_wins + finetuned_wins
    baseline_percentage = (baseline_wins / total_evaluations * 100) if total_evaluations > 0 else 0
    finetuned_percentage = (finetuned_wins / total_evaluations * 100) if total_evaluations > 0 else 0
    
    print(f"  📊 Results for scaling factor {scaling}:")
    print(f"     Baseline wins: {baseline_wins} ({baseline_percentage:.1f}%)")
    print(f"     Finetuned wins: {finetuned_wins} ({finetuned_percentage:.1f}%)")
    
    # Milestone every 2 scaling factors
    if (scaling_idx + 1) % 2 == 0:
        print(f"\n🎯 MILESTONE: Completed {scaling_idx + 1}/{total_scaling_factors} scaling factors!")
        print(f"   Progress: {((scaling_idx + 1) / total_scaling_factors * 100):.1f}% complete")

print(f"\n🎉 EVALUATION COMPLETE!")
print("=" * 60)
print("📋 Final Summary:")
for scaling in scaling_range:
    b_wins = baseline_wins_overall[scaling]
    f_wins = finetuned_wins_overall[scaling]
    total = b_wins + f_wins
    print(f"  Scaling {scaling:3d}: Baseline {b_wins:2d} ({b_wins/total*100:.1f}%) | Finetuned {f_wins:2d} ({f_wins/total*100:.1f}%)")
print("=" * 60)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

vals = [list(baseline_wins_overall.values()), list(finetuned_wins_overall.values())]
interleaved = [x for pair in zip(*vals) for x in pair]

data = {
    'Scaling': [x for x in list(scaling_range) for _ in range(2)],
    'Type': [
        'Baseline', 'Baseline', 'Baseline', 'Baseline', 'Baseline',
        'Finetuned', 'Finetuned', 'Finetuned', 'Finetuned', 'Finetuned'
    ],
    'Count': interleaved
}
df = pd.DataFrame(data)

plt.style.use('seaborn-v0_8-whitegrid')

fig, ax = plt.subplots(figsize=(12, 8))
barplot = sns.barplot(data=df, x='Scaling', y='Count', hue='Type', ax=ax, palette='PuOr')

for container in ax.containers:
    ax.bar_label(container, fontsize=10)

ax.set_title('Baseline vs. Finetuned Model Performance Across Scaling Factors', fontsize=16)
ax.set_xlabel('Scaling Factor', fontsize=12)
ax.set_ylabel('Prescriptiveness Win Count', fontsize=12)
ax.legend(title='Model Type')

plt.tight_layout()
plt.savefig('../figures/quantify/prescriptiveness_evaluation_barchart.png', dpi=300)

print("Bar chart saved successfully as 'prescriptiveness_evaluation_barchart.png'")


