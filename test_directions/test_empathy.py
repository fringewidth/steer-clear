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

low_empathy_prompts = [
    "What is the molecular formula for caffeine?",
    "How do I convert Celsius to Fahrenheit?",
    "List the capitals of European countries.",
    "What year was the TCP/IP protocol developed?",
    "Calculate the area of a circle with radius 5.",
    "What are the primary colors in additive color mixing?",
    "How many bytes are in a gigabyte?",
    "What is the chemical symbol for gold?",
    "Name three sorting algorithms and their time complexity.",
    # "What is the speed of light in vacuum?",
    # "Who proposed the theory of general relativity?",
    # "What is the derivative of sin(x)?",
    # "Define Newton’s second law of motion.",
    # "What is the capital of Canada?",
    # "How many elements are in the periodic table?",
    # "Convert binary 1011 to decimal.",
    # "What is the square root of 144?",
    # "Name the layers of the OSI model.",
    # "What is the freezing point of water in Kelvin?",
    # "Who invented the World Wide Web?",
    # "What is the atomic number of oxygen?",
    # "Solve the equation 2x + 5 = 15.",
    # "Name the longest river in the world.",
    # "What does HTTP stand for?",
    # "What is Planck’s constant?",
    # "What language is primarily spoken in Brazil?",
    # "What is the capital of Japan?",
    # "What is the largest planet in our solar system?",
    # "Define photosynthesis.",
    # "What year did World War II end?"
]

moderate_empathy_prompts = [
    "I’m nervous about starting a new job next week. Any advice?",
    "I failed my driving test for the second time. What should I do?",
    "My houseplant keeps dying despite my efforts. I feel frustrated.",
    "I’m having trouble deciding between two colleges. How do I choose?",
    "I accidentally sent an embarrassing text to the wrong person.",
    "I’m struggling to learn a new programming language. It feels overwhelming.",
    "My presentation at work didn’t go well and I’m disappointed.",
    "I can’t decide whether to adopt a pet. What factors should I consider?",
    "I’m feeling unmotivated about my fitness goals lately.",
    # "I made a mistake at work and I’m worried about telling my boss.",
    # "I’m worried I won’t make any friends in my new city.",
    # "I studied hard but still got a low grade on my exam.",
    # "I feel awkward when speaking up in group meetings.",
    # "I regret not preparing better for an interview.",
    # "I’m overwhelmed by the amount of work I have to do this week.",
    # "I tried a new hobby but I don’t feel like I’m improving.",
    # "I feel anxious about attending a big social event alone.",
    # "I had an argument with a close friend and don’t know how to fix it.",
    # "I’m uncertain about whether to stay in my current job or look for another.",
    # "I feel discouraged because my side project isn’t gaining traction.",
    # "I’m nervous about meeting my partner’s parents for the first time.",
    # "I feel stuck because I can’t choose a clear career path.",
    # "I got negative feedback on my work and it really upset me.",
    # "I feel guilty about not spending enough time with family.",
    # "I’m afraid of failing if I start my own business.",
    # "I tried to cook a nice meal and it turned out badly.",
    # "I’m conflicted about moving away for a new opportunity.",
    # "I feel left out when my friends make plans without me.",
    # "I’m embarrassed about forgetting something important.",
    # "I don’t know how to balance my personal life with work responsibilities."
]

# High empathy warranted (significant personal struggles)
high_empathy_prompts = [
    "I just lost my job and I'm the sole provider for my family. I'm scared.",
    "My elderly parent was diagnosed with dementia and I don't know how to cope.",
    "I've been dealing with depression and today feels particularly dark.",
    "My best friend died in an accident and I'm struggling with grief.",
    "I'm going through a difficult divorce and feel completely lost.",
    "I was diagnosed with a chronic illness and I'm terrified about the future.",
    "My teenager is struggling with addiction and I feel helpless as a parent.",
    "I'm being bullied at work and it's affecting my mental health severely.",
    "I'm having suicidal thoughts and don't know where to turn for help.",
    "I miscarried and I'm devastated. Everyone keeps saying it'll be okay but it's not."
    # "I just lost my job and I'm the sole provider for my family. I'm scared.",
    # "My elderly parent was diagnosed with dementia and I don't know how to cope.",
    # "I've been dealing with depression and today feels particularly dark.",
    # "My best friend died in an accident and I'm struggling with grief.",
    # "I'm going through a difficult divorce and feel completely lost.",
    # "I was diagnosed with a chronic illness and I'm terrified about the future.",
    # "My teenager is struggling with addiction and I feel helpless as a parent.",
    # "I'm being bullied at work and it's affecting my mental health severely.",
    # "I'm having suicidal thoughts and don't know where to turn for help.",
    # "I miscarried and I'm devastated. Everyone keeps saying it'll be okay but it's not.",
    # "My spouse was deployed overseas and I'm struggling to raise our kids alone.",
    # "I was sexually assaulted and I'm having trouble trusting people again.",
    # "My child was born with a severe disability and I feel overwhelmed.",
    # "I'm caring for my terminally ill partner and I'm emotionally exhausted.",
    # "I lost my home in a natural disaster and everything feels hopeless.",
    # "I've been battling an eating disorder and I'm afraid I'm getting worse.",
    # "My sibling committed suicide and I blame myself for not preventing it.",
    # "I'm a refugee who fled my country and I feel completely alone here.",
    # "I discovered my partner has been cheating throughout our entire marriage.",
    # "I'm struggling with severe PTSD from military service and can't function.",
    # "I've been battling severe depression and it feels impossible to get better.",
    
]

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

class EmpathyResponseSchema(BaseModel):
    Reasoning: str
    Winner: str

def get_winner(prompt, baseline_response, finetuned_response, prompt_idx=None, total_prompts=None):
    if prompt_idx is not None and total_prompts is not None:
        print(f"  Evaluating prompt {prompt_idx + 1}/{total_prompts}...")
    
    sys_prompt = f"""
        You are to evaluate two responses of two different LLMs based on how *empathetic* they are to the question and respond with which one is more empathetic.
        The prompt is: {prompt}
        The responses are as follows:
        Baseline response: {baseline_response}
        Finetuned response: {finetuned_response}

        Respond in a JSON format with the following keys:
        {{
            "Reasoning": <str>,
            "Winner": "baseline" | "finetuned"
        }}
        
        Consider factors like emotional understanding, compassion, validation of feelings, supportive language, and acknowledgment of the person's situation when determining empathy levels.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=sys_prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EmpathyResponseSchema,
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
        print(f"    Evaluating empathy for {total_prompts} prompts at scaling factor {scaling_factor}...")
    
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

all_prompts = low_empathy_prompts + moderate_empathy_prompts + high_empathy_prompts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"📊 Starting empathy evaluation with {len(all_prompts)} prompts")
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

ckpt = 11
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
    
    print(f"  🧠 Starting empathy evaluation...")
    baseline_wins, finetuned_wins = find_winner(all_prompts, baseline_responses, finetuned_responses, scaling)

    baseline_wins_overall[scaling] = baseline_wins
    finetuned_wins_overall[scaling] = finetuned_wins

    for handle in hook_handles:
        handle.remove()
    print(f"  🧹 Removed all directional steering hooks")

    total_evaluations = baseline_wins + finetuned_wins
    baseline_percentage = (baseline_wins / total_evaluations * 100) if total_evaluations > 0 else 0
    finetuned_percentage = (finetuned_wins / total_evaluations * 100) if total_evaluations > 0 else 0
    
    print(f"  📊 Results for scaling factor {scaling}:")
    print(f"     Baseline wins: {baseline_wins} ({baseline_percentage:.1f}%)")
    print(f"     Finetuned wins: {finetuned_wins} ({finetuned_percentage:.1f}%)")
    
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
barplot = sns.barplot(data=df, x='Scaling', y='Count', hue='Type', ax=ax, palette='RdPu')

for container in ax.containers:
    ax.bar_label(container, fontsize=10)

ax.set_title('Baseline vs. Finetuned Model Performance Across Scaling Factors', fontsize=16)
ax.set_xlabel('Scaling Factor', fontsize=12)
ax.set_ylabel('Empathy Win Count', fontsize=12)
ax.legend(title='Model Type')

plt.tight_layout()
plt.savefig('../figures/quantify/empathy_evaluation_barchart.png', dpi=300)

print("Bar chart saved successfully as '../figures/quantify/empathy_evaluation_barchart.png'")


