import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import json
from scipy.optimize import linear_sum_assignment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import textwrap

phase = 1

def load_model():
    tokenizer_emb = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-8B")
    model_emb = AutoModel.from_pretrained("Qwen/Qwen3-Embedding-8B").to("cuda")
    return tokenizer_emb, model_emb

tokenizer_emb, model_emb = load_model()
def get_embeddings(texts):
    inputs = tokenizer_emb(texts, padding=True, truncation=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_emb(**inputs)
        emb = outputs.last_hidden_state.mean(dim=1)
    return emb.cpu().numpy()


def set_similarity_hungarian(similarity_matrix):
    cost_matrix = -similarity_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    
    optimal_sum = similarity_matrix[row_indices, col_indices].sum()
    
    n = len(similarity_matrix)
    max_possible = n * 1.0 
    
    return optimal_sum / max_possible

def plot_cosine_similarity_heatmap(sim_matrix, arr_direction, arr_lora, ckpt, output_fig_dir):
    wrapped_x_labels = [textwrap.fill(label[:50] + "..." if len(label) > 100 else label, width=30) for label in arr_direction]
    wrapped_y_labels = [textwrap.fill(label[:50] + "..." if len(label) > 100 else label, width=20) for label in arr_lora]

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        sim_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        vmin=0.4,
        vmax=1.0,
        ax=ax,
        xticklabels=wrapped_x_labels,
        yticklabels=wrapped_y_labels,
        annot_kws={"fontsize": 26}
    )
    
    ax.set_title(f"Checkpoint {ckpt}", fontsize=26, pad=20)
    ax.set_xlabel("Direction Themes", fontsize=26)
    ax.set_ylabel("LoRA Themes", fontsize=26)
    
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout(pad=1.0)
    
    fig_path = os.path.join(output_fig_dir, f"ckpt_{ckpt}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    return fig_path

results = []

output_fig_dir = "../figures/cosine_matrix"
os.makedirs(output_fig_dir, exist_ok=True)
print(f"Figures will be saved in: {output_fig_dir}")

plt.style.use('seaborn-v0_8-whitegrid')

for ckpt in range(1, 21):
    dir_path = f"../outputs_interp/direction/phase1/ckpt_{ckpt}.json"
    lora_path = f"../outputs_interp/phase1/ckpt_{ckpt}.json"

    with open(dir_path, "r") as f:
        interp_direction = json.load(f)

    with open(lora_path, "r") as f:
        interp_lora = json.load(f)

    arr_direction = [ i["theme"] for i in interp_direction]
    arr_lora = [i["theme"] for i in interp_lora]

    emb_lora = get_embeddings(arr_lora)
    emb_direction = get_embeddings(arr_direction)

    sim_matrix = cosine_similarity(emb_lora, emb_direction)

    fig_path = plot_cosine_similarity_heatmap(sim_matrix, arr_direction, arr_lora, ckpt, output_fig_dir)
    print(f"Saved cosine matrix plot to {fig_path}")

    similarity_score = set_similarity_hungarian(sim_matrix)

    max_idx = np.unravel_index(sim_matrix.argmax(), sim_matrix.shape)
    min_idx = np.unravel_index(sim_matrix.argmin(), sim_matrix.shape)

    max_score = sim_matrix[max_idx]
    min_score = sim_matrix[min_idx]

    results.append({
        "ckpt": ckpt,
        "similarity_score": float(similarity_score),
        "max_score": float(max_score),
        "max_lora": arr_lora[max_idx[0]],
        "max_direction": interp_direction[max_idx[1]]["theme"],
        "min_score": float(min_score),
        "min_lora": arr_lora[min_idx[0]],
        "min_direction": interp_direction[min_idx[1]]["theme"],
    })


results.sort(key=lambda x: x["similarity_score"], reverse=True)
output_file = r"../direction_rank.json"
with open(output_file, "w") as f:
    json.dump(results, f, indent=4)


