# stupid-search

an unsupervised method to search for steerable directions in language model embedding space.

## What's here

- `dataset_code/` - Scripts to generate the "situations begetting honesty" dataset 
- `train/` - Training notebooks for the divergence LoRA adapters (phase 1 is the main work, phase 2 was a short experiment training for 2 more epochs that I scrapped)
- `directions_code/` - Extract direction vectors from the trained LoRAs using mean activation differences
- `test_directions/` - Test if the extracted directions are actually linear and steerable (empathy, prescriptiveness, follow-up questions, etc.)
- `test_lora/` - Test the original LoRA adapters before direction extraction
- `model_outputs/` - Raw model responses from all experiments
- `figures/` - Plots and visualizations (cosine similarity matrices, quantitative evaluations)
- `datasets/` - The conversation scenarios designed to escape assistant-like behavior

## Running stuff

Most work happens in Jupyter notebooks. Main training is `train/train_phase1.ipynb`.

Dependencies are in `pyproject.toml` - just `uv sync` to get everything.
