# Pretrained PixArt-α concept checkpoints

The four PixArt-α concept vectors (cartoon, Van Gogh, male, female) are too large to ship in the GitHub repository (≈315 MB each). They are hosted on the Hugging Face Hub instead.

> **Repo:** [`Mattsh21/responsible-t2i-diffusion`](https://huggingface.co/Mattsh21/responsible-t2i-diffusion) on Hugging Face Hub.

## What's in this folder

This folder is intentionally empty in the GitHub repo. After running one of the download options below, you'll have:

```
checkpoints/
├── README.md                         (this file)
├── external_concept_cartoon.pt       (~315 MB)
├── external_concept_van_gogh.pt      (~315 MB)
├── external_concept_male.pt          (~315 MB)
└── external_concept_female.pt        (~315 MB)
```

The files are git-ignored (see top-level `.gitignore`), so they will not be accidentally committed.

## Option A — From Python (recommended)

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Mattsh21/responsible-t2i-diffusion",
    allow_patterns="*.pt",
    local_dir="checkpoints",
)
```

This downloads only the `.pt` files into `checkpoints/`, with progress bars and resume support.

To grab a single concept:

```python
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="Mattsh21/responsible-t2i-diffusion",
    filename="external_concept_female.pt",
    local_dir="checkpoints",
)
print(path)  # checkpoints/external_concept_female.pt
```

## Option B — From the command line

```bash
pip install huggingface_hub
huggingface-cli download Mattsh21/responsible-t2i-diffusion \
    --include "*.pt" \
    --local-dir checkpoints
```

## Option C — In the Colab demo notebook

The notebook at `notebooks/pixart_external_heads_demo.ipynb` handles the download automatically — no manual step needed.

## Verifying the download

Each `.pt` is a `torch.save()` of the trained concept module's state dict. To confirm:

```python
import torch
state = torch.load("checkpoints/external_concept_female.pt", map_location="cpu")
print(type(state), len(state), "tensors")
print("Example keys:", list(state.keys())[:3])
```

You should see entries of the form `layer_<L>_head_<H>` (or `external_heads.layer_<L>_head_<H>` — both forms are handled by the loader).
