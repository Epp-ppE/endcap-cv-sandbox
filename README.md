# endcap-cv-sandbox

Sandbox for the CV part of the AI End-Cap Intelligence project: measure shopper pass-by traffic, stops, dwell and engagement at store end-caps from CCTV. Built and tested on public / proxy footage before real store footage is available.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# Install torch with CUDA first (pick the command for your CUDA version at https://pytorch.org)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pytest
```

## Layout

```
configs/    default.yaml + per-experiment overrides
endcap/     pipeline package (one module per step)
tests/      pytest tests
data/       input videos and labels (gitignored)
outputs/    run results (gitignored)
```
