# Clones TripoSR into vendor/ and installs the ai extra (torch with CUDA 12.4). Run from the repo root.
$ErrorActionPreference = "Stop"
if (-not (Test-Path vendor/TripoSR)) { git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR }
uv sync --extra ai
# PyPI ships a CPU-only torch on Windows. TripoSR needs CUDA locally; without it, hf3d falls back to the
# Hugging Face Space. The CUDA build is a 2.4 GB download, so install it separately and retry if it drops:
#   uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv run python -c "import torch; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
