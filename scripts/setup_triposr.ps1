# Clones TripoSR into vendor/ and installs the ai extra (torch with CUDA 12.4). Run from the repo root.
$ErrorActionPreference = "Stop"
if (-not (Test-Path vendor/TripoSR)) { git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR vendor/TripoSR }
uv sync --extra ai
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
