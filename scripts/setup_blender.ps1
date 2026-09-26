# Installs bpy 4.2 into vendor/bpy-env (about 300 MB), separate from the app's own .venv: bpy is Python 3.11
# only and may clash on numpy with the app's dependencies. Run from the repo root.
$ErrorActionPreference = "Stop"
uv venv vendor/bpy-env --python 3.11
uv pip install --python vendor/bpy-env/Scripts/python.exe "bpy==4.2.23"
