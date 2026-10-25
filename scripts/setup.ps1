# One-shot Windows dependency setup.
#
# Prerequisite: Python 3.11 venv created AND activated, e.g.:
#   py -3.11 -m venv .venv
#   .venv\Scripts\activate
#
# Then run from the project root:
#   .\scripts\setup.ps1
#
# This installs everything in the correct order. The tricky bit is that
# resemble-enhance depends on deepspeed==0.12.4, which cannot build on Windows
# without the CUDA toolkit + a C++ compiler. We never use deepspeed (only
# resemble-enhance *inference*, a plain torch forward pass), so we install a
# tiny inference-only deepspeed stub that satisfies the dependency instead.

$ErrorActionPreference = "Stop"

Write-Host "==> 1/5  Build tooling (setuptools < 70 for torch 2.1.1's pkg_resources.packaging)"
python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" wheel

Write-Host "==> 2/5  numpy 1.26.x (torch 2.1.1 is incompatible with numpy 2.x) + CUDA torch"
python -m pip install "numpy==1.26.4"
# cu121 wheels are compatible with the laptop RTX 4050 driver (CUDA UMD 13.3).
python -m pip install torch==2.1.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

Write-Host "==> 3/5  deepspeed inference-only stub (no CUDA toolkit / compiler needed)"
python -m pip install ".\stubs\deepspeed-stub"

Write-Host "==> 4/5  resemble-enhance WITHOUT its deps (deepspeed already satisfied by the stub)"
python -m pip install "resemble-enhance==0.0.1" --no-deps

Write-Host "==> 5/5  Everything else"
python -m pip install -r requirements.txt

Write-Host ""
Write-Host "==> Verifying imports..."
python -c "import torch; assert torch.cuda.is_available(); from resemble_enhance.enhancer.inference import enhance; import faster_whisper, gradio, groq, rapidfuzz; print('All imports OK. CUDA:', torch.cuda.get_device_name(0))"
Write-Host ""
Write-Host "Setup complete. Run the app with:  python app.py"
