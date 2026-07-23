#!/usr/bin/env bash
set -euo pipefail

python3 -m venv --system-site-packages embed-env
source embed-env/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  'sentence-transformers==5.2.3' \
  'transformers==5.3.0' \
  'numpy==1.26.4' \
  'scipy==1.13.1' \
  'scikit-learn==1.5.2' \
  'Pillow>=10.0' \
  'psutil>=5.9'

python - <<'PY'
import torch
import sentence_transformers
import transformers

print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("sentence-transformers:", sentence_transformers.__version__)
print("transformers:", transformers.__version__)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available")
PY
