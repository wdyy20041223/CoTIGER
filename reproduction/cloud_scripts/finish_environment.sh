#!/usr/bin/env bash
set -Eeuo pipefail
base=/root/genrec-reproduction
trap 'rc=$?; printf "{\"state\":\"failed\",\"exit_code\":%s}\n" "$rc" > "$base/manifests/environment-status.json"' ERR
printf '{"state":"installing_verified_wheel_cache"}\n' > "$base/manifests/environment-status.json"
"$base/.venv/bin/python" -m pip --isolated install --no-index --find-links "$base/wheelhouse" -r "$base/genrec/requirements.txt" sentencepiece
"$base/.venv/bin/python" -m pip --isolated install --no-deps --no-build-isolation -e "$base/genrec"
"$base/.venv/bin/python" -m pip check
"$base/.venv/bin/python" -m pip freeze > "$base/manifests/pip-freeze.txt"
"$base/.venv/bin/python" - <<'PY'
import json, platform, torch
from pathlib import Path
from genrec.models.tiger import Tiger
from genrec.models.rqvae import RqVae
assert torch.cuda.is_available(), 'CUDA unavailable'
x=torch.randn(128,128,device='cuda')
assert torch.isfinite(x@x).all()
result={'state':'ready','python':platform.python_version(),'torch':torch.__version__,
        'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),
        'bf16_supported':torch.cuda.is_bf16_supported(including_emulation=False),
        'bf16_emulation_supported':torch.cuda.is_bf16_supported()}
Path('/root/genrec-reproduction/manifests/environment-status.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
PY
