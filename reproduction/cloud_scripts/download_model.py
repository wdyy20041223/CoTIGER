"""Download a pinned Sentence-T5 model using original Hugging Face CDN URLs."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

BASE = Path('/root/genrec-reproduction')
DEST = BASE/'models_hub/sentence-t5-xl'
DEST.mkdir(parents=True,exist_ok=True)
manifest = json.loads((BASE/'sentence-t5-xl-manifest.json').read_text())
try:
    with zipfile.ZipFile(BASE/'sentence-t5-xl-small.zip') as archive:
        archive.extractall(DEST)
    for record in manifest['files']:
        path = DEST/record['name']
        path.parent.mkdir(parents=True,exist_ok=True)
        if not record.get('in_archive') and not path.exists():
            partial=path.with_suffix(path.suffix+'.part')
            print(f'Downloading {record["name"]}: {record["bytes"]} bytes from official CDN',flush=True)
            subprocess.run(['curl','--fail','--location','--silent','--show-error','--connect-timeout','15',
                            '--max-time','7200','--retry','4','--retry-all-errors','--continue-at','-',
                            '--output',str(partial),record['url']],check=True)
            partial.replace(path)
        digest=hashlib.sha256()
        with path.open('rb') as inp:
            while chunk:=inp.read(4*1024*1024):
                digest.update(chunk)
        assert path.stat().st_size==record['bytes'], record['name']+' size mismatch'
        assert digest.hexdigest()==record['sha256'], record['name']+' SHA256 mismatch'
        print(f'Verified {record["name"]}',flush=True)
    status={'state':'ready','model':manifest['model'],'revision':manifest['revision'],
            'files':[{k:v for k,v in f.items() if k not in {'url'}} for f in manifest['files']]}
except Exception as error:
    status={'state':'failed','error':str(error)}
    raise
finally:
    (BASE/'manifests/model-status.json').write_text(json.dumps(status,indent=2))
