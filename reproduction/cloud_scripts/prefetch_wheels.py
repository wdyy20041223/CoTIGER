"""Cache pip's already-resolved wheels from official PyPI with byte ranges."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import urllib.request

from pip._vendor.packaging.utils import parse_wheel_filename
from download_data import ranged_download

BASE=Path('/root/genrec-reproduction')
CACHE=BASE/'wheelhouse'
CACHE.mkdir(exist_ok=True)
installed={d.metadata['Name'].lower().replace('_','-'):d.version for d in
           importlib.metadata.distributions(path=[str(BASE/'.venv/lib/python3.10/site-packages')])}
metadata_files=list(Path('/tmp').glob('pip-unpack-*/*.whl.metadata'))
downloaded=list(Path('/tmp').glob('pip-unpack-*/*.whl'))
tasks={}
for path in metadata_files:
    filename=path.name.removesuffix('.metadata')
    name,version,_,_=parse_wheel_filename(filename)
    if installed.get(name)==str(version):
        continue
    tasks[filename]=(name,str(version),filename)
assert len(tasks)>30,'Resolved pip metadata is incomplete'

def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as file:
        while chunk:=file.read(4*1024*1024):
            result.update(chunk)
    return result.hexdigest()

def fetch(task):
    name,version,filename=task
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json',timeout=60) as response:
        metadata=json.load(response)
    record=next(file for file in metadata['urls'] if file['filename']==filename)
    assert record['url'].startswith('https://files.pythonhosted.org/')
    destination=CACHE/filename
    expected=record['digests']['sha256']
    if not destination.exists():
        for existing in downloaded:
            if existing.name==filename and existing.stat().st_size==record['size'] and digest(existing)==expected:
                shutil.copyfile(existing,destination)
                print('Reused complete pip download:',filename,flush=True)
                break
    if not destination.exists():
        partial=destination.with_suffix('.whl.part')
        ranged_download(record['url'],partial)
        assert digest(partial)==expected,'Wheel SHA256 mismatch'
        partial.replace(destination)
    assert destination.stat().st_size==record['size']
    assert digest(destination)==expected
    print('Official wheel verified:',filename,record['size'],flush=True)
    return {'name':name,'version':version,'filename':filename,'bytes':record['size'],
            'sha256':expected,'source':record['url']}

if __name__=='__main__':
    print('Pending resolved wheels:',len(tasks),flush=True)
    status={'state':'downloading'}
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            files=list(pool.map(fetch,tasks.values()))
        status={'state':'ready','files':files}
    except Exception as error:
        status={'state':'failed','error':str(error)}
        raise
    finally:
        (BASE/'manifests/wheelhouse-status.json').write_text(json.dumps(status,indent=2))
